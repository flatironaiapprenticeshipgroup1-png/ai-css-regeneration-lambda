"""
Status publisher module for the AI CSS Regeneration Lambda.

This module provides functionality to publish status updates about CSS regeneration
progress to Ably real-time channels and persist the status to DynamoDB.

DynamoDB key schema (confirmed):
  Partition key: RegeneratedWebsiteId (String)
  Sort key:      RegeneratedWebsiteUrl (String)

Both publish_status_update() and get_current_sequence() must supply both key
attributes.  The website_url is available from the SQS message body and must
be passed in by the caller (handler.py).
"""
import asyncio  # Required: ably-python >= 2.0.0 channel.publish() is a coroutine
import json
import os
from datetime import datetime, timezone

import boto3
from ably import AblyRest

_secrets_client = None
_ably_client = None
_dynamodb_client = None
_ably_api_key = None


def _get_secrets_client():
    """Get or create a boto3 Secrets Manager client (cached as singleton)."""
    global _secrets_client
    if _secrets_client is None:
        _secrets_client = boto3.client("secretsmanager")
    return _secrets_client


def _get_ably_api_key():
    """
    Retrieve the Ably API key from AWS Secrets Manager.

    The key is cached to avoid repeated secret manager calls.
    Expects environment variable ABLY_SECRET_NAME to be set.

    Returns:
        str: The Ably API key
    """
    global _ably_api_key
    if _ably_api_key is None:
        secret_name = os.environ["ABLY_SECRET_NAME"]
        response = _get_secrets_client().get_secret_value(SecretId=secret_name)
        _ably_api_key = json.loads(response["SecretString"])["AblyApiKey"]
    return _ably_api_key


def _get_ably_client():
    """Get or create an Ably REST client (cached as singleton)."""
    global _ably_client
    if _ably_client is None:
        _ably_client = AblyRest(key=_get_ably_api_key())
    return _ably_client


def _get_dynamodb_client():
    """Get or create a boto3 DynamoDB client (cached as singleton)."""
    global _dynamodb_client
    if _dynamodb_client is None:
        _dynamodb_client = boto3.client("dynamodb")
    return _dynamodb_client


def get_current_sequence(website_id: str, website_url: str) -> int:
    """
    Read the latest persisted sequence number for this website from DynamoDB.

    The crawler lambda writes CurrentSequence to the item after each status
    publish.  The AI lambda calls this at the start of each record to
    initialise its own counter at N so the first AI publish becomes N+1,
    producing a globally ordered stream across both phases.

    Falls back to 0 on any error (DynamoDB unavailable, item not found, etc.)
    so the AI lambda still publishes rather than crashing — the frontend will
    show the events, just without gap-free ordering relative to the crawler.

    Args:
        website_id:  Partition key of the DynamoDB item.
        website_url: Sort key of the DynamoDB item (confirmed composite schema).

    Returns:
        int: The last sequence number written by the crawler, or 0 if unknown.
    """
    try:
        response = _get_dynamodb_client().get_item(
            TableName=os.environ["DYNAMODB_TABLE_NAME"],
            Key={
                # Composite key — both attributes required by the table schema
                "RegeneratedWebsiteId": {"S": website_id},
                "RegeneratedWebsiteUrl": {"S": website_url},
            },
            # Only fetch the one attribute we need to minimise read cost
            ProjectionExpression="CurrentSequence",
        )
        item = response.get("Item", {})
        return int(item.get("CurrentSequence", {}).get("N", "0"))
    except Exception as exc:
        # Log and degrade gracefully: AI events will still be published,
        # they may just overlap sequence numbers with crawler events.
        print(f"[status_publisher] WARNING: could not read CurrentSequence "
              f"for {website_id}: {exc}. Starting AI sequence from 0.")
        return 0


def publish_status_update(
    website_id: str,
    website_url: str,
    phase: str,
    step: str,
    status: str,
    message: str,
    sequence: int,
    publisher: str = "ai-css-regeneration-lambda",
    result_url: str = None,
    error: str = None,
) -> dict:
    """
    Publish a CSS regeneration status update to Ably and persist to DynamoDB.

    This function sends a status update via Ably's real-time messaging to notify
    subscribers of progress, and updates the website record in DynamoDB.

    Args:
        website_id:  Partition key — unique identifier for the website being regenerated.
        website_url: Sort key — URL of the website (required by composite DynamoDB schema).
        phase:       Current phase of regeneration (e.g., 'ai').
        step:        Current step within the phase.
        status:      Status value (e.g., 'processing', 'completed', 'failed').
        message:     Human-readable status message.
        sequence:    Globally ordered sequence number (continues from crawler events).
        publisher:   Source identifier (default: "ai-css-regeneration-lambda").
        result_url:  Optional URL to the regeneration result.
        error:       Optional error message if status is 'failed'.

    Returns:
        dict: The payload that was published containing all status information.
    """
    timestamp = datetime.now(timezone.utc).isoformat()

    payload = {
        "websiteId": website_id,
        "phase": phase,
        "step": step,
        "status": status,
        "message": message,
        "sequence": sequence,
        "publisher": publisher,
        "timestamp": timestamp,
        "resultUrl": result_url,
        "error": error,
    }

    channel = _get_ably_client().channels.get(f"regeneration:{website_id}")
    # ably-python >= 2.0.0: publish() is a coroutine; asyncio.run() is required
    # in a synchronous Lambda context to actually send the message.
    asyncio.run(channel.publish("regeneration-status", payload))

    update_expr = (
        "SET CurrentPhase = :phase, CurrentStep = :step, "
        "RegenerationStatus = :status, CurrentSequence = :seq, "
        "LastUpdatedAt = :ts"
    )
    expr_vals = {
        ":phase": {"S": phase},
        ":step": {"S": step},
        ":status": {"S": status},
        ":seq": {"N": str(sequence)},
        ":ts": {"S": timestamp},
    }

    if result_url is not None:
        update_expr += ", ResultUrl = :result_url"
        expr_vals[":result_url"] = {"S": result_url}

    if error is not None:
        update_expr += ", ErrorMessage = :error"
        expr_vals[":error"] = {"S": error}

    _get_dynamodb_client().update_item(
        TableName=os.environ["DYNAMODB_TABLE_NAME"],
        Key={
            # Composite key — both attributes are required by the confirmed table schema:
            #   Partition key: RegeneratedWebsiteId (String)
            #   Sort key:      RegeneratedWebsiteUrl (String)
            "RegeneratedWebsiteId": {"S": website_id},
            "RegeneratedWebsiteUrl": {"S": website_url},
        },
        UpdateExpression=update_expr,
        ExpressionAttributeValues=expr_vals,
    )

    return payload
