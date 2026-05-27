"""
Lambda handler for AI-powered CSS regeneration with Ably real-time status updates.

Processes SQS records containing regeneration requests, reads original CSS from S3,
calls OpenAI to regenerate CSS, saves results to S3, and publishes status updates
via the Ably real-time messaging service.

Sequence ordering
-----------------
Each SQS record initialises its sequence counter from the DynamoDB item written
by the crawler lambda (via get_current_sequence).  This ensures the AI phase's
events are globally ordered *after* the crawler phase's events on the same Ably
channel.  The frontend subscriber deduplicates by sequence number, so without
this initialisation all AI events (sequences 1–N) would be discarded because the
crawler already published sequences 1–M.

Environment Variables:
  - SECRET_NAME:      AWS Secrets Manager secret ID containing OpenAI API key.
  - BUCKET_NAME:      S3 bucket containing original CSS files.
  - DYNAMODB_TABLE_NAME: DynamoDB table for status persistence and sequence reads.
  - ABLY_SECRET_NAME: AWS Secrets Manager secret ID containing Ably API key.
  - RESULT_BASE_URL:  (optional) Base URL for generating result links in status updates.
"""

import json
import os

import boto3
from openai import OpenAI
# get_current_sequence reads the crawler's last persisted sequence from DynamoDB
# so the AI phase continues the global counter rather than restarting at 1.
from status_publisher import get_current_sequence, publish_status_update

s3 = boto3.client("s3")
secrets_client = boto3.client("secretsmanager")


def _get_openai_client():
    """Initialize OpenAI client with API key from AWS Secrets Manager."""
    secret = secrets_client.get_secret_value(SecretId=os.environ["SECRET_NAME"])
    api_key = json.loads(secret["SecretString"])["OpenAIAPIKey"]
    return OpenAI(api_key=api_key)


openai_client = _get_openai_client()
BUCKET = os.environ["BUCKET_NAME"]


def lambda_handler(event, context):
    """
    Process CSS regeneration requests from SQS.

    Expected SQS message body:
      {
        "RegeneratedWebsiteId": "website-123",
        "RegeneratedWebsiteUrl": "https://example.com",
        "RegenerationTheme": "modern-dark"  # optional
      }

    Args:
        event: SQS event containing records with regeneration requests.
        context: Lambda context object.

    Returns:
        dict: HTTP response with statusCode and body.
              200 on success, 500 on failure.
    """
    for record in event["Records"]:
        body = json.loads(record["body"])
        website_id = body["RegeneratedWebsiteId"]
        url = body.get("RegeneratedWebsiteUrl", "")
        theme = body.get("RegenerationTheme", "")

        # Seed the AI phase's sequence counter from the crawler's last persisted
        # value so the AI events form a continuous stream with the crawler events.
        # get_current_sequence degrades gracefully to 0 if the read fails.
        seq = get_current_sequence(website_id, url)

        def publish(step, status, message, result_url=None, error=None):
            """Publish a status update with auto-incrementing sequence number."""
            nonlocal seq
            seq += 1
            publish_status_update(
                website_id=website_id,
                # website_url is the DynamoDB sort key — required by the confirmed
                # composite key schema (RegeneratedWebsiteId + RegeneratedWebsiteUrl).
                website_url=url,
                phase="ai",
                step=step,
                status=status,
                message=message,
                sequence=seq,
                result_url=result_url,
                error=error,
            )

        try:
            publish("received", "processing", "AI regeneration request received")

            publish("reading_source_css", "processing", "Reading original CSS from S3")
            obj = s3.get_object(Bucket=BUCKET, Key=f"{website_id}/original-styles.css")
            original_css = obj["Body"].read().decode("utf-8")

            publish("building_prompt", "processing", "Building OpenAI prompt")
            if theme:
                theme_prompt = f"Please regenerate the CSS to apply this theme: {theme}"
            else:
                theme_prompt = "No specific theme provided. Please regenerate the CSS with modern best practices."

            publish("calling_openai", "processing", "Calling OpenAI to regenerate CSS")
            response = openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a CSS and web design expert."},
                    {"role": "user", "content": f"{theme_prompt}\n\nOriginal CSS:\n{original_css}"},
                ],
            )
            regenerated_css = response.choices[0].message.content

            publish("saving_regenerated_css", "processing", "Saving regenerated CSS to S3")
            output_key = f"{website_id}/Regenerated-Styles.css"
            s3.put_object(
                Bucket=BUCKET,
                Key=output_key,
                Body=regenerated_css,
                ContentType="text/css",
            )

            result_base = os.environ.get("RESULT_BASE_URL", "").rstrip("/")
            result_url = f"{result_base}/{website_id}/index.html" if result_base else None

            publish("completed", "completed", "Website regeneration complete", result_url=result_url)

        except Exception as e:
            publish("failed", "failed", f"AI regeneration failed: {e}", error=str(e))
            return {"statusCode": 500, "body": json.dumps({"error": str(e)})}

    return {"statusCode": 200, "body": json.dumps({"message": "success"})}
