"""
Test suite for the AI CSS Regeneration Lambda handler.

Tests verify:
1. Happy path: all expected steps published with correct sequence numbers
2. OpenAI failure: batchItemFailures returned and "failed" event published
3. S3 write failure: batchItemFailures returned and "failed" event published
4. Sequence numbers: always strictly increasing within a single invocation
5. Sequence continuation: AI sequences start at crawler_last_seq + 1
6. Phase: all published events use phase="ai" not "crawler"

Important test-infrastructure notes
------------------------------------
- Every test imports lambda_function AND calls lambda_handler() INSIDE the
  same patch with-block. This is required because status_publisher.py lazily
  initialises its boto3 / Ably singletons on first use; if those singletons
  are created outside the patch context they hit real AWS and raise
  NoCredentialsError.

- There are two separate DynamoDB surfaces:
    * boto3.client("dynamodb") — used by status_publisher (low-level DynamoDB
      JSON API) for get_current_sequence() and update_item per publish.
    * boto3.resource("dynamodb").Table(...) — used by lambda_function itself
      (high-level API) for idempotency guard and status field updates.

- mock_ably_channel.publish is an AsyncMock. status_publisher wraps the
  Ably publish call with asyncio.run(), which requires the mock to return an
  awaitable coroutine rather than a plain MagicMock object.

- mock_dynamodb_client.get_item returns {} by default (no Item key) so
  get_current_sequence() resolves to 0 and sequences start at 1.
  Tests that verify continuation override this to return a real Item.

- The OpenAI mock response must include finish_reason and usage attributes
  because lambda_function logs these values with %d and %.6f format strings.
"""

import json
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

os.environ.setdefault("BUCKET_NAME", "test-bucket")
os.environ.setdefault("SECRET_NAME", "test-secret")
os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")
os.environ["DYNAMODB_TABLE_NAME"] = "test-table"
os.environ["ABLY_SECRET_NAME"] = "test/ably-secret"

WEBSITE_ID = "test-123"
URL = "https://example.com"

# boto3 and OpenAI are called at module level in lambda_function.py, so we must
# patch before importing — otherwise the real AWS/OpenAI calls fire on import.
_mock_s3 = MagicMock()
_mock_secrets = MagicMock()
_mock_secrets.get_secret_value.return_value = {
    "SecretString": json.dumps({"OpenAIAPIKey": "test-key"})
}
_mock_openai_client = MagicMock()
_mock_dynamodb_resource = MagicMock()


def _boto3_client_factory(service, **_):
    return _mock_s3 if service == "s3" else _mock_secrets


with patch("boto3.client", side_effect=_boto3_client_factory), patch(
    "boto3.resource", return_value=_mock_dynamodb_resource
), patch("openai.OpenAI", return_value=_mock_openai_client):
    from lambda_function import lambda_handler


def make_event(website_id=WEBSITE_ID, url=URL, theme="cyberpunk"):
    return {
        "Records": [
            {
                "body": json.dumps(
                    {
                        "RegeneratedWebsiteId": website_id,
                        "RegeneratedWebsiteUrl": url,
                        "RegenerationTheme": theme,
                    }
                )
            }
        ]
    }


def make_mocks():
    """
    Build and return all mock objects needed by one test.

    DynamoDB surfaces
    -----------------
    mock_dynamodb_client: boto3.client("dynamodb") used by status_publisher
      (low-level DynamoDB JSON API). Controls get_current_sequence() and the
      update_item calls made on each publish. Returns {} (no Item) by default
      so get_current_sequence() returns 0 and sequences start at 1. Override
      mock_dynamodb_client.get_item.return_value in specific tests to simulate
      a crawler having already written a CurrentSequence value.

    mock_table: the Table object from boto3.resource("dynamodb").Table(...)
      used by lambda_function for idempotency checks and status field updates.
      Returns {} by default so the idempotency guard does not skip processing.

    AsyncMock for channel.publish
    -----------------------------
    ably-python >= 2.0.0 makes channel.publish() a coroutine.
    status_publisher wraps it with asyncio.run(), so the mock must return
    an awaitable. AsyncMock satisfies this requirement.

    Returns:
        tuple: (mock_s3, mock_dynamodb_client, mock_table, mock_dynamodb_resource,
                mock_ably_channel, mock_ably_rest, mock_openai, boto3_client_factory)
    """
    mock_s3 = MagicMock()
    mock_s3.get_object.return_value = {
        "Body": MagicMock(read=lambda: b"body { color: red; }")
    }

    # Low-level DynamoDB client used by status_publisher
    mock_dynamodb_client = MagicMock()
    mock_dynamodb_client.get_item.return_value = {}

    # High-level DynamoDB resource Table used by lambda_function
    mock_table = MagicMock()
    mock_table.get_item.return_value = {}
    mock_dynamodb_resource = MagicMock()
    mock_dynamodb_resource.Table.return_value = mock_table

    mock_secrets = MagicMock()
    mock_secrets.get_secret_value.side_effect = lambda SecretId: {
        "test-secret": {
            "SecretString": json.dumps({"OpenAIAPIKey": "fake-openai-key"})
        },
        "test/ably-secret": {
            "SecretString": json.dumps({"AblyApiKey": "fake-ably-key"})
        },
    }[SecretId]

    def boto3_client_factory(service, **_):
        return {
            "s3": mock_s3,
            "dynamodb": mock_dynamodb_client,
            "secretsmanager": mock_secrets,
        }[service]

    mock_openai = MagicMock()
    # finish_reason and usage must be concrete values — lambda_function logs them
    # with %d and %.6f format strings which fail against MagicMock objects.
    mock_openai.chat.completions.create.return_value = MagicMock(
        choices=[
            MagicMock(
                message=MagicMock(content="body { color: neon; }"),
                finish_reason="stop",
            )
        ],
        usage=MagicMock(prompt_tokens=100, completion_tokens=200),
    )

    mock_ably_channel = MagicMock()
    mock_ably_channel.publish = AsyncMock()
    mock_ably_rest = MagicMock()
    mock_ably_rest.channels.get.return_value = mock_ably_channel

    return (
        mock_s3,
        mock_dynamodb_client,
        mock_table,
        mock_dynamodb_resource,
        mock_ably_channel,
        mock_ably_rest,
        mock_openai,
        boto3_client_factory,
    )


def _clear_modules():
    """Remove cached lambda_function/status_publisher so each test gets a fresh import."""
    for mod in ["lambda_function", "status_publisher"]:
        if mod in sys.modules:
            del sys.modules[mod]


# Expected publish steps for a single-chunk CSS (the default b"body { color: red; }" mock).
# The mock CSS is one rule block, so split_css_into_chunks returns one chunk, producing
# exactly one "regenerating_css_chunks_completed" event.
EXPECTED_STEPS = [
    "chunking",
    "regenerating_css",
    "regenerating_css_chunks_completed",
    "Finalizing",
]


def test_happy_path_publishes_all_steps():
    """
    Verify that a successful run publishes all expected steps in order with
    monotonically increasing sequence numbers, writes the CSS to S3, and
    updates DynamoDB status via both the resource table and the publisher client.
    """
    (
        mock_s3,
        mock_dynamodb_client,
        mock_table,
        mock_dynamodb_resource,
        mock_channel,
        mock_ably_rest,
        mock_openai,
        boto3_client_factory,
    ) = make_mocks()
    _clear_modules()

    with patch("boto3.client", side_effect=boto3_client_factory), patch(
        "boto3.resource", return_value=mock_dynamodb_resource
    ), patch("ably.AblyRest", return_value=mock_ably_rest), patch(
        "openai.OpenAI", return_value=mock_openai
    ):
        import lambda_function

        result = lambda_function.lambda_handler(make_event(), {})

    assert result == {"batchItemFailures": []}

    steps = [c.args[1]["step"] for c in mock_channel.publish.call_args_list]
    assert steps == EXPECTED_STEPS, f"Unexpected steps: {steps}"

    seqs = [c.args[1]["sequence"] for c in mock_channel.publish.call_args_list]
    # With no prior crawler sequence (get_item returns {}), starts at 1
    assert seqs == list(
        range(1, len(EXPECTED_STEPS) + 1)
    ), f"Unexpected sequences: {seqs}"

    last = mock_channel.publish.call_args_list[-1].args[1]
    assert last["status"] == "completed"

    assert mock_s3.put_object.call_count == 1
    key = mock_s3.put_object.call_args.kwargs["Key"]
    assert key == f"{WEBSITE_ID}/Regenerated-Styles.css"

    # status_publisher's update_item called once per Ably publish
    assert mock_dynamodb_client.update_item.call_count == len(EXPECTED_STEPS)

    # lambda_function's table update_item called for "processing" + "completed"
    assert mock_table.update_item.call_count == 2

    print("test_happy_path_publishes_all_steps: PASSED")


def test_openai_failure_publishes_failed():
    """
    Verify that an OpenAI error causes batchItemFailures to be returned and a
    "failed" Ably event to be published.
    Steps published before the failure (chunking, regenerating_css) are present;
    Finalizing is not published since the error short-circuits the handler.
    """
    (
        _,
        _,
        _,
        mock_dynamodb_resource,
        mock_channel,
        mock_ably_rest,
        mock_openai,
        boto3_client_factory,
    ) = make_mocks()
    mock_openai.chat.completions.create.side_effect = Exception("OpenAI error")
    _clear_modules()

    with patch("boto3.client", side_effect=boto3_client_factory), patch(
        "boto3.resource", return_value=mock_dynamodb_resource
    ), patch("ably.AblyRest", return_value=mock_ably_rest), patch(
        "openai.OpenAI", return_value=mock_openai
    ):
        import lambda_function

        result = lambda_function.lambda_handler(make_event(), {})

    assert result == {"batchItemFailures": [{"itemIdentifier": WEBSITE_ID}]}

    steps = [c.args[1]["step"] for c in mock_channel.publish.call_args_list]
    assert "chunking" in steps
    assert "regenerating_css" in steps
    assert "Finalizing" not in steps
    assert "failed" in steps

    failed_event = next(c.args[1] for c in mock_channel.publish.call_args_list if c.args[1]["step"] == "failed")
    assert failed_event["status"] == "failed"
    assert failed_event["error"] is not None

    print("test_openai_failure_publishes_failed: PASSED")


def test_s3_write_failure_publishes_failed():
    """
    Verify that an S3 write error causes batchItemFailures to be returned and a
    "failed" Ably event to be published.
    The CSS chunks are regenerated successfully before the write fails, so
    regenerating_css_chunks_completed is published but Finalizing is not.
    """
    (
        mock_s3,
        _,
        _,
        mock_dynamodb_resource,
        mock_channel,
        mock_ably_rest,
        mock_openai,
        boto3_client_factory,
    ) = make_mocks()
    mock_s3.put_object.side_effect = Exception("S3 write error")
    _clear_modules()

    with patch("boto3.client", side_effect=boto3_client_factory), patch(
        "boto3.resource", return_value=mock_dynamodb_resource
    ), patch("ably.AblyRest", return_value=mock_ably_rest), patch(
        "openai.OpenAI", return_value=mock_openai
    ):
        import lambda_function

        result = lambda_function.lambda_handler(make_event(), {})

    assert result == {"batchItemFailures": [{"itemIdentifier": WEBSITE_ID}]}

    steps = [c.args[1]["step"] for c in mock_channel.publish.call_args_list]
    assert "Finalizing" not in steps
    assert "failed" in steps

    print("test_s3_write_failure_publishes_failed: PASSED")


def test_sequence_numbers_always_increase():
    """
    Verify that all published events carry strictly increasing sequence numbers
    with no gaps or duplicates within a single invocation.
    Frontend deduplication relies on this.
    """
    (
        _,
        _,
        _,
        mock_dynamodb_resource,
        mock_channel,
        mock_ably_rest,
        mock_openai,
        boto3_client_factory,
    ) = make_mocks()
    _clear_modules()

    with patch("boto3.client", side_effect=boto3_client_factory), patch(
        "boto3.resource", return_value=mock_dynamodb_resource
    ), patch("ably.AblyRest", return_value=mock_ably_rest), patch(
        "openai.OpenAI", return_value=mock_openai
    ):
        import lambda_function

        lambda_function.lambda_handler(make_event(), {})

    seqs = [c.args[1]["sequence"] for c in mock_channel.publish.call_args_list]
    assert seqs == sorted(seqs) and len(seqs) == len(
        set(seqs)
    ), f"Sequences not strictly increasing: {seqs}"

    print("test_sequence_numbers_always_increase: PASSED")


def test_sequence_continues_from_crawler():
    """
    Verify that the AI lambda reads CurrentSequence from DynamoDB and starts
    its own counter at crawler_last_seq + 1, producing a globally ordered
    event stream that the frontend will not discard.

    Without this behaviour the AI lambda would restart at sequence 1, which
    the frontend deduplicates away (it already saw sequences 1–N from the
    crawler), silently hiding all AI progress updates.
    """
    (
        _,
        mock_dynamodb_client,
        _,
        mock_dynamodb_resource,
        mock_channel,
        mock_ably_rest,
        mock_openai,
        boto3_client_factory,
    ) = make_mocks()

    # Simulate the crawler having published 6 events (sequences 1–6).
    # The AI lambda should start at 7.
    CRAWLER_LAST_SEQ = 6
    mock_dynamodb_client.get_item.return_value = {
        "Item": {
            "RegeneratedWebsiteId": {"S": WEBSITE_ID},
            "RegeneratedWebsiteUrl": {"S": URL},
            "CurrentSequence": {"N": str(CRAWLER_LAST_SEQ)},
        }
    }
    _clear_modules()

    with patch("boto3.client", side_effect=boto3_client_factory), patch(
        "boto3.resource", return_value=mock_dynamodb_resource
    ), patch("ably.AblyRest", return_value=mock_ably_rest), patch(
        "openai.OpenAI", return_value=mock_openai
    ):
        import lambda_function

        result = lambda_function.lambda_handler(make_event(), {})

    assert result == {"batchItemFailures": []}

    seqs = [c.args[1]["sequence"] for c in mock_channel.publish.call_args_list]

    # First AI event must immediately follow the last crawler event
    assert (
        seqs[0] == CRAWLER_LAST_SEQ + 1
    ), f"AI sequence should start at {CRAWLER_LAST_SEQ + 1}, got {seqs[0]}"

    # All AI sequences must be strictly greater than the crawler's last sequence
    assert all(
        s > CRAWLER_LAST_SEQ for s in seqs
    ), f"Some AI sequences overlap with crawler range (≤{CRAWLER_LAST_SEQ}): {seqs}"

    # Sequences must still be strictly increasing within the AI phase
    assert seqs == sorted(seqs) and len(seqs) == len(
        set(seqs)
    ), f"AI sequences not strictly increasing: {seqs}"

    # Verify get_item was called with the correct composite key
    get_item_call = mock_dynamodb_client.get_item.call_args
    key_used = get_item_call.kwargs["Key"]
    assert key_used["RegeneratedWebsiteId"]["S"] == WEBSITE_ID
    assert key_used["RegeneratedWebsiteUrl"]["S"] == URL

    print("test_sequence_continues_from_crawler: PASSED")


def test_events_use_ai_phase():
    """Verify all published events use phase='ai', not 'crawler'."""
    (
        _,
        _,
        _,
        mock_dynamodb_resource,
        mock_channel,
        mock_ably_rest,
        mock_openai,
        boto3_client_factory,
    ) = make_mocks()
    _clear_modules()

    with patch("boto3.client", side_effect=boto3_client_factory), patch(
        "boto3.resource", return_value=mock_dynamodb_resource
    ), patch("ably.AblyRest", return_value=mock_ably_rest), patch(
        "openai.OpenAI", return_value=mock_openai
    ):
        import lambda_function

        lambda_function.lambda_handler(make_event(), {})

    phases = [c.args[1]["phase"] for c in mock_channel.publish.call_args_list]
    assert all(p == "ai" for p in phases), f"Expected all phases to be 'ai', got: {phases}"

    print("test_events_use_ai_phase: PASSED")


if __name__ == "__main__":
    test_happy_path_publishes_all_steps()
    test_openai_failure_publishes_failed()
    test_s3_write_failure_publishes_failed()
    test_sequence_numbers_always_increase()
    test_sequence_continues_from_crawler()
    test_events_use_ai_phase()
    print("All tests passed.")
