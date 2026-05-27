"""
Test suite for the AI CSS Regeneration Lambda handler.

Tests verify:
1. Happy path: all expected steps published with correct sequence numbers
2. OpenAI failure: failed step published, 500 returned
3. S3 write failure: failed step published, 500 returned
4. Sequence numbers: always strictly increasing within a single invocation
5. Sequence continuation: AI sequences start at crawler_last_seq + 1

Important test-infrastructure notes
------------------------------------
- Every test imports handler AND calls lambda_handler() INSIDE the same
  patch with-block.  This is required because status_publisher.py lazily
  initialises its boto3 / Ably singletons on first use; if those singletons
  are created outside the patch context they hit real AWS and raise
  NoCredentialsError.

- mock_ably_channel.publish is an AsyncMock.  status_publisher wraps the
  Ably publish call with asyncio.run(), which requires the mock to return an
  awaitable coroutine rather than a plain MagicMock object.

- mock_dynamodb.get_item returns {} by default (no Item key) so
  get_current_sequence() resolves to 0 and sequences start at 1.
  Tests that verify continuation override this to return a real Item.
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

# boto3 and OpenAI are called at module level in handler.py, so we must patch
# before importing — otherwise the real AWS/OpenAI calls fire on import.
_mock_s3 = MagicMock()
_mock_secrets = MagicMock()
_mock_secrets.get_secret_value.return_value = {
    "SecretString": json.dumps({"OpenAIAPIKey": "test-key"})
}
_mock_openai_client = MagicMock()


def _boto3_client_factory(service, **kwargs):
    return _mock_s3 if service == "s3" else _mock_secrets


with patch("boto3.client", side_effect=_boto3_client_factory), \
     patch("openai.OpenAI", return_value=_mock_openai_client):
    from lambda_function import lambda_handler


def make_event(website_id="test-123", url="https://example.com", theme="cyberpunk"):
    return {
        "Records": [
            {
                "body": json.dumps({
                    "RegeneratedWebsiteId": website_id,
                    "RegeneratedWebsiteUrl": url,
                    "RegenerationTheme": theme,
                })
            }
        ]
    }


def make_mocks():
    """
    Build and return all mock objects needed by one test.

    DynamoDB get_item default
    -------------------------
    Returns {} (empty response, no 'Item' key) so get_current_sequence()
    resolves to 0 and the first published event gets sequence 1.
    Override mock_dynamodb.get_item.return_value in specific tests to
    simulate a crawler having already written a CurrentSequence value.

    AsyncMock for channel.publish
    -----------------------------
    ably-python >= 2.0.0 makes channel.publish() a coroutine.
    status_publisher wraps it with asyncio.run(), so the mock must return
    an awaitable.  AsyncMock satisfies this requirement.

    Returns:
        tuple: (mock_s3, mock_dynamodb, mock_ably_channel, mock_ably_rest,
                mock_openai, boto3_factory)
    """
    mock_s3 = MagicMock()
    mock_s3.get_object.return_value = {"Body": MagicMock(read=lambda: b"body { color: red; }")}

    mock_dynamodb = MagicMock()
    # Default: no Item in DynamoDB — get_current_sequence() returns 0, seq starts at 1.
    # Override in individual tests to simulate a specific crawler end-sequence.
    mock_dynamodb.get_item.return_value = {}

    mock_secrets = MagicMock()
    mock_secrets.get_secret_value.side_effect = lambda SecretId: {
        "test/openai-secret": {"SecretString": json.dumps({"OpenAIAPIKey": "fake-openai-key"})},
        "test/ably-secret": {"SecretString": json.dumps({"AblyApiKey": "fake-ably-key"})},
    }[SecretId]

    def boto3_factory(service, **kwargs):
        return {"s3": mock_s3, "dynamodb": mock_dynamodb, "secretsmanager": mock_secrets}[service]

    mock_openai = MagicMock()
    mock_openai.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content="body { color: neon; }"))]
    )

    mock_ably_channel = MagicMock()
    # AsyncMock required: asyncio.run(channel.publish(...)) needs an awaitable return value
    mock_ably_channel.publish = AsyncMock()
    mock_ably_rest = MagicMock()
    mock_ably_rest.channels.get.return_value = mock_ably_channel

    return mock_s3, mock_dynamodb, mock_ably_channel, mock_ably_rest, mock_openai, boto3_factory


def _clear_modules():
    """Remove cached handler/status_publisher so each test gets a fresh import."""
    for mod in ["handler", "status_publisher"]:
        if mod in sys.modules:
            del sys.modules[mod]


def test_happy_path_publishes_all_steps():
    """
    Verify that a successful run publishes all expected steps in order with
    monotonically increasing sequence numbers, writes the CSS to S3, and
    persists status to DynamoDB for every published event.
    """
    mock_s3, mock_dynamodb, mock_channel, mock_ably_rest, mock_openai, boto3_factory = make_mocks()
    _clear_modules()

    # Import AND call must be inside the with-block so lazy singletons use mocks.
    with patch("boto3.client", side_effect=boto3_factory), \
         patch("ably.AblyRest", return_value=mock_ably_rest), \
         patch("openai.OpenAI", return_value=mock_openai):
        import handler
        result = handler.lambda_handler(make_event(), {})

    assert result["statusCode"] == 200

    steps = [c.args[1]["step"] for c in mock_channel.publish.call_args_list]
    assert steps == [
        "received",
        "reading_source_css",
        "building_prompt",
        "calling_openai",
        "saving_regenerated_css",
        "completed",
    ], f"Unexpected steps: {steps}"

    seqs = [c.args[1]["sequence"] for c in mock_channel.publish.call_args_list]
    # With no prior crawler sequence (get_item returns {}), starts at 1
    assert seqs == list(range(1, len(steps) + 1)), f"Unexpected sequences: {seqs}"

    last = mock_channel.publish.call_args_list[-1].args[1]
    assert last["status"] == "completed"

    assert mock_s3.put_object.call_count == 1
    key = mock_s3.put_object.call_args.kwargs["Key"]
    assert key == f"{WEBSITE_ID}/Regenerated-Styles.css"
    # One DynamoDB update_item per published event (status persistence)
    assert mock_dynamodb.update_item.call_count == len(steps)

    # Verify update_item uses the composite key (both partition and sort key)
    for call in mock_dynamodb.update_item.call_args_list:
        key_used = call.kwargs["Key"]
        assert "RegeneratedWebsiteId" in key_used, "Missing partition key in DynamoDB update"
        assert "RegeneratedWebsiteUrl" in key_used, "Missing sort key in DynamoDB update"

    print("test_happy_path_publishes_all_steps: PASSED")


def test_openai_failure_publishes_failed():
    """
    Verify that an OpenAI error causes a 'failed' step to be published and a
    500 response to be returned.
    """
    mock_s3, mock_dynamodb, mock_channel, mock_ably_rest, mock_openai, boto3_factory = make_mocks()
    mock_openai.chat.completions.create.side_effect = Exception("OpenAI error")
    _clear_modules()

    with patch("boto3.client", side_effect=boto3_factory), \
         patch("ably.AblyRest", return_value=mock_ably_rest), \
         patch("openai.OpenAI", return_value=mock_openai):
        import handler
        result = handler.lambda_handler(make_event(), {})

    assert result["statusCode"] == 500

    steps = [c.args[1]["step"] for c in mock_channel.publish.call_args_list]
    assert "failed" in steps, f"Expected 'failed' step, got: {steps}"
    failed = next(c.args[1] for c in mock_channel.publish.call_args_list
                  if c.args[1]["step"] == "failed")
    assert failed["status"] == "failed"
    assert failed["error"] is not None
    print("test_openai_failure_publishes_failed: PASSED")


def test_s3_write_failure_publishes_failed():
    """
    Verify that an S3 write error causes a 'failed' step to be published and
    a 500 response to be returned.
    """
    mock_s3, mock_dynamodb, mock_channel, mock_ably_rest, mock_openai, boto3_factory = make_mocks()
    mock_s3.put_object.side_effect = Exception("S3 write error")
    _clear_modules()

    with patch("boto3.client", side_effect=boto3_factory), \
         patch("ably.AblyRest", return_value=mock_ably_rest), \
         patch("openai.OpenAI", return_value=mock_openai):
        import handler
        result = handler.lambda_handler(make_event(), {})

    assert result["statusCode"] == 500

    steps = [c.args[1]["step"] for c in mock_channel.publish.call_args_list]
    assert "failed" in steps, f"Expected 'failed' step, got: {steps}"
    print("test_s3_write_failure_publishes_failed: PASSED")


def test_sequence_numbers_always_increase():
    """
    Verify that all published events carry strictly increasing sequence numbers
    with no gaps or duplicates within a single invocation.
    Frontend deduplication relies on this.
    """
    mock_s3, mock_dynamodb, mock_channel, mock_ably_rest, mock_openai, boto3_factory = make_mocks()
    _clear_modules()

    with patch("boto3.client", side_effect=boto3_factory), \
         patch("ably.AblyRest", return_value=mock_ably_rest), \
         patch("openai.OpenAI", return_value=mock_openai):
        import handler
        handler.lambda_handler(make_event(), {})

    seqs = [c.args[1]["sequence"] for c in mock_channel.publish.call_args_list]
    assert seqs == sorted(seqs) and len(seqs) == len(set(seqs)), \
        f"Sequences not strictly increasing: {seqs}"
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
    mock_s3, mock_dynamodb, mock_channel, mock_ably_rest, mock_openai, boto3_factory = make_mocks()

    # Simulate the crawler having published 6 events (sequences 1–6).
    # The AI lambda should start at 7.
    CRAWLER_LAST_SEQ = 6
    mock_dynamodb.get_item.return_value = {
        "Item": {
            "RegeneratedWebsiteId": {"S": WEBSITE_ID},
            "RegeneratedWebsiteUrl": {"S": URL},
            "CurrentSequence": {"N": str(CRAWLER_LAST_SEQ)},
        }
    }
    _clear_modules()

    with patch("boto3.client", side_effect=boto3_factory), \
         patch("ably.AblyRest", return_value=mock_ably_rest), \
         patch("openai.OpenAI", return_value=mock_openai):
        import handler
        result = handler.lambda_handler(make_event(), {})

    assert result["statusCode"] == 200

    seqs = [c.args[1]["sequence"] for c in mock_channel.publish.call_args_list]

    # First AI event must immediately follow the last crawler event
    assert seqs[0] == CRAWLER_LAST_SEQ + 1, (
        f"AI sequence should start at {CRAWLER_LAST_SEQ + 1}, got {seqs[0]}"
    )

    # All AI sequences must be strictly greater than the crawler's last sequence
    assert all(s > CRAWLER_LAST_SEQ for s in seqs), (
        f"Some AI sequences overlap with crawler range (≤{CRAWLER_LAST_SEQ}): {seqs}"
    )

    # Sequences must still be strictly increasing within the AI phase
    assert seqs == sorted(seqs) and len(seqs) == len(set(seqs)), (
        f"AI sequences not strictly increasing: {seqs}"
    )

    # Verify get_item was called with the correct composite key
    get_item_call = mock_dynamodb.get_item.call_args
    key_used = get_item_call.kwargs["Key"]
    assert key_used["RegeneratedWebsiteId"]["S"] == WEBSITE_ID
    assert key_used["RegeneratedWebsiteUrl"]["S"] == URL

    print("test_sequence_continues_from_crawler: PASSED")


if __name__ == "__main__":
    test_happy_path_publishes_all_steps()
    test_openai_failure_publishes_failed()
    test_s3_write_failure_publishes_failed()
    test_sequence_numbers_always_increase()
    test_sequence_continues_from_crawler()
    print("All tests passed.")
