"""
Test suite for the AI HTML+CSS Regeneration Lambda handler.

Tests verify:
1. Happy path: all expected steps published with correct sequence numbers
2. OpenAI failure: 500 returned
3. S3 write failure: 500 returned
4. Sequence numbers: always strictly increasing within a single invocation
5. Sequence continuation: AI sequences start at crawler_last_seq + 1

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

- The OpenAI mock response returns JSON with "html" and "css" keys because
  lambda_function now uses response_format={"type": "json_object"} and
  json.loads() on the content.

- s3.get_object is called twice per run: once for index.html and once for
  original-styles.css. The mock uses side_effect to return different bodies
  based on the Key argument.
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

# Sample HTML that produces one head chunk and one body chunk (two total chunks).
MOCK_HTML = (
    b"<html><head><title>Test</title></head>"
    b"<body><div class='hero'>Hello</div></body></html>"
)
MOCK_CSS = b".hero { color: red; font-size: 16px; }"
MOCK_CHUNK_RESPONSE = json.dumps({"html": "<p>regenerated</p>", "css": ".hero { color: neon; }"})

_mock_s3 = MagicMock()
_mock_secrets = MagicMock()
_mock_secrets.get_secret_value.return_value = {
    "SecretString": json.dumps({"OpenAIAPIKey": "test-key"})
}
_mock_openai_client = MagicMock()
_mock_dynamodb_resource = MagicMock()


def _boto3_client_factory(service, **_):
    return _mock_s3 if service == "s3" else _mock_secrets


with patch("boto3.client", side_effect=_boto3_client_factory), \
     patch("boto3.resource", return_value=_mock_dynamodb_resource), \
     patch("openai.OpenAI", return_value=_mock_openai_client):
    from lambda_function import lambda_handler


def make_event(website_id=WEBSITE_ID, url=URL, theme="cyberpunk"):
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

    s3.get_object uses side_effect to return HTML for index.html requests
    and CSS for original-styles.css requests, since lambda_function now
    reads both assets before chunking.

    The OpenAI mock returns JSON with "html" and "css" keys to match
    the response_format={"type": "json_object"} prompt and json.loads() call.
    """
    mock_s3 = MagicMock()

    def get_object_side_effect(Bucket=None, Key=None):
        if Key and Key.endswith("index.html"):
            return {"Body": MagicMock(read=lambda: MOCK_HTML)}
        return {"Body": MagicMock(read=lambda: MOCK_CSS)}

    mock_s3.get_object.side_effect = get_object_side_effect

    mock_dynamodb_client = MagicMock()
    mock_dynamodb_client.get_item.return_value = {}

    mock_table = MagicMock()
    mock_table.get_item.return_value = {}
    mock_dynamodb_resource = MagicMock()
    mock_dynamodb_resource.Table.return_value = mock_table

    mock_secrets = MagicMock()
    mock_secrets.get_secret_value.side_effect = lambda SecretId: {
        "test-secret": {"SecretString": json.dumps({"OpenAIAPIKey": "fake-openai-key"})},
        "test/ably-secret": {"SecretString": json.dumps({"AblyApiKey": "fake-ably-key"})},
    }[SecretId]

    def boto3_client_factory(service, **_):
        return {
            "s3": mock_s3,
            "dynamodb": mock_dynamodb_client,
            "secretsmanager": mock_secrets,
        }[service]

    mock_openai = MagicMock()
    mock_openai.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(
            message=MagicMock(content=MOCK_CHUNK_RESPONSE),
            finish_reason="stop",
        )],
        usage=MagicMock(prompt_tokens=100, completion_tokens=200),
    )

    mock_ably_channel = MagicMock()
    mock_ably_channel.publish = AsyncMock()
    mock_ably_rest = MagicMock()
    mock_ably_rest.channels.get.return_value = mock_ably_channel

    return (
        mock_s3, mock_dynamodb_client, mock_table, mock_dynamodb_resource,
        mock_ably_channel, mock_ably_rest, mock_openai, boto3_client_factory,
    )


def _clear_modules():
    for mod in ["lambda_function", "status_publisher"]:
        if mod in sys.modules:
            del sys.modules[mod]


# MOCK_HTML produces two chunks: head + one body chunk → two "regenerating_chunk_completed" events.
EXPECTED_STEPS = [
    "chunking",
    "regenerating_html_and_css",
    "regenerating_chunk_completed",  # head chunk
    "regenerating_chunk_completed",  # body chunk
    "Finalizing",
]


def test_happy_path_publishes_all_steps():
    """
    Verify a successful run publishes all expected steps in order with
    monotonically increasing sequence numbers, writes both HTML and CSS to S3,
    and updates DynamoDB status.

    Because head and body chunks run in parallel the two regenerating_chunk_completed
    events may arrive in either order, so we assert on counts and boundaries
    rather than strict positional equality for those middle events.
    """
    (mock_s3, mock_dynamodb_client, mock_table, mock_dynamodb_resource,
     mock_channel, mock_ably_rest, mock_openai, boto3_client_factory) = make_mocks()
    _clear_modules()

    with patch("boto3.client", side_effect=boto3_client_factory), \
         patch("boto3.resource", return_value=mock_dynamodb_resource), \
         patch("ably.AblyRest", return_value=mock_ably_rest), \
         patch("openai.OpenAI", return_value=mock_openai):
        import lambda_function
        result = lambda_function.lambda_handler(make_event(), {})

    assert result["statusCode"] == 200

    steps = [c.args[1]["step"] for c in mock_channel.publish.call_args_list]

    # First two and last step are deterministic; the middle chunk-complete events are parallel
    assert steps[0] == "chunking"
    assert steps[1] == "regenerating_html_and_css"
    assert steps[-1] == "Finalizing"
    assert steps[2:-1].count("regenerating_chunk_completed") == len(steps) - 3

    seqs = [c.args[1]["sequence"] for c in mock_channel.publish.call_args_list]
    assert seqs == list(range(1, len(EXPECTED_STEPS) + 1)), f"Unexpected sequences: {seqs}"

    last = mock_channel.publish.call_args_list[-1].args[1]
    assert last["status"] == "completed"

    # Two S3 writes: Regenerated-Index.html and Regenerated-Styles.css
    assert mock_s3.put_object.call_count == 2
    written_keys = {call.kwargs["Key"] for call in mock_s3.put_object.call_args_list}
    assert written_keys == {
        f"{WEBSITE_ID}/Regenerated-Index.html",
        f"{WEBSITE_ID}/Regenerated-Styles.css",
    }

    assert mock_dynamodb_client.update_item.call_count == len(EXPECTED_STEPS)
    assert mock_table.update_item.call_count == 2

    print("test_happy_path_publishes_all_steps: PASSED")


def test_openai_failure_returns_500():
    """
    Verify that an OpenAI error causes a 500 response.
    chunking and regenerating_html_and_css are published before the failure;
    Finalizing is not published.
    """
    (_, _, _, mock_dynamodb_resource,
     mock_channel, mock_ably_rest, mock_openai, boto3_client_factory) = make_mocks()
    mock_openai.chat.completions.create.side_effect = Exception("OpenAI error")
    _clear_modules()

    with patch("boto3.client", side_effect=boto3_client_factory), \
         patch("boto3.resource", return_value=mock_dynamodb_resource), \
         patch("ably.AblyRest", return_value=mock_ably_rest), \
         patch("openai.OpenAI", return_value=mock_openai):
        import lambda_function
        result = lambda_function.lambda_handler(make_event(), {})

    assert result["statusCode"] == 500

    steps = [c.args[1]["step"] for c in mock_channel.publish.call_args_list]
    assert "chunking" in steps
    assert "regenerating_html_and_css" in steps
    assert "Finalizing" not in steps

    print("test_openai_failure_returns_500: PASSED")


def test_s3_write_failure_returns_500():
    """
    Verify that an S3 write error causes a 500 response.
    Chunk regeneration succeeds, so regenerating_chunk_completed events are present,
    but Finalizing is not published since the write fails first.
    """
    (mock_s3, _, _, mock_dynamodb_resource,
     mock_channel, mock_ably_rest, mock_openai, boto3_client_factory) = make_mocks()
    mock_s3.put_object.side_effect = Exception("S3 write error")
    _clear_modules()

    with patch("boto3.client", side_effect=boto3_client_factory), \
         patch("boto3.resource", return_value=mock_dynamodb_resource), \
         patch("ably.AblyRest", return_value=mock_ably_rest), \
         patch("openai.OpenAI", return_value=mock_openai):
        import lambda_function
        result = lambda_function.lambda_handler(make_event(), {})

    assert result["statusCode"] == 500

    steps = [c.args[1]["step"] for c in mock_channel.publish.call_args_list]
    assert "Finalizing" not in steps

    print("test_s3_write_failure_returns_500: PASSED")


def test_sequence_numbers_always_increase():
    """
    Verify all published events carry strictly increasing sequence numbers.
    """
    (_, _, _, mock_dynamodb_resource,
     mock_channel, mock_ably_rest, mock_openai, boto3_client_factory) = make_mocks()
    _clear_modules()

    with patch("boto3.client", side_effect=boto3_client_factory), \
         patch("boto3.resource", return_value=mock_dynamodb_resource), \
         patch("ably.AblyRest", return_value=mock_ably_rest), \
         patch("openai.OpenAI", return_value=mock_openai):
        import lambda_function
        lambda_function.lambda_handler(make_event(), {})

    seqs = [c.args[1]["sequence"] for c in mock_channel.publish.call_args_list]
    assert seqs == sorted(seqs) and len(seqs) == len(set(seqs)), \
        f"Sequences not strictly increasing: {seqs}"

    print("test_sequence_numbers_always_increase: PASSED")


def test_sequence_continues_from_crawler():
    """
    Verify the AI lambda reads CurrentSequence from DynamoDB and starts
    its counter at crawler_last_seq + 1, producing a globally ordered stream.
    """
    (_, mock_dynamodb_client, _, mock_dynamodb_resource,
     mock_channel, mock_ably_rest, mock_openai, boto3_client_factory) = make_mocks()

    CRAWLER_LAST_SEQ = 6
    mock_dynamodb_client.get_item.return_value = {
        "Item": {
            "RegeneratedWebsiteId": {"S": WEBSITE_ID},
            "RegeneratedWebsiteUrl": {"S": URL},
            "CurrentSequence": {"N": str(CRAWLER_LAST_SEQ)},
        }
    }
    _clear_modules()

    with patch("boto3.client", side_effect=boto3_client_factory), \
         patch("boto3.resource", return_value=mock_dynamodb_resource), \
         patch("ably.AblyRest", return_value=mock_ably_rest), \
         patch("openai.OpenAI", return_value=mock_openai):
        import lambda_function
        result = lambda_function.lambda_handler(make_event(), {})

    assert result["statusCode"] == 200

    seqs = [c.args[1]["sequence"] for c in mock_channel.publish.call_args_list]

    assert seqs[0] == CRAWLER_LAST_SEQ + 1, (
        f"AI sequence should start at {CRAWLER_LAST_SEQ + 1}, got {seqs[0]}"
    )
    assert all(s > CRAWLER_LAST_SEQ for s in seqs), (
        f"Some AI sequences overlap with crawler range (≤{CRAWLER_LAST_SEQ}): {seqs}"
    )
    assert seqs == sorted(seqs) and len(seqs) == len(set(seqs)), (
        f"AI sequences not strictly increasing: {seqs}"
    )

    get_item_call = mock_dynamodb_client.get_item.call_args
    key_used = get_item_call.kwargs["Key"]
    assert key_used["RegeneratedWebsiteId"]["S"] == WEBSITE_ID
    assert key_used["RegeneratedWebsiteUrl"]["S"] == URL

    print("test_sequence_continues_from_crawler: PASSED")


if __name__ == "__main__":
    test_happy_path_publishes_all_steps()
    test_openai_failure_returns_500()
    test_s3_write_failure_returns_500()
    test_sequence_numbers_always_increase()
    test_sequence_continues_from_crawler()
    print("All tests passed.")
