"""
Test suite for the AI HTML/CSS Regeneration Lambda handler.

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
import re
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

from html_chunker import split_html_into_chunks
from inline_html_regenerator import regenerate_html


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
        "Body": MagicMock(
            read=lambda: b"<html><head><title>Original</title></head>"
            b"<body><div style=\"color:red\">Hi</div></body></html>"
        )
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
                message=MagicMock(content='<div style="color:neon">Hi</div>'),
                finish_reason="stop",
            )
        ],
        usage=MagicMock(prompt_tokens=100, completion_tokens=200),
    )

    mock_ably_channel = MagicMock()
    mock_ably_channel.publish = AsyncMock()
    mock_ably_rest = MagicMock()
    mock_ably_rest.channels.get.return_value = mock_ably_channel
    mock_ably_rest.close = AsyncMock()

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


# Expected publish steps for the default mock HTML, which has a single top-level
# <body> child. split_html_into_chunks returns [head, body-div], and the head chunk
# doesn't go through the model, so there's exactly one "regenerating_html_chunks_completed" event.
EXPECTED_STEPS = [
    "chunking",
    "regenerating_html",
    "regenerating_html_chunks_completed",
    "Finalizing",
]


def test_happy_path_publishes_all_steps():
    """
    Verify that a successful run publishes all expected steps in order with
    monotonically increasing sequence numbers, writes the regenerated HTML to S3, and
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
    assert key == f"{WEBSITE_ID}/Regenerated-Index.html"
    assert mock_s3.put_object.call_args.kwargs["ContentType"] == "text/html; charset=utf-8"

    # status_publisher's update_item called once per Ably publish
    assert mock_dynamodb_client.update_item.call_count == len(EXPECTED_STEPS)

    # lambda_function's table update_item called for "processing" + "completed"
    assert mock_table.update_item.call_count == 2

    print("test_happy_path_publishes_all_steps: PASSED")


def test_openai_failure_publishes_failed():
    """
    Verify that an OpenAI error causes batchItemFailures to be returned and a
    "failed" Ably event to be published.
    Steps published before the failure (chunking, regenerating_html) are present;
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
    assert "regenerating_html" in steps
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
    The HTML chunks are regenerated successfully before the write fails, so
    regenerating_html_chunks_completed is published but Finalizing is not.
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


def test_split_html_into_chunks_labels_head_and_body():
    """A well-formed document yields a 'head' chunk followed by 'body' chunk(s)."""
    html = '<html><head><title>T</title></head><body><div class="a">Hi</div></body></html>'
    chunks, labels = split_html_into_chunks(html)
    assert labels == ["head", "body"]
    assert "<title>T</title>" in chunks[0]
    assert '<div class="a">Hi</div>' in chunks[1]
    print("test_split_html_into_chunks_labels_head_and_body: PASSED")


def test_split_html_into_chunks_strips_stale_stylesheet_link():
    """
    A leftover <link rel="stylesheet" href="./Regenerated-Styles.css"> from the old
    pipeline shape must be removed — that file is no longer written by this lambda.
    """
    html = (
        '<html><head><link rel="stylesheet" href="./Regenerated-Styles.css">'
        "<title>T</title></head><body><div>Hi</div></body></html>"
    )
    chunks, labels = split_html_into_chunks(html)
    assert "Regenerated-Styles.css" not in chunks[0]
    print("test_split_html_into_chunks_strips_stale_stylesheet_link: PASSED")


def test_split_html_into_chunks_packs_body_children_within_max_chars():
    """Small top-level body children are packed together into one chunk when they
    fit under max_chars, but a child that alone exceeds max_chars starts a new one."""
    html = (
        "<html><head></head><body>"
        '<div class="a">small</div><div class="b">also-small</div>'
        "</body></html>"
    )
    chunks, labels = split_html_into_chunks(html, max_chars=1000)
    assert labels == ["head", "body"]
    assert "class=\"a\"" in chunks[1] and "class=\"b\"" in chunks[1]

    big_html = (
        "<html><head></head><body>"
        f'<div class="a">{"x" * 50}</div><div class="b">{"y" * 50}</div>'
        "</body></html>"
    )
    small_chunks, small_labels = split_html_into_chunks(big_html, max_chars=60)
    assert small_labels == ["head", "body", "body"]
    print("test_split_html_into_chunks_packs_body_children_within_max_chars: PASSED")


def test_split_html_into_chunks_falls_back_to_raw_without_head_or_body():
    """Documents that can't be parsed into head/body fall back to a single raw chunk."""
    html = "<div>not a full document</div>"
    chunks, labels = split_html_into_chunks(html)
    assert labels == ["raw"]
    assert chunks == [html]
    print("test_split_html_into_chunks_falls_back_to_raw_without_head_or_body: PASSED")


def test_regenerate_html_chunk_keeps_original_when_model_returns_empty():
    """
    End-to-end: when the mocked OpenAI response returns empty content for a chunk,
    the final S3-written HTML must still contain that chunk's original content
    rather than losing it.
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
    mock_s3.get_object.return_value = {
        "Body": MagicMock(
            read=lambda: b'<html><head></head><body><div class="keep-me">Hi</div></body></html>'
        )
    }
    mock_openai.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content=""), finish_reason="stop")],
        usage=MagicMock(prompt_tokens=100, completion_tokens=200),
    )
    _clear_modules()

    with patch("boto3.client", side_effect=boto3_client_factory), patch(
        "boto3.resource", return_value=mock_dynamodb_resource
    ), patch("ably.AblyRest", return_value=mock_ably_rest), patch(
        "openai.OpenAI", return_value=mock_openai
    ):
        import lambda_function

        result = lambda_function.lambda_handler(make_event(), {})

    assert result == {"batchItemFailures": []}

    written_html = mock_s3.put_object.call_args.kwargs["Body"].decode("utf-8")
    assert 'class="keep-me"' in written_html, "Original chunk content should be preserved"

    print("test_regenerate_html_chunk_keeps_original_when_model_returns_empty: PASSED")


def test_regenerate_html_chunk_keeps_original_when_model_drops_image():
    """
    End-to-end: if the model's regenerated chunk is missing an <img> src that was
    present in the original, the final S3-written HTML must still contain that
    image rather than losing it (mirrors the empty-output fallback).
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
    mock_s3.get_object.return_value = {
        "Body": MagicMock(
            read=lambda: b'<html><head></head><body><img src="./images/img-0.jpg"></body></html>'
        )
    }
    mock_openai.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content="<p>no image here</p>"), finish_reason="stop")],
        usage=MagicMock(prompt_tokens=100, completion_tokens=200),
    )
    _clear_modules()

    with patch("boto3.client", side_effect=boto3_client_factory), patch(
        "boto3.resource", return_value=mock_dynamodb_resource
    ), patch("ably.AblyRest", return_value=mock_ably_rest), patch(
        "openai.OpenAI", return_value=mock_openai
    ):
        import lambda_function

        result = lambda_function.lambda_handler(make_event(), {})

    assert result == {"batchItemFailures": []}

    written_html = mock_s3.put_object.call_args.kwargs["Body"].decode("utf-8")
    assert 'src="./images/img-0.jpg"' in written_html, "Dropped image should be restored"

    print("test_regenerate_html_chunk_keeps_original_when_model_drops_image: PASSED")


def test_none_theme_does_not_leak_into_prompt():
    """
    When RegenerationTheme is None, the system prompt sent to the model must use a
    safe fallback label instead of interpolating the literal string "None".
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
    _clear_modules()

    with patch("boto3.client", side_effect=boto3_client_factory), patch(
        "boto3.resource", return_value=mock_dynamodb_resource
    ), patch("ably.AblyRest", return_value=mock_ably_rest), patch(
        "openai.OpenAI", return_value=mock_openai
    ):
        import lambda_function

        result = lambda_function.lambda_handler(make_event(theme=None), {})

    assert result == {"batchItemFailures": []}

    system_msg = mock_openai.chat.completions.create.call_args.kwargs["messages"][0]["content"]
    assert re.search(r"\bNone\b", system_msg) is None, "Literal 'None' must not leak into the prompt"
    assert "a clean modern redesign" in system_msg

    print("test_none_theme_does_not_leak_into_prompt: PASSED")


def test_regenerate_html_reassembles_style_block_with_import_before_other_rules():
    """
    Multiple body chunks can each emit hover/keyframe rules and font @import
    statements via a trailing <!--STYLE:...--> comment (since inline style="..."
    attributes can't express :hover, @keyframes, or @import). regenerate_html must
    strip those comments from the body and assemble them into a single <style>
    block in <head>, with any @import lines placed first per the CSS spec.
    """
    big_a = "a" * 20000
    big_b = "b" * 20000
    html = (
        "<html><head><title>T</title></head><body>"
        f'<div class="chunk-a">{big_a}</div>'
        f'<div class="chunk-b">{big_b}</div>'
        "</body></html>"
    )

    def side_effect(**kwargs):
        user_content = kwargs["messages"][1]["content"]
        if "chunk-a" in user_content:
            content = (
                '<div class="chunk-a" style="color:red">A</div>'
                "<!--STYLE:@import url('https://fonts.googleapis.com/css2?family=Orbitron');"
                " .chunk-a-hover:hover{color:gold} @keyframes pulse{0%{opacity:1}}-->"
            )
        else:
            content = '<div class="chunk-b" style="color:blue">B</div>'
        return MagicMock(
            choices=[MagicMock(message=MagicMock(content=content), finish_reason="stop")],
            usage=MagicMock(prompt_tokens=100, completion_tokens=200),
        )

    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = side_effect

    result = regenerate_html(mock_client, html, "theme prompt", "retro arcade")

    assert result.count("<style>") == 1
    style_block = result.split("<style>")[1].split("</style>")[0]
    assert style_block.strip().startswith("@import"), "@import must precede other rules"
    assert ".chunk-a-hover:hover{color:gold}" in style_block
    assert "@keyframes pulse" in style_block
    assert "<!--STYLE:" not in result, "STYLE comment should be stripped from the body"
    assert 'class="chunk-a"' in result and 'class="chunk-b"' in result

    print("test_regenerate_html_reassembles_style_block_with_import_before_other_rules: PASSED")


def test_regenerate_html_raw_fallback_returns_model_output_directly():
    """When head/body can't be parsed, regenerate_html sends the whole document as
    one chunk and returns the model's output as-is, with no <!DOCTYPE> wrapping added."""
    html = "<div>just a fragment, not a full document</div>"
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content="<div>regenerated fragment</div>"), finish_reason="stop")],
        usage=MagicMock(prompt_tokens=100, completion_tokens=200),
    )

    result = regenerate_html(mock_client, html, "theme prompt", "retro arcade")

    assert result == "<div>regenerated fragment</div>"
    assert "<!DOCTYPE" not in result

    print("test_regenerate_html_raw_fallback_returns_model_output_directly: PASSED")


def test_regenerate_html_raw_fallback_extracts_style_comment_into_style_block():
    """When head/body can't be parsed and the model still emits a trailing
    <!--STYLE:...--> comment (e.g. for a keyframes animation), it must be turned
    into a real <style> block rather than left as an inert HTML comment, or the
    animation would silently never render."""
    html = "<div>just a fragment, not a full document</div>"
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = MagicMock(
        choices=[
            MagicMock(
                message=MagicMock(
                    content='<div class="pulse">regenerated fragment</div>'
                    "<!--STYLE:.pulse{animation:pulse 2s infinite} @keyframes pulse{0%{opacity:1}}-->"
                ),
                finish_reason="stop",
            )
        ],
        usage=MagicMock(prompt_tokens=100, completion_tokens=200),
    )

    result = regenerate_html(mock_client, html, "theme prompt", "retro arcade")

    assert "<!--STYLE:" not in result, "STYLE comment should be stripped from the output"
    assert "<style>" in result and "</style>" in result
    style_block = result.split("<style>")[1].split("</style>")[0]
    assert "@keyframes pulse" in style_block
    assert '<div class="pulse">regenerated fragment</div>' in result

    print("test_regenerate_html_raw_fallback_extracts_style_comment_into_style_block: PASSED")


def test_regenerate_html_dedupes_identical_style_rules_across_chunks():
    """Chunks are regenerated independently and in parallel, so two chunks can
    emit byte-identical hover/keyframe rules (e.g. a shared fade-in animation).
    The assembled <style> block must not contain duplicate copies."""
    big_a = "a" * 20000
    big_b = "b" * 20000
    html = (
        "<html><head><title>T</title></head><body>"
        f'<div class="chunk-a">{big_a}</div>'
        f'<div class="chunk-b">{big_b}</div>'
        "</body></html>"
    )
    shared_rule = "@keyframes fadeIn{0%{opacity:0}100%{opacity:1}}"

    def side_effect(**kwargs):
        user_content = kwargs["messages"][1]["content"]
        if "chunk-a" in user_content:
            content = f'<div class="chunk-a fade-in" style="color:red">A</div><!--STYLE:{shared_rule}-->'
        else:
            content = f'<div class="chunk-b fade-in" style="color:blue">B</div><!--STYLE:{shared_rule}-->'
        return MagicMock(
            choices=[MagicMock(message=MagicMock(content=content), finish_reason="stop")],
            usage=MagicMock(prompt_tokens=100, completion_tokens=200),
        )

    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = side_effect

    result = regenerate_html(mock_client, html, "theme prompt", "retro arcade")

    style_block = result.split("<style>")[1].split("</style>")[0]
    assert style_block.count("@keyframes fadeIn") == 1, "Identical rules from different chunks must be deduped"

    print("test_regenerate_html_dedupes_identical_style_rules_across_chunks: PASSED")


if __name__ == "__main__":
    test_happy_path_publishes_all_steps()
    test_openai_failure_publishes_failed()
    test_s3_write_failure_publishes_failed()
    test_sequence_numbers_always_increase()
    test_sequence_continues_from_crawler()
    test_events_use_ai_phase()
    test_split_html_into_chunks_labels_head_and_body()
    test_split_html_into_chunks_strips_stale_stylesheet_link()
    test_split_html_into_chunks_packs_body_children_within_max_chars()
    test_split_html_into_chunks_falls_back_to_raw_without_head_or_body()
    test_regenerate_html_chunk_keeps_original_when_model_returns_empty()
    test_regenerate_html_chunk_keeps_original_when_model_drops_image()
    test_none_theme_does_not_leak_into_prompt()
    test_regenerate_html_reassembles_style_block_with_import_before_other_rules()
    test_regenerate_html_raw_fallback_returns_model_output_directly()
    test_regenerate_html_raw_fallback_extracts_style_comment_into_style_block()
    test_regenerate_html_dedupes_identical_style_rules_across_chunks()
    print("All tests passed.")
