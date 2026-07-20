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

from bs4 import BeautifulSoup

from html_chunker import split_html_into_chunks, split_node_into_parts
from inline_html_regenerator import (
    _extract_img_data_srcs,
    _extract_img_srcs,
    _hook_class_candidates,
    regenerate_html,
)


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
# doesn't go through the model, so there's exactly one "regenerating_html_and_styling_chunks_completed" event.
EXPECTED_STEPS = [
    "chunking",
    "regenerating_html_and_styling",
    "regenerating_html_and_styling_chunks_completed",
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
    Steps published before the failure (chunking, regenerating_html_and_styling) are present;
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
    assert "regenerating_html_and_styling" in steps
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
    regenerating_html_and_styling_chunks_completed is published but Finalizing is not.
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
    # A child exceeding max_chars now splits into as many pieces as needed to
    # preserve all its content (rather than being truncated into a single
    # chunk), so it starts one or more new chunks, not exactly one.
    assert small_labels[0] == "head"
    assert small_labels[1:] == ["body"] * (len(small_labels) - 1)
    assert len(small_labels) > 3
    reconstructed_body = "".join(small_chunks[1:])
    assert reconstructed_body.count("x") == 50 and reconstructed_body.count("y") == 50
    print("test_split_html_into_chunks_packs_body_children_within_max_chars: PASSED")


def test_split_node_into_parts_preserves_wrapping_tag_when_splitting():
    """Splitting an oversized element must keep its own tag/attributes on every
    resulting part — dropping them corrupts structure."""
    inner = "".join(f"<p>{'p' * 20}</p>" for _ in range(20))
    soup = BeautifulSoup(f'<div class="wrap" data-x="y">{inner}</div>', "html.parser")
    div = soup.find("div")

    parts = split_node_into_parts(div, max_chars=100)

    assert len(parts) > 1, "Expected the oversized div to be split into multiple parts"
    for part in parts:
        assert part.startswith('<div class="wrap" data-x="y">')
        assert part.endswith("</div>")

    reconstructed = "".join(p[len('<div class="wrap" data-x="y">') : -len("</div>")] for p in parts)
    assert reconstructed == inner, "Splitting must not lose or duplicate content"

    print("test_split_node_into_parts_preserves_wrapping_tag_when_splitting: PASSED")


def test_split_node_into_parts_preserves_script_tag_when_splitting():
    """An oversized <script> tag must stay wrapped in <script>...</script> when
    split — previously the wrapping tag was dropped entirely, turning the JS
    body into inert text on the page."""
    long_js = "console.log('x');" * 50
    soup = BeautifulSoup(f"<script>{long_js}</script>", "html.parser")
    script = soup.find("script")

    parts = split_node_into_parts(script, max_chars=100)

    assert parts, "Expected at least one part"
    for part in parts:
        assert part.startswith("<script>") and part.endswith(
            "</script>"
        ), f"Script content must stay wrapped in its own tag, got: {part!r}"

    print("test_split_node_into_parts_preserves_script_tag_when_splitting: PASSED")


def test_split_node_into_parts_splits_oversized_leaf_without_dropping_content():
    """A leaf node (e.g. the text inside a <script>) that alone exceeds
    max_chars must be split into multiple pieces, not silently truncated to
    one max_chars piece with the remainder discarded."""
    long_js = "console.log('x');" * 50
    soup = BeautifulSoup(f"<script>{long_js}</script>", "html.parser")
    script = soup.find("script")

    parts = split_node_into_parts(script, max_chars=100)

    assert len(parts) > 1
    reconstructed = "".join(p[len("<script>") : -len("</script>")] for p in parts)
    assert reconstructed == long_js, "No JS content should be silently dropped"

    print("test_split_node_into_parts_splits_oversized_leaf_without_dropping_content: PASSED")


def test_split_html_into_chunks_splits_oversized_head_into_multiple_head_chunks():
    """A <head> that alone exceeds max_chars must be split into multiple
    'head'-labeled chunks, not silently truncated."""
    long_meta = "".join(f'<meta name="tag{i}" content="{"x" * 20}">' for i in range(20))
    html = f"<html><head>{long_meta}</head><body><div>Hi</div></body></html>"

    chunks, labels = split_html_into_chunks(html, max_chars=100)

    head_indices = [i for i, label in enumerate(labels) if label == "head"]
    assert len(head_indices) > 1
    assert labels[len(head_indices) :] == ["body"] * (len(labels) - len(head_indices))
    reconstructed_head = "".join(chunks[i] for i in head_indices)
    for i in range(20):
        assert f'name="tag{i}"' in reconstructed_head

    print("test_split_html_into_chunks_splits_oversized_head_into_multiple_head_chunks: PASSED")


def test_split_html_into_chunks_falls_back_to_raw_without_head_or_body():
    """Documents that can't be parsed into head/body fall back to a single raw chunk."""
    html = "<div>not a full document</div>"
    chunks, labels = split_html_into_chunks(html)
    assert labels == ["raw"]
    assert chunks == [html]
    print("test_split_html_into_chunks_falls_back_to_raw_without_head_or_body: PASSED")


def test_split_html_into_chunks_splits_large_document_without_head_or_body():
    """A large fragment that can't be parsed into head/body must still respect
    max_chars — previously the whole document shipped as one unsplit chunk,
    reintroducing the oversized-request timeout this chunking exists to fix."""
    html = f'<div class="a">{"x" * 50}</div>' f'<div class="b">{"y" * 50}</div>'
    chunks, labels = split_html_into_chunks(html, max_chars=60)
    assert labels == ["raw"] * len(labels)
    assert len(labels) > 1
    assert all(len(c) <= 60 for c in chunks)
    reconstructed = "".join(chunks)
    assert reconstructed.count("x") == 50 and reconstructed.count("y") == 50
    print("test_split_html_into_chunks_splits_large_document_without_head_or_body: PASSED")


def test_split_html_into_chunks_strips_stale_stylesheet_link_without_head_or_body():
    """The stale Regenerated-Styles.css link must be stripped even when the
    document doesn't parse into a clean head/body shape."""
    html = '<link rel="stylesheet" href="./Regenerated-Styles.css"><div>Hi</div>'
    chunks, labels = split_html_into_chunks(html)
    assert labels == ["raw"]
    assert "Regenerated-Styles.css" not in "".join(chunks)
    print("test_split_html_into_chunks_strips_stale_stylesheet_link_without_head_or_body: PASSED")


def test_extract_img_srcs_ignores_data_src_attribute():
    """The img-src regex must not match inside a data-src attribute (used for
    lazy-loaded images) — matching it would validate the img-preservation
    safety check against the wrong URL."""
    assert _extract_img_srcs('<img data-src="lazy.jpg" src="placeholder.jpg">') == {"placeholder.jpg"}
    assert _extract_img_srcs('<img src="real.jpg" data-src="lazy.jpg">') == {"real.jpg"}
    assert _extract_img_srcs('<img srcset="a.jpg 1x, b.jpg 2x" src="real.jpg">') == {"real.jpg"}
    print("test_extract_img_srcs_ignores_data_src_attribute: PASSED")


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


def test_extract_img_data_srcs_matches_only_data_src_attribute():
    """The data-src regex must only match data-src, not src (symmetric to the
    existing _extract_img_srcs test that data-src doesn't leak into src)."""
    assert _extract_img_data_srcs('<img data-src="lazy.jpg" src="placeholder.jpg">') == {"lazy.jpg"}
    assert _extract_img_data_srcs('<img src="real.jpg">') == set()
    print("test_extract_img_data_srcs_matches_only_data_src_attribute: PASSED")


def test_regenerate_html_chunk_keeps_original_when_images_are_swapped():
    """End-to-end: model returns the same SET of src values but swaps which
    <img> tag has which — the set-based check alone would miss this, so a
    position-preserved check must catch it and keep the original chunk."""
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
            read=lambda: b'<html><head></head><body>'
            b'<img src="cat.jpg"><img src="dog.jpg">'
            b"</body></html>"
        )
    }
    mock_openai.chat.completions.create.return_value = MagicMock(
        choices=[
            MagicMock(
                message=MagicMock(content='<img src="dog.jpg"><img src="cat.jpg">'),
                finish_reason="stop",
            )
        ],
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
    assert written_html.index('src="cat.jpg"') < written_html.index(
        'src="dog.jpg"'
    ), "Original order must be restored; swapped output must be rejected"

    print("test_regenerate_html_chunk_keeps_original_when_images_are_swapped: PASSED")


def test_regenerate_html_chunk_keeps_original_when_data_src_altered():
    """End-to-end: src is preserved but data-src (the real lazy-load URL) is
    altered — must be detected even though the existing src-only check passes."""
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
            read=lambda: b'<html><head></head><body>'
            b'<img data-src="real.jpg" src="placeholder.jpg">'
            b"</body></html>"
        )
    }
    mock_openai.chat.completions.create.return_value = MagicMock(
        choices=[
            MagicMock(
                message=MagicMock(content='<img data-src="wrong.jpg" src="placeholder.jpg">'),
                finish_reason="stop",
            )
        ],
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
    assert 'data-src="real.jpg"' in written_html
    assert 'data-src="wrong.jpg"' not in written_html

    print("test_regenerate_html_chunk_keeps_original_when_data_src_altered: PASSED")


def test_regenerate_html_chunk_accepts_model_appending_new_decorative_image():
    """A model that keeps all original <img> tags/order and appends a brand-new
    one at the end must NOT be rejected by the new position-preserved check."""
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = MagicMock(
        choices=[
            MagicMock(
                message=MagicMock(content='<img src="cat.jpg"><img src="new-decorative.jpg">'),
                finish_reason="stop",
            )
        ],
        usage=MagicMock(prompt_tokens=100, completion_tokens=200),
    )
    html = '<html><head></head><body><img src="cat.jpg"></body></html>'

    result = regenerate_html(mock_client, html, "theme prompt", "retro arcade")

    assert "new-decorative.jpg" in result

    print("test_regenerate_html_chunk_accepts_model_appending_new_decorative_image: PASSED")


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
    # Hook identifiers get suffixed with the chunk index (here, chunk 1) so
    # they can never collide with another chunk's independently-chosen names.
    assert ".chunk-a-hover-c1:hover{color:gold}" in style_block
    assert "@keyframes pulse-c1{0%{opacity:1}}" in style_block
    assert "<!--STYLE:" not in result, "STYLE comment should be stripped from the body"
    assert 'class="chunk-a"' in result and 'class="chunk-b"' in result

    print("test_regenerate_html_reassembles_style_block_with_import_before_other_rules: PASSED")


def test_regenerate_html_reassembles_multi_chunk_head():
    """End-to-end: an oversized <head> splits into multiple 'head' chunks;
    regenerate_html must concatenate ALL of them (not just chunks[0]) into
    the final output's <head>, with nothing missing, and without doubling
    up the <head> wrapper tag."""
    long_text = "x" * 40000
    html = f"<html><head><title>{long_text}</title></head><body><div>Hi</div></body></html>"
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content="<div>regenerated</div>"), finish_reason="stop")],
        usage=MagicMock(prompt_tokens=100, completion_tokens=200),
    )

    result = regenerate_html(mock_client, html, "theme prompt", "retro arcade")

    assert result.count("x") == 40000, "No head content should be silently dropped"
    assert ">regenerated</div>" in result
    assert 'style="width:100%;height:100%"' in result, (
        "Single top-level body div should be forced to fill the page"
    )
    assert result.count("<head") == 1, "Head chunks must merge into a single, non-nested <head>"

    print("test_regenerate_html_reassembles_multi_chunk_head: PASSED")


def test_regenerate_html_forces_full_width_height_on_single_root_wrapper():
    """When the regenerated body has exactly one top-level element (the common
    'single wrapper div holds the whole page' shape), it must be forced to
    width:100%/height:100% so the page fills the iframe it's displayed in,
    with any pre-existing conflicting width/height replaced rather than
    duplicated alongside the forced values."""
    html = '<html><head></head><body><div style="width:300px;height:200px;color:red">Hi</div></body></html>'
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = MagicMock(
        choices=[
            MagicMock(
                message=MagicMock(content='<div style="width:300px;height:200px;color:neon">Hi</div>'),
                finish_reason="stop",
            )
        ],
        usage=MagicMock(prompt_tokens=100, completion_tokens=200),
    )

    result = regenerate_html(mock_client, html, "theme prompt", "retro arcade")

    style_attr = re.search(r'<div style="([^"]*)">Hi</div>', result).group(1)
    assert "width:100%" in style_attr
    assert "height:100%" in style_attr
    assert "width:300px" not in style_attr
    assert "height:200px" not in style_attr
    assert "color:neon" in style_attr, "Unrelated declarations must survive the merge"
    assert style_attr.count("width:") == 1 and style_attr.count("height:") == 1

    print("test_regenerate_html_forces_full_width_height_on_single_root_wrapper: PASSED")


def test_regenerate_html_does_not_force_full_width_with_multiple_top_level_siblings():
    """When the regenerated body has more than one top-level element (e.g.
    header/main/footer siblings), there's no single 'main div' to target, so
    neither sibling should be forced to width:100%/height:100%."""
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
            content = '<div class="chunk-a" style="color:red">A</div>'
        else:
            content = '<div class="chunk-b" style="color:blue">B</div>'
        return MagicMock(
            choices=[MagicMock(message=MagicMock(content=content), finish_reason="stop")],
            usage=MagicMock(prompt_tokens=100, completion_tokens=200),
        )

    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = side_effect

    result = regenerate_html(mock_client, html, "theme prompt", "retro arcade")

    assert "width:100%" not in result
    assert "height:100%" not in result

    print("test_regenerate_html_does_not_force_full_width_with_multiple_top_level_siblings: PASSED")


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
    # The hook class/keyframe name gets suffixed with the chunk index (here,
    # chunk 0, the only chunk) so it can never collide with another chunk's.
    assert "@keyframes pulse-c0{0%{opacity:1}}" in style_block
    assert '<div class="pulse-c0">regenerated fragment</div>' in result

    print("test_regenerate_html_raw_fallback_extracts_style_comment_into_style_block: PASSED")


def test_hook_class_candidates_excludes_descendant_selector_classes():
    """A compound/descendant selector like `.card-hover:hover .icon{...}`
    must only treat `.card-hover` (the leftmost token, where the model's own
    hook class lives) as a rename candidate. `.icon` is a descendant
    reference to some other element's existing class and must not be
    renamed, or unrelated elements sharing that class get corrupted."""
    style_content = ".card-hover:hover .icon{transform:scale(1.1)}"
    assert _hook_class_candidates(style_content) == {"card-hover"}
    print("test_hook_class_candidates_excludes_descendant_selector_classes: PASSED")


def test_hook_class_candidates_simple_hover_rule_regression():
    """Regression: the common case (single-class hover hook) must still work."""
    assert _hook_class_candidates(".hover-lift:hover{color:gold}") == {"hover-lift"}
    print("test_hook_class_candidates_simple_hover_rule_regression: PASSED")


def test_hook_class_candidates_keyframes_only_untouched():
    """Regression: a keyframes-only STYLE comment yields no class candidates
    (keyframe names are handled separately by _KEYFRAMES_NAME_RE)."""
    assert _hook_class_candidates("@keyframes pulse{0%{opacity:1}100%{opacity:0}}") == set()
    print("test_hook_class_candidates_keyframes_only_untouched: PASSED")


def test_hook_class_candidates_ignores_import_statement():
    """An @import clause (no braces) must not get swallowed into the next
    rule's selector header when splitting on '{'."""
    style_content = "@import url('https://fonts.googleapis.com/css2?family=Orbitron'); .hover-lift:hover{color:gold}"
    assert _hook_class_candidates(style_content) == {"hover-lift"}
    print("test_hook_class_candidates_ignores_import_statement: PASSED")


def test_regenerate_html_namespaces_colliding_hover_classes_across_chunks():
    """Chunks are regenerated independently and in parallel with zero shared
    context, so two chunks can independently choose the SAME hook class name
    for DIFFERENT hover rules (e.g. both call it ".hover-lift"). Merging by
    exact rule text would let both rules ship under one name, with one
    silently overriding the other in the browser. Each chunk's hook classes
    must be namespaced so they can never collide."""
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
                '<div class="chunk-a hover-lift" style="color:red">A</div>'
                "<!--STYLE:.hover-lift:hover{color:gold}-->"
            )
        else:
            content = (
                '<div class="chunk-b hover-lift" style="color:blue">B</div>'
                "<!--STYLE:.hover-lift:hover{color:teal}-->"
            )
        return MagicMock(
            choices=[MagicMock(message=MagicMock(content=content), finish_reason="stop")],
            usage=MagicMock(prompt_tokens=100, completion_tokens=200),
        )

    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = side_effect

    result = regenerate_html(mock_client, html, "theme prompt", "retro arcade")

    style_block = result.split("<style>")[1].split("</style>")[0]
    assert style_block.count(":hover{") == 2, "Both chunks' hover rules must survive, not collapse into one"
    assert ".hover-lift-c1:hover{color:gold}" in style_block
    assert ".hover-lift-c2:hover{color:teal}" in style_block
    assert 'class="chunk-a hover-lift-c1"' in result
    assert 'class="chunk-b hover-lift-c2"' in result

    print("test_regenerate_html_namespaces_colliding_hover_classes_across_chunks: PASSED")


def test_regenerate_html_namespaces_colliding_keyframes_across_chunks():
    """Same collision risk as hover classes, but for @keyframes animation
    names referenced via the animation shorthand — each chunk's keyframes must
    be namespaced so two different animations sharing a name don't collide."""
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
                '<div class="chunk-a" style="color:red;animation:fade-in 1s">A</div>'
                "<!--STYLE:@keyframes fade-in{0%{opacity:0}100%{opacity:1}}-->"
            )
        else:
            content = (
                '<div class="chunk-b" style="color:blue;animation:fade-in 2s">B</div>'
                "<!--STYLE:@keyframes fade-in{0%{opacity:1}100%{opacity:0}}-->"
            )
        return MagicMock(
            choices=[MagicMock(message=MagicMock(content=content), finish_reason="stop")],
            usage=MagicMock(prompt_tokens=100, completion_tokens=200),
        )

    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = side_effect

    result = regenerate_html(mock_client, html, "theme prompt", "retro arcade")

    style_block = result.split("<style>")[1].split("</style>")[0]
    assert "@keyframes fade-in-c1{0%{opacity:0}100%{opacity:1}}" in style_block
    assert "@keyframes fade-in-c2{0%{opacity:1}100%{opacity:0}}" in style_block
    assert "animation:fade-in-c1 1s" in result
    assert "animation:fade-in-c2 2s" in result

    print("test_regenerate_html_namespaces_colliding_keyframes_across_chunks: PASSED")


def test_regenerate_html_does_not_rename_descendant_class_in_compound_hover_selector():
    """End-to-end: a model-emitted compound/descendant hover selector must not
    corrupt an unrelated element in the SAME chunk that legitimately carries
    the referenced class for a different purpose."""
    html = "<html><head><title>T</title></head><body><div>seed</div></body></html>"
    content = (
        '<div class="card-hover"><span class="icon">x</span></div>'
        '<div class="icon">unrelated, must stay icon</div>'
        "<!--STYLE:.card-hover:hover .icon{transform:scale(1.1)}-->"
    )
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content=content), finish_reason="stop")],
        usage=MagicMock(prompt_tokens=100, completion_tokens=200),
    )

    result = regenerate_html(mock_client, html, "theme prompt", "retro arcade")

    assert 'class="card-hover-c1"' in result
    assert 'class="icon">unrelated, must stay icon' in result
    style_block = result.split("<style>")[1].split("</style>")[0]
    assert ".card-hover-c1:hover .icon{transform:scale(1.1)}" in style_block

    print("test_regenerate_html_does_not_rename_descendant_class_in_compound_hover_selector: PASSED")


def test_regenerate_html_merges_multiple_raw_chunks():
    """When head/body can't be parsed and the document is large enough to
    split into multiple raw chunks, regenerate_html must concatenate all of
    them (not just the first) and still extract any STYLE comment into a
    single <style> block."""
    big_a = "a" * 20000
    big_b = "b" * 20000
    html = f'<div class="a">{big_a}</div><div class="b">{big_b}</div>'

    chunks, labels = split_html_into_chunks(html)
    assert labels == ["raw", "raw"], "Test setup expects the fragment to split into two raw chunks"

    def side_effect(**kwargs):
        user_content = kwargs["messages"][1]["content"]
        if 'class="a"' in user_content:
            content = '<div class="a fade">A regenerated</div><!--STYLE:@keyframes fade{0%{opacity:0}}-->'
        else:
            content = '<div class="b">B regenerated</div>'
        return MagicMock(
            choices=[MagicMock(message=MagicMock(content=content), finish_reason="stop")],
            usage=MagicMock(prompt_tokens=100, completion_tokens=200),
        )

    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = side_effect

    result = regenerate_html(mock_client, html, "theme prompt", "retro arcade")

    assert "A regenerated" in result and "B regenerated" in result
    assert "<!--STYLE:" not in result
    assert "<style>" in result and "</style>" in result
    style_block = result.split("<style>")[1].split("</style>")[0]
    assert "@keyframes fade-c0" in style_block

    print("test_regenerate_html_merges_multiple_raw_chunks: PASSED")


def test_regenerate_chunk_prompt_includes_layout_preservation():
    """Every chunk's system prompt must include the layout-preservation rules
    (so flex/grid/positioning survive the theme rewrite)."""
    html = '<html><head></head><body><div style="display:flex">Hi</div></body></html>'
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content='<div style="display:flex">Hi</div>'), finish_reason="stop")],
        usage=MagicMock(prompt_tokens=100, completion_tokens=200),
    )

    regenerate_html(mock_client, html, "theme prompt", "retro arcade")

    system_msg = mock_client.chat.completions.create.call_args.kwargs["messages"][0]["content"]
    assert "LAYOUT & POSITIONING" in system_msg
    assert "grid-template-columns" in system_msg
    assert "flex-direction" in system_msg

    print("test_regenerate_chunk_prompt_includes_layout_preservation: PASSED")


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
    test_split_node_into_parts_preserves_wrapping_tag_when_splitting()
    test_split_node_into_parts_preserves_script_tag_when_splitting()
    test_split_node_into_parts_splits_oversized_leaf_without_dropping_content()
    test_split_html_into_chunks_splits_oversized_head_into_multiple_head_chunks()
    test_split_html_into_chunks_falls_back_to_raw_without_head_or_body()
    test_split_html_into_chunks_splits_large_document_without_head_or_body()
    test_split_html_into_chunks_strips_stale_stylesheet_link_without_head_or_body()
    test_extract_img_srcs_ignores_data_src_attribute()
    test_regenerate_html_chunk_keeps_original_when_model_returns_empty()
    test_regenerate_html_chunk_keeps_original_when_model_drops_image()
    test_extract_img_data_srcs_matches_only_data_src_attribute()
    test_regenerate_html_chunk_keeps_original_when_images_are_swapped()
    test_regenerate_html_chunk_keeps_original_when_data_src_altered()
    test_regenerate_html_chunk_accepts_model_appending_new_decorative_image()
    test_none_theme_does_not_leak_into_prompt()
    test_regenerate_html_reassembles_style_block_with_import_before_other_rules()
    test_regenerate_html_reassembles_multi_chunk_head()
    test_regenerate_html_forces_full_width_height_on_single_root_wrapper()
    test_regenerate_html_does_not_force_full_width_with_multiple_top_level_siblings()
    test_regenerate_html_raw_fallback_returns_model_output_directly()
    test_regenerate_html_raw_fallback_extracts_style_comment_into_style_block()
    test_hook_class_candidates_excludes_descendant_selector_classes()
    test_hook_class_candidates_simple_hover_rule_regression()
    test_hook_class_candidates_keyframes_only_untouched()
    test_hook_class_candidates_ignores_import_statement()
    test_regenerate_html_namespaces_colliding_hover_classes_across_chunks()
    test_regenerate_html_namespaces_colliding_keyframes_across_chunks()
    test_regenerate_html_does_not_rename_descendant_class_in_compound_hover_selector()
    test_regenerate_html_merges_multiple_raw_chunks()
    test_regenerate_chunk_prompt_includes_layout_preservation()
    print("All tests passed.")
