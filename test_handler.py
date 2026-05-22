import json
import os
import sys
from unittest.mock import MagicMock, patch

os.environ["BUCKET_NAME"] = "test-bucket"
os.environ["SECRET_NAME"] = "test/openai-secret"
os.environ["AWS_DEFAULT_REGION"] = "us-east-1"
os.environ["DYNAMODB_TABLE_NAME"] = "test-table"
os.environ["ABLY_SECRET_NAME"] = "test/ably-secret"

WEBSITE_ID = "abc-123"
URL = "https://example.com"
THEME = "dark"


def make_event(website_id=WEBSITE_ID, url=URL, theme=THEME):
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
    mock_s3 = MagicMock()
    mock_s3.get_object.return_value = {"Body": MagicMock(read=lambda: b"body { color: red; }")}

    mock_dynamodb = MagicMock()

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
    mock_ably_rest = MagicMock()
    mock_ably_rest.channels.get.return_value = mock_ably_channel

    return mock_s3, mock_dynamodb, mock_ably_channel, mock_ably_rest, mock_openai, boto3_factory


def _fresh_import(boto3_factory, mock_ably_rest, mock_openai):
    for mod in ["handler", "status_publisher"]:
        if mod in sys.modules:
            del sys.modules[mod]
    with patch("boto3.client", side_effect=boto3_factory), \
         patch("ably.AblyRest", return_value=mock_ably_rest), \
         patch("openai.OpenAI", return_value=mock_openai):
        import handler
        return handler


def test_happy_path_publishes_all_steps():
    mock_s3, mock_dynamodb, mock_channel, mock_ably_rest, mock_openai, boto3_factory = make_mocks()
    handler = _fresh_import(boto3_factory, mock_ably_rest, mock_openai)

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
    assert seqs == list(range(1, len(steps) + 1))

    last = mock_channel.publish.call_args_list[-1].args[1]
    assert last["status"] == "completed"

    assert mock_s3.put_object.call_count == 1
    key = mock_s3.put_object.call_args.kwargs["Key"]
    assert key == f"{WEBSITE_ID}/Regenerated-Styles.css"
    assert mock_dynamodb.update_item.call_count == len(steps)
    print("test_happy_path_publishes_all_steps: PASSED")


def test_openai_failure_publishes_failed():
    mock_s3, mock_dynamodb, mock_channel, mock_ably_rest, mock_openai, boto3_factory = make_mocks()
    mock_openai.chat.completions.create.side_effect = Exception("OpenAI error")
    handler = _fresh_import(boto3_factory, mock_ably_rest, mock_openai)

    result = handler.lambda_handler(make_event(), {})
    assert result["statusCode"] == 500

    steps = [c.args[1]["step"] for c in mock_channel.publish.call_args_list]
    assert "failed" in steps
    failed = next(c.args[1] for c in mock_channel.publish.call_args_list if c.args[1]["step"] == "failed")
    assert failed["status"] == "failed"
    assert failed["error"] is not None
    print("test_openai_failure_publishes_failed: PASSED")


def test_s3_write_failure_publishes_failed():
    mock_s3, mock_dynamodb, mock_channel, mock_ably_rest, mock_openai, boto3_factory = make_mocks()
    mock_s3.put_object.side_effect = Exception("S3 write error")
    handler = _fresh_import(boto3_factory, mock_ably_rest, mock_openai)

    result = handler.lambda_handler(make_event(), {})
    assert result["statusCode"] == 500

    steps = [c.args[1]["step"] for c in mock_channel.publish.call_args_list]
    assert "failed" in steps
    print("test_s3_write_failure_publishes_failed: PASSED")


def test_sequence_numbers_always_increase():
    mock_s3, mock_dynamodb, mock_channel, mock_ably_rest, mock_openai, boto3_factory = make_mocks()
    handler = _fresh_import(boto3_factory, mock_ably_rest, mock_openai)
    handler.lambda_handler(make_event(), {})

    seqs = [c.args[1]["sequence"] for c in mock_channel.publish.call_args_list]
    assert seqs == sorted(seqs) and len(seqs) == len(set(seqs))
    print("test_sequence_numbers_always_increase: PASSED")


if __name__ == "__main__":
    test_happy_path_publishes_all_steps()
    test_openai_failure_publishes_failed()
    test_s3_write_failure_publishes_failed()
    test_sequence_numbers_always_increase()
    print("All tests passed.")
