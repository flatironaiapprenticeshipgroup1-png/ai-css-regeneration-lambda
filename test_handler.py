import json
import os
from unittest.mock import MagicMock, patch

os.environ.setdefault("BUCKET_NAME", "test-bucket")
os.environ.setdefault("SECRET_NAME", "test-secret")
os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")

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
    from handler import lambda_handler


def make_event(website_id="test-123", url="https://example.com", theme="cyberpunk"):
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


def test_handler():
    _mock_s3.get_object.return_value = {
        "Body": MagicMock(read=lambda: b"body { color: red; }")
    }
    mock_completion = MagicMock()
    mock_completion.choices[0].message.content = "body { color: neon; }"
    _mock_openai_client.chat.completions.create.return_value = mock_completion

    result = lambda_handler(make_event(), None)

    assert result["statusCode"] == 200
    _mock_s3.put_object.assert_called_once()
    call_kwargs = _mock_s3.put_object.call_args.kwargs
    assert call_kwargs["Key"] == "test-123/Regenerated-Styles.css"
    assert call_kwargs["ContentType"] == "text/css"
    print("All assertions passed:", result)


if __name__ == "__main__":
    test_handler()
