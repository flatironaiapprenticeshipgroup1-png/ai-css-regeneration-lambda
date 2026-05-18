import json
import os
from unittest.mock import MagicMock, patch

os.environ.setdefault("OPENAI_API_KEY", "test-key")


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


@patch("handler.client")
@patch("handler.s3")
def test_handler(mock_s3, mock_client):
    mock_s3.get_object.return_value = {
        "Body": MagicMock(read=lambda: b"body { color: red; }")
    }

    mock_completion = MagicMock()
    mock_completion.choices[0].message.content = "body { color: neon; }"
    mock_client.chat.completions.create.return_value = mock_completion

    from handler import lambda_handler

    result = lambda_handler(make_event(), None)

    assert result["statusCode"] == 200
    mock_s3.put_object.assert_called_once()
    call_kwargs = mock_s3.put_object.call_args.kwargs
    assert call_kwargs["Key"] == "test-123/Regenerated-Styles.css"
    assert call_kwargs["ContentType"] == "text/css"
    print("All assertions passed:", result)


if __name__ == "__main__":
    test_handler()
