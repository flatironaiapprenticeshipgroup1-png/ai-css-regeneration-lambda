import json
import os
import threading
from status_publisher import get_current_sequence, publish_status_update

import boto3
from openai import OpenAI

from inline_html_regenerator import regenerate_html

s3 = boto3.client("s3")
secrets_client = boto3.client("secretsmanager")
dynamodb = boto3.resource("dynamodb")


def lambda_handler(event, context):
    secret = json.loads(
        secrets_client.get_secret_value(SecretId=os.environ["SECRET_NAME"])["SecretString"]
    )
    client = OpenAI(api_key=secret["OpenAIAPIKey"])

    failed_message_ids = []

    for record in event["Records"]:
        body = json.loads(record["body"])
        website_id = body["RegeneratedWebsiteId"]
        website_url = body["RegeneratedWebsiteUrl"]
        regeneration_theme = body.get("RegenerationTheme")

        try:
            print(f"Received regeneration request for website ID: {website_id}")
            print(f"Website URL: {website_url}")
            print(f"Regeneration theme: {regeneration_theme}")

            # Idempotency guard: skip if another Lambda invocation already claimed this job.
            # SQS delivers at-least-once, so the same message can arrive while a prior
            # invocation is still running (visibility timeout expired) or after a crash.
            table = dynamodb.Table(os.environ["DYNAMODB_TABLE_NAME"])
            existing = table.get_item(
                Key={"RegeneratedWebsiteId": website_id, "RegeneratedWebsiteUrl": website_url}
            ).get("Item", {})
            current_status = existing.get("RegenerationStatus")
            if current_status in ("ai_lambda_processing", "completed"):
                print(f"Skipping duplicate invocation for {website_id}: status is already '{current_status}'")
                continue

            table.update_item(
                Key={"RegeneratedWebsiteId": website_id, "RegeneratedWebsiteUrl": website_url},
                UpdateExpression="SET RegenerationStatus = :s",
                ExpressionAttributeValues={":s": "processing"},
            )

            seq = get_current_sequence(website_id, website_url)
            seq_lock = threading.Lock()

            def publish(step, status, message, result_url=None, error=None):
                nonlocal seq
                with seq_lock:
                    seq += 1
                    current_seq = seq
                publish_status_update(
                    website_id=website_id,
                    website_url=website_url,
                    phase="ai",
                    step=step,
                    status=status,
                    message=message,
                    sequence=current_seq,
                    result_url=result_url,
                    error=error,
                )

            def on_chunk_complete(chunk_index, total_chunks):
                publish(
                    step="regenerating_html_and_styling_chunks_completed",
                    status="ai_lambda_processing",
                    message=f"Regenerated chunk {chunk_index + 1} of {total_chunks}"
                )

            response = s3.get_object(
                Bucket=os.environ["BUCKET_NAME"],
                Key=f"{website_id}/Regenerated-Index.html",
            )
            content = response["Body"].read().decode("utf-8")
            print(f"HTML file size: {len(content)} characters")

            if regeneration_theme is None:
                theme_prompt = (
                    "Regenerate the page using modern practices while maintaining the original feel."
                )
            else:
                theme_prompt = (
                    f"Regenerate the page using the theme: {regeneration_theme}."
                )

            publish(step="chunking", status="ai_lambda_processing", message="Compressing HTML into chunks for processing")
            publish(step="regenerating_html_and_styling", status="ai_lambda_processing", message="Ai Regenerating HTML and styling for the website")
            regenerated_html = regenerate_html(client, content, theme_prompt, regeneration_theme, on_chunk_complete)
            del content  # drop the original full-document string before the encode/upload below
            print(f"Regenerated HTML total size: {len(regenerated_html)} characters")

            s3.put_object(
                Bucket=os.environ["BUCKET_NAME"],
                Key=f"{website_id}/Regenerated-Index.html",
                Body=regenerated_html.encode("utf-8"),
                ContentType="text/html; charset=utf-8",
                CacheControl="no-store, no-cache, must-revalidate",
            )
            print(f"Regenerated HTML saved to S3 for website ID {website_id}")

            table.update_item(
                Key={
                    "RegeneratedWebsiteId": website_id,
                    "RegeneratedWebsiteUrl": website_url,
                },
                UpdateExpression="SET RegenerationStatus = :status",
                ExpressionAttributeValues={":status": "completed"},
            )
            print(f"DynamoDB status updated to completed for website ID {website_id}")
            publish(step="Finalizing", status="completed", message="Finished HTML and CSS Regeneration")

        except Exception as e:
            print(f"Error processing record for {website_id}: {e}")
            # Publish terminal failed event if publish() is defined (it may not be if error happened very early)
            try:
                publish("failed", "failed", f"AI regeneration failed: {e}", error=str(e))
            except Exception:
                pass
            # Persist failure to DynamoDB
            try:
                table.update_item(
                    Key={"RegeneratedWebsiteId": website_id, "RegeneratedWebsiteUrl": website_url},
                    UpdateExpression="SET RegenerationStatus = :s, ErrorMessage = :e",
                    ExpressionAttributeValues={":s": "failed", ":e": str(e)},
                )
            except Exception:
                pass
            failed_message_ids.append({"itemIdentifier": record.get("messageId", website_id)})

    return {"batchItemFailures": failed_message_ids}
