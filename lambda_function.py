import json
import logging
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from status_publisher import get_current_sequence, publish_status_update

import boto3
import openai
from openai import OpenAI

logging.getLogger().setLevel(logging.INFO)
logger = logging.getLogger(__name__)

s3 = boto3.client("s3")
secrets_client = boto3.client("secretsmanager")
dynamodb = boto3.resource("dynamodb")
MAX_CHARS_PER_CHUNK = 30_000

def split_css_into_chunks(css: str, max_chars: int = MAX_CHARS_PER_CHUNK) -> list[str]:
    blocks = []
    current = []
    depth = 0
    i = 0

    while i < len(css):
        ch = css[i]
        if ch in ('"', "'"):
            quote = ch
            current.append(ch)
            i += 1
            while i < len(css) and css[i] != quote:
                if css[i] == "\\" and i + 1 < len(css):
                    current.append(css[i])
                    i += 1
                current.append(css[i])
                i += 1
            if i < len(css):
                current.append(css[i])
            i += 1
            continue

        current.append(ch)

        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                block = "".join(current).strip()
                if block:
                    blocks.append(block)
                current = []

        i += 1

    trailing = "".join(current).strip()
    if trailing:
        blocks.insert(0, trailing)

    chunks = []
    current_chunk_parts = []
    current_chunk_size = 0

    for block in blocks:
        block_size = len(block)
        if current_chunk_parts and current_chunk_size + block_size > max_chars:
            chunks.append("\n\n".join(current_chunk_parts))
            current_chunk_parts = [block]
            current_chunk_size = block_size
        else:
            current_chunk_parts.append(block)
            current_chunk_size += block_size

    if current_chunk_parts:
        chunks.append("\n\n".join(current_chunk_parts))

    return chunks



def lambda_handler(event, context):
    try:
        secret = json.loads(
            secrets_client.get_secret_value(SecretId=os.environ["SECRET_NAME"])["SecretString"]
        )
        client = OpenAI(api_key=secret["OpenAIAPIKey"])

        for record in event["Records"]:
            body = json.loads(record["body"])
            website_id = body["RegeneratedWebsiteId"]
            logger.info("Received regeneration request for website ID: %s", website_id)
            website_url = body["RegeneratedWebsiteUrl"]
            logger.info("Website URL: %s", website_url)
            regeneration_theme = body.get("RegenerationTheme")
            logger.info("Regeneration theme: %s", regeneration_theme)

            # Idempotency guard: skip if another Lambda invocation already claimed this job.
            # SQS delivers at-least-once, so the same message can arrive while a prior
            # invocation is still running (visibility timeout expired) or after a crash.
            table = dynamodb.Table(os.environ["DYNAMODB_TABLE_NAME"])
            existing = table.get_item(
                Key={"RegeneratedWebsiteId": website_id, "RegeneratedWebsiteUrl": website_url}
            ).get("Item", {})
            current_status = existing.get("RegenerationStatus")
            if current_status in ("ai_lambda_processing", "completed"):
                logger.warning("Skipping duplicate invocation for %s: status is already '%s'", website_id, current_status)
                publish(step="Finalizing", status="completed", message="Finished Css Regeneration")
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
                    phase="crawler",
                    step=step,
                    status=status,
                    message=message,
                    sequence=current_seq,
                    result_url=result_url,
                    error=error,
                )

            def regenerate_css_chunk(
                client: OpenAI,
                chunk: str,
                theme_prompt: str,
                chunk_index: int,
                total_chunks: int,
            ) -> str:
                system_msg = (
                    f"""You are a CSS and web design expert specializing in dramatic visual transformations.

                        You will receive chunks of a CSS file. Rewrite them completely to match this theme: {regeneration_theme}

                        You MUST change ALL of the following — not just colors:

                        TYPOGRAPHY:
                        Replace every font-family declaration with theme-appropriate fonts,
                        Use @import to load Google Fonts if needed (add at the top),
                        Change font sizes, weights, letter-spacing, and line-height to match the theme,

                        COLORS:
                        Replace every background-color, color, and border-color,
                        Build a cohesive color palette — do not just swap one color for another,
                        Apply the palette consistently across all elements,

                        BORDERS & SHAPES:
                        Change border styles, widths, and border-radius values,
                        A futuristic theme might use sharp corners; organic themes use rounded ones,

                        SPACING & LAYOUT:
                        Change padding and margin values to reflect the theme's density,
                        Compact themes feel tight; luxurious themes use generous whitespace,

                        DECORATIVE EFFECTS:
                        Add or rewrite box-shadow, text-shadow, and gradients,
                        Use background-image gradients where appropriate,

                        ANIMATIONS:
                        Add animations like hover effects or keyframe animations that fit the theme

                        also add cool dramatic animations in the background to make the website more visually appealing and engaging

                        IT IS VERY IMPORTANT THAT THE WEBSITE LOOKS CLEAN AND NOT CLUNKY/MESSY

                        RULES:
                        Return ONLY valid CSS — no explanations, no markdown, no code fences,
                        Do not remove any CSS selectors or classes — every original selector must appear in your output,
                        Do not add or reference HTML elements that don't exist in the original,
                        The transformation must be immediately obvious at a glance"""
                )
                user_msg = (
                    f"{theme_prompt}\n\n"
                    f"This is chunk {chunk_index + 1} of {total_chunks} from the full stylesheet. "
                    f"Regenerate this css: {chunk}"
                )
                logger.info(
                    "[OpenAI] Sending chunk %d/%d to gpt-4o | input_chars=%d | max_tokens=16384",
                    chunk_index + 1, total_chunks, len(chunk),
                )
                try:
                    response = client.chat.completions.create(
                        model="gpt-4o",
                        messages=[
                            {"role": "system", "content": system_msg},
                            {"role": "user", "content": user_msg},
                        ],
                        max_tokens=16384,
                    )
                except openai.RateLimitError:
                    logger.error(
                        "[OpenAI] Rate limit hit on chunk %d/%d — request rejected by OpenAI",
                        chunk_index + 1, total_chunks, exc_info=True,
                    )
                    raise
                except openai.APITimeoutError:
                    logger.error(
                        "[OpenAI] Request timed out on chunk %d/%d",
                        chunk_index + 1, total_chunks, exc_info=True,
                    )
                    raise
                except openai.APIStatusError as e:
                    logger.error(
                        "[OpenAI] API error on chunk %d/%d | status=%s | message=%s",
                        chunk_index + 1, total_chunks, e.status_code, e.message, exc_info=True,
                    )
                    raise
                except openai.APIError:
                    logger.error(
                        "[OpenAI] Unexpected API error on chunk %d/%d",
                        chunk_index + 1, total_chunks, exc_info=True,
                    )
                    raise

                finish_reason = response.choices[0].finish_reason
                prompt_tokens = response.usage.prompt_tokens
                completion_tokens = response.usage.completion_tokens
                input_cost  = prompt_tokens     / 1_000_000 * 2.50
                output_cost = completion_tokens / 1_000_000 * 10.00
                total_cost  = input_cost + output_cost

                logger.info(
                    "[OpenAI] Chunk %d/%d complete | finish_reason=%s | prompt_tokens=%d | completion_tokens=%d | cost=$%.6f",
                    chunk_index + 1, total_chunks, finish_reason, prompt_tokens, completion_tokens, total_cost,
                )
                if finish_reason != "stop":
                    logger.warning(
                        "[OpenAI] Chunk %d/%d finish_reason='%s' — output may be truncated or incomplete",
                        chunk_index + 1, total_chunks, finish_reason,
                    )

                publish(
                    step="regenerating_css_chunks_completed",
                    status="ai_lambda_processing",
                    message=f"Regenerated chunk {chunk_index + 1} of {total_chunks}"
                )

                return response.choices[0].message.content
            
            
            response = s3.get_object(
                Bucket=os.environ["BUCKET_NAME"],
                Key=f"{website_id}/original-styles.css",
            )
            content = response["Body"].read().decode("utf-8")
            logger.info("CSS file size: %d characters", len(content))

            if regeneration_theme is None:
                theme_prompt = (
                    "Regenerate the CSS using modern practices while maintaining the original feel."
                )
            else:
                theme_prompt = (
                    f"Regenerate the CSS using the theme: {regeneration_theme}."
                )

            # Split into chunks if the file is large
            publish(step="chunking", status="ai_lambda_processing", message="Compressing CSS into chunks for processing")
            chunks = split_css_into_chunks(content)
            logger.info("Split CSS into %d chunk(s) for processing", len(chunks))

            # process all chunks in parallel (I/O-bound — threads wait on OpenAI, not CPU)
            results = {}
            publish(step="regenerating_css", status="ai_lambda_processing", message="Ai Regenerating Styling CSS for the website")
            with ThreadPoolExecutor(max_workers=len(chunks)) as executor:
                futures = {
                    executor.submit(regenerate_css_chunk, client, chunk, theme_prompt, i, len(chunks)): i
                    for i, chunk in enumerate(chunks)
                }
                for future in as_completed(futures):
                    idx = futures[future]
                    try:
                        results[idx] = future.result()
                    except Exception:
                        logger.error(
                            "[OpenAI] Chunk %d/%d failed — aborting regeneration",
                            idx + 1, len(chunks), exc_info=True,
                        )
                        raise
            
            regenerated_parts = [results[i] for i in range(len(chunks))]

            regenerated_css = "\n\n".join(regenerated_parts)
            logger.info("Regenerated CSS total size: %d characters", len(regenerated_css))

            s3.put_object(
                Bucket=os.environ["BUCKET_NAME"],
                Key=f"{website_id}/Regenerated-Styles.css",
                Body=regenerated_css.encode("utf-8"),
                ContentType="text/css",
                CacheControl="no-store, no-cache, must-revalidate",
            )
            logger.info("Regenerated CSS saved to S3 for website ID %s", website_id)

            table.update_item(
                Key={
                    "RegeneratedWebsiteId": website_id,
                    "RegeneratedWebsiteUrl": website_url,
                },
                UpdateExpression="SET RegenerationStatus = :status",
                ExpressionAttributeValues={":status": "completed"},
            )
            logger.info("DynamoDB status updated to completed for website ID %s", website_id)
            publish(step="Finalizing", status="completed", message="Finished Css Regeneration")

        return {
            "statusCode": 200,
            "body": json.dumps(
                {
                    "message": "Regeneration completed successfully",
                    "websiteId": website_id,
                }
            ),
        }
    except Exception as e:
        logger.error("Error processing regeneration request", exc_info=True)
        return {
            "statusCode": 500,
            "body": json.dumps(
                {"message": "An error occurred during regeneration", "error": str(e)}
            ),
        }
