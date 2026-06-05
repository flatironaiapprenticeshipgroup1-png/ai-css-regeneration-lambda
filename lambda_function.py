import json
import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from status_publisher import get_current_sequence, publish_status_update

import boto3
import openai
from openai import OpenAI
from bs4 import BeautifulSoup

logging.getLogger().setLevel(logging.INFO)
logger = logging.getLogger(__name__)

s3 = boto3.client("s3")
secrets_client = boto3.client("secretsmanager")
dynamodb = boto3.resource("dynamodb")

MAX_CHARS_PER_CHUNK = 30_000


def parse_css_blocks(css: str) -> list[str]:
    """Parse CSS into individual top-level rule blocks (no size grouping)."""
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

    return blocks


def split_html_into_chunks(html: str, max_chars: int = MAX_CHARS_PER_CHUNK) -> tuple[list[str], list[str]]:
    """Split HTML into (chunks, labels) where each label is 'head', 'body', or 'raw'."""
    soup = BeautifulSoup(html, "html.parser")
    head = soup.find("head")
    body = soup.find("body")

    if not head or not body:
        return [html], ["raw"]

    chunks = [str(head)]
    labels = ["head"]

    current_parts: list[str] = []
    current_size = 0

    for child in body.children:
        child_str = str(child)
        child_size = len(child_str)
        if current_parts and current_size + child_size > max_chars:
            chunks.append("".join(current_parts))
            labels.append("body")
            current_parts = [child_str]
            current_size = child_size
        else:
            current_parts.append(child_str)
            current_size += child_size

    if current_parts:
        chunks.append("".join(current_parts))
        labels.append("body")

    return chunks, labels


def extract_selectors(html_chunk: str) -> set[str]:
    """Extract class names, IDs, and tag names referenced in an HTML chunk."""
    soup = BeautifulSoup(html_chunk, "html.parser")
    selectors: set[str] = set()
    for tag in soup.find_all(True):
        selectors.add(tag.name)
        for cls in tag.get("class", []):
            selectors.add(cls)
        if tag.get("id"):
            selectors.add(tag["id"])
    return selectors


def get_css_for_chunk(css_blocks: list[str], selectors: set[str]) -> str:
    """Return CSS blocks that reference the given selectors, falling back to all blocks."""
    relevant = [b for b in css_blocks if any(s in b for s in selectors)]
    return "\n\n".join(relevant) if relevant else "\n\n".join(css_blocks)


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
            table = dynamodb.Table(os.environ["DYNAMODB_TABLE_NAME"])
            existing = table.get_item(
                Key={"RegeneratedWebsiteId": website_id, "RegeneratedWebsiteUrl": website_url}
            ).get("Item", {})
            current_status = existing.get("RegenerationStatus")
            if current_status in ("ai_lambda_processing", "completed"):
                logger.warning("Skipping duplicate invocation for %s: status is already '%s'", website_id, current_status)
                publish(step="Finalizing", status="completed", message="Finished regeneration")
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

            def regenerate_chunk(
                chunk_client: OpenAI,
                html_chunk: str,
                label: str,
                css_chunk: str,
                chunk_index: int,
                total_chunks: int,
            ) -> dict:
                """
                Regenerate an HTML+CSS pair. Returns {"html": "...", "css": "..."}.

                Head chunks regenerate HTML only (updating stylesheet link) and return css="".
                Body chunks regenerate both HTML and CSS together so the AI has full context
                about the markup it is styling.
                """
                if label == "head":
                    system_msg = (
                        f"You are an HTML expert for a website theme regeneration system.\n\n"
                        f"Transform the <head> contents for the theme: {regeneration_theme}\n\n"
                        f"REQUIRED:\n"
                        f"- Replace ALL <link rel=\"stylesheet\"> tags with exactly one: "
                        f"<link rel=\"stylesheet\" href=\"./Regenerated-Styles.css\">\n"
                        f"- Remove ALL <style> blocks\n"
                        f"- Update <title> to reflect the theme\n\n"
                        f"RETURN FORMAT:\n"
                        f"Return valid JSON with key \"html\" containing the inner <head> contents "
                        f"and key \"css\" as an empty string. No <head> wrapper tags, no markdown."
                    )
                    user_msg = (
                        f"Chunk {chunk_index + 1} of {total_chunks}. "
                        f"Regenerate this HTML head:\n\n{html_chunk}"
                    )
                else:
                    system_msg = (
                        f"You are an elite HTML and CSS web design expert. "
                        f"Completely rebuild both the HTML and its CSS for the theme: {regeneration_theme}\n\n"
                        f"You receive an HTML fragment and the CSS rules that apply to it. "
                        f"Regenerate BOTH completely so the theme is applied consistently.\n\n"
                        f"HTML RULES:\n"
                        f"- Completely restructure the HTML — new section order, semantic tags "
                        f"(<section>, <article>, <header>, <nav>, <figure>, etc.)\n"
                        f"- Add eye-catching decorative elements that fit the theme\n"
                        f"- Use data-* attributes for theme effects (data-glitch, data-parallax, etc.)\n"
                        f"- Rewrite headings and copy to match the theme tone\n"
                        f"- Preserve factual content (product names, prices, key data)\n"
                        f"- Use inline styles for LAYOUT ONLY (flex, grid, position, width, height, gap)\n"
                        f"- Do NOT use inline styles for visual properties (colors, fonts, borders, shadows)\n"
                        f"- Do NOT include <head>, <html>, or <body> wrapper tags\n\n"
                        f"CSS RULES:\n"
                        f"- Transform ALL selectors with theme-appropriate typography, colors, spacing, "
                        f"borders, shadows, and animations\n"
                        f"- Build a cohesive color palette — not just individual color swaps\n"
                        f"- Keep ALL original selectors — do not remove any\n"
                        f"- Add hover effects and keyframe animations that fit the theme\n"
                        f"- Add dramatic background animations to make the site visually engaging\n"
                        f"- Return only valid CSS — no explanations, no code fences\n\n"
                        f"IT IS VERY IMPORTANT THAT THE WEBSITE LOOKS CLEAN AND NOT CLUNKY\n\n"
                        f"RETURN FORMAT:\n"
                        f"Return valid JSON with exactly two keys:\n"
                        f"  \"html\": the regenerated HTML fragment\n"
                        f"  \"css\": the regenerated CSS rules for this section\n"
                        f"No other keys, no markdown, no code fences around the JSON."
                    )
                    user_msg = (
                        f"Chunk {chunk_index + 1} of {total_chunks}.\n\n"
                        f"HTML:\n{html_chunk}\n\n"
                        f"CSS for this section:\n{css_chunk}"
                    )

                logger.info(
                    "[OpenAI] Sending chunk %d/%d (%s) | input_chars=%d | max_tokens=16384",
                    chunk_index + 1, total_chunks, label, len(html_chunk) + len(css_chunk),
                )

                try:
                    response = chunk_client.chat.completions.create(
                        model="gpt-4o",
                        messages=[
                            {"role": "system", "content": system_msg},
                            {"role": "user", "content": user_msg},
                        ],
                        max_tokens=16384,
                        response_format={"type": "json_object"},
                    )
                except openai.RateLimitError:
                    logger.error("[OpenAI] Rate limit on chunk %d/%d", chunk_index + 1, total_chunks, exc_info=True)
                    raise
                except openai.APITimeoutError:
                    logger.error("[OpenAI] Timeout on chunk %d/%d", chunk_index + 1, total_chunks, exc_info=True)
                    raise
                except openai.APIStatusError as e:
                    logger.error(
                        "[OpenAI] API error on chunk %d/%d | status=%s | message=%s",
                        chunk_index + 1, total_chunks, e.status_code, e.message, exc_info=True,
                    )
                    raise
                except openai.APIError:
                    logger.error("[OpenAI] Unexpected error on chunk %d/%d", chunk_index + 1, total_chunks, exc_info=True)
                    raise

                finish_reason = response.choices[0].finish_reason
                prompt_tokens = response.usage.prompt_tokens
                completion_tokens = response.usage.completion_tokens
                input_cost = prompt_tokens / 1_000_000 * 2.50
                output_cost = completion_tokens / 1_000_000 * 10.00

                logger.info(
                    "[OpenAI] Chunk %d/%d complete | finish_reason=%s | prompt_tokens=%d | "
                    "completion_tokens=%d | cost=$%.6f",
                    chunk_index + 1, total_chunks, finish_reason, prompt_tokens, completion_tokens,
                    input_cost + output_cost,
                )
                if finish_reason != "stop":
                    logger.warning(
                        "[OpenAI] Chunk %d/%d finish_reason='%s' — output may be truncated",
                        chunk_index + 1, total_chunks, finish_reason,
                    )

                publish(
                    step="regenerating_chunk_completed",
                    status="ai_lambda_processing",
                    message=f"Regenerated HTML and CSS chunk {chunk_index + 1} of {total_chunks}",
                )

                return json.loads(response.choices[0].message.content)

            # Read original HTML and CSS from S3
            html_obj = s3.get_object(Bucket=os.environ["BUCKET_NAME"], Key=f"{website_id}/index.html")
            original_html = html_obj["Body"].read().decode("utf-8")
            logger.info("HTML file size: %d characters", len(original_html))

            css_obj = s3.get_object(Bucket=os.environ["BUCKET_NAME"], Key=f"{website_id}/original-styles.css")
            original_css = css_obj["Body"].read().decode("utf-8")
            logger.info("CSS file size: %d characters", len(original_css))

            # Parse CSS into individual blocks so we can filter by selector relevance per chunk
            css_blocks = parse_css_blocks(original_css)
            logger.info("Parsed CSS into %d individual blocks", len(css_blocks))

            # Split HTML into chunks and pair each body chunk with its relevant CSS
            publish(
                step="chunking",
                status="ai_lambda_processing",
                message="Pairing HTML sections with their CSS for AI processing",
            )
            html_chunks, labels = split_html_into_chunks(original_html)
            logger.info("Split HTML into %d chunk(s) for processing", len(html_chunks))

            css_per_chunk: list[str] = []
            for chunk, label in zip(html_chunks, labels):
                if label == "body":
                    selectors = extract_selectors(chunk)
                    css_per_chunk.append(get_css_for_chunk(css_blocks, selectors))
                else:
                    css_per_chunk.append("")

            total_chunks = len(html_chunks)
            results: dict[int, dict] = {}

            publish(
                step="regenerating_html_and_css",
                status="ai_lambda_processing",
                message="AI regenerating HTML and CSS together for each section",
            )
            with ThreadPoolExecutor(max_workers=total_chunks) as executor:
                futures = {
                    executor.submit(
                        regenerate_chunk, client,
                        html_chunks[i], labels[i], css_per_chunk[i],
                        i, total_chunks,
                    ): i
                    for i in range(total_chunks)
                }
                for future in as_completed(futures):
                    idx = futures[future]
                    try:
                        results[idx] = future.result()
                    except Exception:
                        logger.error("[OpenAI] Chunk %d/%d failed — aborting", idx + 1, total_chunks, exc_info=True)
                        raise

            ordered = [results[i] for i in range(total_chunks)]

            if labels[0] == "raw":
                regenerated_html = ordered[0].get("html", "")
                regenerated_css = ordered[0].get("css", "")
            else:
                head_html = ordered[0].get("html", "")
                body_parts = [r.get("html", "") for r in ordered[1:]]
                regenerated_html = (
                    f"<!DOCTYPE html>\n<html>\n"
                    f"<head>{head_html}</head>\n"
                    f"<body>{''.join(body_parts)}</body>\n"
                    f"</html>"
                )
                regenerated_css = "\n\n".join(r.get("css", "") for r in ordered[1:] if r.get("css"))

            logger.info("Regenerated HTML size: %d characters", len(regenerated_html))
            logger.info("Regenerated CSS size: %d characters", len(regenerated_css))

            s3.put_object(
                Bucket=os.environ["BUCKET_NAME"],
                Key=f"{website_id}/Regenerated-Index.html",
                Body=regenerated_html.encode("utf-8"),
                ContentType="text/html",
                CacheControl="no-store, no-cache, must-revalidate",
            )
            s3.put_object(
                Bucket=os.environ["BUCKET_NAME"],
                Key=f"{website_id}/Regenerated-Styles.css",
                Body=regenerated_css.encode("utf-8"),
                ContentType="text/css",
                CacheControl="no-store, no-cache, must-revalidate",
            )
            logger.info("Regenerated HTML and CSS saved to S3 for website ID %s", website_id)

            table.update_item(
                Key={"RegeneratedWebsiteId": website_id, "RegeneratedWebsiteUrl": website_url},
                UpdateExpression="SET RegenerationStatus = :status",
                ExpressionAttributeValues={":status": "completed"},
            )
            logger.info("DynamoDB status updated to completed for website ID %s", website_id)
            publish(step="Finalizing", status="completed", message="Finished regenerating HTML and CSS")

        return {
            "statusCode": 200,
            "body": json.dumps({"message": "Regeneration completed successfully", "websiteId": website_id}),
        }
    except Exception as e:
        logger.error("Error processing regeneration request", exc_info=True)
        return {
            "statusCode": 500,
            "body": json.dumps({"message": "An error occurred during regeneration", "error": str(e)}),
        }
