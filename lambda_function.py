import json
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from status_publisher import get_current_sequence, publish_status_update

import boto3
from openai import OpenAI

s3 = boto3.client("s3")
secrets_client = boto3.client("secretsmanager")
dynamodb = boto3.resource("dynamodb")
MAX_CHARS_PER_CHUNK = 30_000
MAX_CONCURRENT_CHUNK_REQUESTS = 10

_LEADING_CODE_FENCE_RE = re.compile(r"^\s*```[a-zA-Z0-9_-]*[ \t]*\n")
_TRAILING_CODE_FENCE_RE = re.compile(r"\n?[ \t]*```\s*$")


def _strip_code_fences(text: str) -> str:
    """
    Strips a leading ```css-style fence and/or trailing ``` fence the model
    added despite being told not to. The two fences are stripped
    independently so a response truncated at max_tokens before the model
    closed its fence still has the leading one removed.
    """
    if not text:
        return text
    stripped = _LEADING_CODE_FENCE_RE.sub("", text, count=1)
    stripped = _TRAILING_CODE_FENCE_RE.sub("", stripped, count=1)
    return stripped


def parse_css_blocks(css: str) -> list[str]:
    """Split CSS into top-level rule blocks (selector(s) + braces + body),
    tracking brace depth so nested rules (e.g. inside @media) stay intact
    and quoted strings don't confuse brace counting."""
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


def extract_selectors(block: str) -> list[str]:
    """Return the comma-separated selectors (or @-rule prelude) preceding a block's opening brace."""
    head = block.split("{", 1)[0]
    return [s.strip() for s in head.split(",") if s.strip()]


def selector_present(selector: str, css: str) -> bool:
    """Check whether a selector token still appears in regenerated CSS, guarding
    against partial matches (e.g. '.gb_H' incorrectly matching inside '.gb_HX')."""
    pattern = re.escape(selector)
    return re.search(rf"(?<![\w-]){pattern}(?![\w-])", css) is not None


def find_missing_blocks(original_chunk: str, regenerated_css: str) -> list[str]:
    """Return original rule blocks whose selectors are entirely absent from the
    regenerated CSS — i.e. blocks the model dropped rather than rewrote."""
    missing = []
    for block in parse_css_blocks(original_chunk):
        selectors = extract_selectors(block)
        if not selectors:
            continue
        if not any(selector_present(sel, regenerated_css) for sel in selectors):
            missing.append(block)
    return missing


def split_css_into_chunks(css: str, max_chars: int = MAX_CHARS_PER_CHUNK) -> list[str]:
    blocks = parse_css_blocks(css)
    chunks = []
    current_chunk_parts = []
    current_chunk_size = 0

    for block in blocks:
        block_size = len(block)

        # A single block (e.g. a @font-face rule with an embedded base64
        # data URI) can itself exceed max_chars. Splitting can't respect CSS
        # syntax at this size, so just slice it into max_chars-sized pieces.
        if block_size > max_chars:
            if current_chunk_parts:
                chunks.append("\n\n".join(current_chunk_parts))
                current_chunk_parts = []
                current_chunk_size = 0
            for start in range(0, block_size, max_chars):
                chunks.append(block[start:start + max_chars])
            continue

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

            def regenerate_css_chunk(
                client: OpenAI,
                chunk: str,
                theme_prompt: str,
                chunk_index: int,
                total_chunks: int,
            ) -> str:
                theme_description = (
                    regeneration_theme if regeneration_theme
                    else "modern practices while maintaining the original feel"
                )
                system_msg = (
                    f"""You are a CSS and web design expert specializing in dramatic visual transformations.

                        You will receive chunks of a CSS file. Rewrite them completely to match this theme: {theme_description}

                        You MUST change ALL of the following — not just colors:

                        TYPOGRAPHY:
                        Replace every font-family declaration with theme-appropriate fonts,
                        Use @import to load Google Fonts if needed (add at the top),
                        Change font sizes, weights, letter-spacing, and line-height to match the theme,

                        COLORS:
                        If the theme names or strongly implies a specific color or hue (e.g. "neon pink", "forest green", "royal purple", "sunset orange"), that color IS the anchor of the palette — use it prominently and repeatedly as the dominant color, not as a token accent buried in a single rule,
                        Derive every other color in the palette FROM that anchor — complementary/analogous hues, plus lighter tints and darker shades of the anchor itself for hover states, borders, and backgrounds — rather than inventing unrelated colors,
                        If the theme does not name a color, choose one cohesive anchor hue that fits the theme's mood and build the palette the same way,
                        Replace every background-color, color, and border-color using this palette,
                        Apply the palette consistently across all elements — the same handful of colors (plus their tints/shades) should recur throughout the whole stylesheet, not a different color per selector,

                        BORDERS & SHAPES:
                        Change border styles, widths, and border-radius values,
                        A futuristic theme might use sharp corners; organic themes use rounded ones,

                        SPACING & LAYOUT:
                        Change padding and margin values to reflect the theme's density,
                        Compact themes feel tight; luxurious themes use generous whitespace,

                        DECORATIVE EFFECTS:
                        Add or rewrite box-shadow, text-shadow, and gradients,
                        Use background-image gradients where appropriate,

                        IT IS VERY IMPORTANT THAT THE WEBSITE LOOKS CLEAN AND NOT CLUNKY/MESSY

                        RULES:
                        Return ONLY valid CSS — no explanations, no markdown, no code fences,
                        Do not remove any CSS selectors or classes — every original selector must appear in your output,
                        Your output must contain at least as many rule blocks as the input — recount before responding if you are unsure you covered every selector,
                        Do not add or reference HTML elements that don't exist in the original,
                        Do not change width, height, max-width, max-height, object-fit, or aspect-ratio on img, picture, svg, video, or canvas elements — copy those values through unchanged so images and svgs keep their original size,
                        The transformation must be immediately obvious at a glance"""
                )
                chunk_blocks = parse_css_blocks(chunk)
                chunk_selectors = [sel for block in chunk_blocks for sel in extract_selectors(block)]
                user_msg = (
                    f"{theme_prompt}\n\n"
                    f"This is chunk {chunk_index + 1} of {total_chunks} from the full stylesheet. "
                    f"This chunk has exactly {len(chunk_blocks)} CSS rule blocks, covering these selectors, "
                    f"every one of which MUST appear in your output (do not omit, merge, or rename any): "
                    f"{', '.join(chunk_selectors)}\n\n"
                    f"Regenerate this css: {chunk}"
                )
                print(f"Processing chunk {chunk_index + 1}/{total_chunks} ({len(chunk)} chars)...")
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": user_msg},
                    ],
                    max_tokens=16384,
                )

                publish(
                    step=f"regenerating_css_chunks_completed",
                    status="ai_lambda_processing",
                    message=f"Regenerated chunk {chunk_index + 1} of {total_chunks}"
                )

                regenerated_css = _strip_code_fences(response.choices[0].message.content)
                missing_blocks = find_missing_blocks(chunk, regenerated_css)
                if missing_blocks:
                    print(
                        f"Chunk {chunk_index + 1}/{total_chunks}: model dropped {len(missing_blocks)} "
                        f"rule block(s); restoring original CSS for those selectors so elements don't "
                        f"fall back to unstyled/default sizing"
                    )
                    regenerated_css += (
                        "\n\n/* Restored: original rules omitted by AI regeneration */\n"
                        + "\n\n".join(missing_blocks)
                    )
                return regenerated_css


            response = s3.get_object(
                Bucket=os.environ["BUCKET_NAME"],
                Key=f"{website_id}/original-styles.css",
            )
            content = response["Body"].read().decode("utf-8")
            print(f"CSS file size: {len(content)} characters")

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
            print(f"Split CSS into {len(chunks)} chunk(s) for processing")

            # process all chunks in parallel (I/O-bound — threads wait on OpenAI, not CPU)
            publish(step="regenerating_css", status="ai_lambda_processing", message="Ai Regenerating Styling CSS for the website")
            if not chunks:
                # Original stylesheet had no rule blocks (e.g. an empty/whitespace-only
                # original-styles.css from a legacy or retried job) — nothing to send
                # to the model. ThreadPoolExecutor requires max_workers > 0, so this
                # must be short-circuited rather than handed to the pool below.
                print("No CSS rule blocks to regenerate; producing empty stylesheet")
                regenerated_css = ""
            else:
                results = {}
                with ThreadPoolExecutor(max_workers=min(len(chunks), MAX_CONCURRENT_CHUNK_REQUESTS)) as executor:
                    futures = {
                        executor.submit(regenerate_css_chunk, client, chunk, theme_prompt, i, len(chunks)): i
                        for i, chunk in enumerate(chunks)
                    }
                    for future in as_completed(futures):
                        idx = futures[future]
                        results[idx] = future.result()

                regenerated_parts = [results[i] for i in range(len(chunks))]
                regenerated_css = "\n\n".join(regenerated_parts)

            print(f"Regenerated CSS total size: {len(regenerated_css)} characters")

            s3.put_object(
                Bucket=os.environ["BUCKET_NAME"],
                Key=f"{website_id}/Regenerated-Styles.css",
                Body=regenerated_css.encode("utf-8"),
                ContentType="text/css",
                CacheControl="no-store, no-cache, must-revalidate",
            )
            print(f"Regenerated CSS saved to S3 for website ID {website_id}")

            table.update_item(
                Key={
                    "RegeneratedWebsiteId": website_id,
                    "RegeneratedWebsiteUrl": website_url,
                },
                UpdateExpression="SET RegenerationStatus = :status",
                ExpressionAttributeValues={":status": "completed"},
            )
            print(f"DynamoDB status updated to completed for website ID {website_id}")
            publish(step="Finalizing", status="completed", message="Finished Css Regeneration")

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
