import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

import boto3
from openai import OpenAI

s3 = boto3.client("s3")
secrets_client = boto3.client("secretsmanager")
dynamodb = boto3.resource("dynamodb")
MAX_CHARS_PER_CHUNK = 60_000 * 4


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


def generate_style_guide(client: OpenAI, css: str, theme_prompt: str) -> str:
    sample = css[:40_000]
    print("Generating style guide from CSS sample...")
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a CSS design-systems expert. "
                    "Analyse the CSS provided and return a concise JSON style guide. "
                    "Return ONLY valid JSON — no markdown fences, no extra text."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"{theme_prompt}\n\n"
                    "Analyse this CSS and return a JSON style guide with these keys:\n"
                    "primary_color, secondary_color, accent_color, background_color, text_color, "
                    "font_family_body, font_family_heading, base_font_size, line_height, "
                    "border_radius, spacing_unit, box_shadow, button_style, link_style, "
                    "any_other_key_design_tokens.\n\n"
                    f"CSS sample:\n{sample}"
                ),
            },
        ],
    )
    style_guide = response.choices[0].message.content
    print(f"Style guide generated: {style_guide}")
    return style_guide


def regenerate_css_chunk(
    client: OpenAI,
    chunk: str,
    theme_prompt: str,
    style_guide: str,
    chunk_index: int,
    total_chunks: int,
) -> str:
    system_msg = (
        "You are a CSS and web design expert. "
        "You will receive a portion of a larger CSS file. "
        "Regenerate ONLY the CSS rules provided — do not add new rules or remove existing ones. "
        "Return valid CSS only, with no markdown fences, no explanations, and no extra text."
    )
    user_msg = (
        f"{theme_prompt}\n\n"
        f"You MUST follow this style guide exactly so all chunks look consistent:\n{style_guide}\n\n"
        f"This is chunk {chunk_index + 1} of {total_chunks} from the full stylesheet. "
        f"Regenerate these CSS rules:\n\n{chunk}"
    )
    print(f"Processing chunk {chunk_index + 1}/{total_chunks} ({len(chunk)} chars)...")
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
    )
    return response.choices[0].message.content


def lambda_handler(event, context):
    try:
        secret = json.loads(
            secrets_client.get_secret_value(SecretId=os.environ["SECRET_NAME"])["SecretString"]
        )
        client = OpenAI(api_key=secret["OpenAIAPIKey"])

        for record in event["Records"]:
            body = json.loads(record["body"])
            website_id = body["RegeneratedWebsiteId"]
            print(f"Received regeneration request for website ID: {website_id}")
            website_url = body["RegeneratedWebsiteUrl"]
            print(f"Website URL: {website_url}")
            regeneration_theme = body.get("RegenerationTheme")
            print(f"Regeneration theme: {regeneration_theme}")

            response = s3.get_object(
                Bucket=os.environ["BUCKET_NAME"],
                Key=f"{website_id}/original-styles.css",
            )
            content = response["Body"].read().decode("utf-8")
            print(f"CSS file size: {len(content)} characters")

            if regeneration_theme is None:
                theme_prompt = (
                    "No specific theme provided. Regenerate the CSS using modern practices "
                    "while maintaining the original feel and intent."
                )
            else:
                theme_prompt = (
                    f"Regenerate the CSS using the theme: {regeneration_theme}. "
                    "Ensure the CSS follows modern practices."
                )

            # Split into chunks if the file is large
            chunks = split_css_into_chunks(content)
            print(f"Split CSS into {len(chunks)} chunk(s) for processing")

            # generate a style guide once so all chunks stay visually consistent
            style_guide = generate_style_guide(client, content, theme_prompt)

            # process all chunks in parallel (I/O-bound — threads wait on OpenAI, not CPU)
            results = {}
            with ThreadPoolExecutor(max_workers=len(chunks)) as executor:
                futures = {
                    executor.submit(regenerate_css_chunk, client, chunk, theme_prompt, style_guide, i, len(chunks)): i
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
            )
            print(f"Regenerated CSS saved to S3 for website ID {website_id}")

            table = dynamodb.Table(os.environ["DYNAMODB_TABLE_NAME"])
            table.update_item(
                Key={
                    "RegeneratedWebsiteId": website_id,
                    "RegeneratedWebsiteUrl": website_url,
                },
                UpdateExpression="SET RegenerationStatus = :status",
                ExpressionAttributeValues={":status": "completed"},
            )
            print(f"DynamoDB status updated to completed for website ID {website_id}")

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
        print(f"Error processing regeneration request: {e}")
        return {
            "statusCode": 500,
            "body": json.dumps(
                {"message": "An error occurred during regeneration", "error": str(e)}
            ),
        }
