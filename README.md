# ai-css-regeneration-lambda

Second and final stage of the AI Website Regenerator pipeline. Regenerates a
website's CSS to match a requested theme with GPT-4o, after
[`webpage-crawler-lambda`](../webpage-crawler-lambda) has captured the
original site and regenerated its HTML.

## What it does

Triggered by an SQS message containing `{RegeneratedWebsiteId,
RegeneratedWebsiteUrl, RegenerationTheme}`, `lambda_function.py` runs:

1. **Idempotency check** — reads the job's DynamoDB status and skips the
   message if it's already `ai_lambda_processing` or `completed` (SQS
   delivers at-least-once, so duplicate deliveries are expected).
2. **Load original CSS** — reads `{website_id}/original-styles.css` from S3
   (written earlier by the crawler lambda).
3. **Chunk** — splits the stylesheet into top-level rule blocks (tracking
   brace depth so `@media`/`@layer` nesting and quoted strings are handled
   correctly), then bin-packs blocks into ~30k-character chunks. A single
   oversized block (e.g. one huge `@layer` from a real-world stylesheet) is
   recursively split at its own nested rule or declaration boundaries
   rather than sliced at an arbitrary character offset.
4. **Regenerate** — sends each chunk to GPT-4o in parallel
   (`ThreadPoolExecutor`), with a prompt describing the requested theme and
   listing every selector in that chunk that must survive in the output
   (typography, color palette anchored to the theme, borders/shapes,
   spacing, and decorative effects are all in scope; image/svg sizing
   attributes are explicitly left untouched).
5. **Verify & restore** — checks the regenerated chunk for any selector the
   model dropped. If a rule block is missing, its original layout
   declarations (padding, borders, sizing, etc.) are restored **with all
   color-bearing declarations stripped**, so the element doesn't fall back
   to unstyled but also doesn't smuggle the pre-theme colors back in.
6. **Save** — reassembles the chunks and writes
   `{website_id}/Regenerated-Styles.css` to S3, then marks the DynamoDB job
   `completed`.

Progress is published at each step to Ably (channel
`regeneration:{website_id}`, event `regeneration-status`) and mirrored to
DynamoDB. The sequence counter is continued from wherever the crawler
lambda left off (`get_current_sequence`), so the frontend sees one
continuous stream of events across both lambdas.

## Files

| File | Purpose |
|---|---|
| `lambda_function.py` | Lambda entry point: CSS chunking, GPT-4o regeneration, missing-block restoration |
| `status_publisher.py` | Publishes progress to Ably and persists status to DynamoDB |
| `test_handler.py` | Unit tests |

## Environment variables

- `BUCKET_NAME`
- `DYNAMODB_TABLE_NAME`
- `SECRET_NAME` — Secrets Manager secret containing `OpenAIAPIKey`
- `ABLY_SECRET_NAME` — Secrets Manager secret containing `AblyApiKey`

## Running locally / deployment

Dependencies are managed with Pipenv (`Pipfile`, `Pipfile.lock`), Python 3.11.

```bash
pipenv install
```

Tests:

```bash
pipenv run pytest
```
