import json
import os

import boto3
from openai import OpenAI

s3 = boto3.client("s3")
secrets_client = boto3.client("secretsmanager")


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
            print(f"Original content for website ID {website_id}:\n{content}")
            if regeneration_theme is None:
                theme_prompt = "No specific theme provided. Please regenerate the CSS and HTML using modern practices while maintaining the original feel and intent."
            else:
                theme_prompt = f"Please regenerate the CSS and HTML using the theme: {regeneration_theme}. Ensure the CSS follows modern practices."
            print(f"Theme prompt for regeneration: {theme_prompt}")
            call = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a CSS and web design expert."},
                    {"role": "user", "content": f"{theme_prompt}\n\n{content}"},
                ],
            )
            regenerated_css = call.choices[0].message.content
            print(f"Regenerated CSS for website ID {website_id}:\n{regenerated_css}")
            s3.put_object(
                Bucket=os.environ["BUCKET_NAME"],
                Key=f"{website_id}/Regenerated-Styles.css",
                Body=regenerated_css.encode("utf-8"),
                ContentType="text/css",
            )
            print(f"Regenerated CSS saved to S3 for website ID {website_id}")
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
