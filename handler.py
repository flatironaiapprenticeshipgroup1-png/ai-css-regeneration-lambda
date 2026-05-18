import json
import os

import boto3
from openai import OpenAI

# Initialize AWS S3 client and OpenAI client
s3 = boto3.client("s3")
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


# Lambda handler function to process regeneration requests
def lambda_handler(event, context):
    try:
        record = event["Records"][0]
        body = json.loads(record["body"])
        website_id = body["RegeneratedWebsiteId"]
        # Log the received regeneration request details
        print(f"Received regeneration request for website ID: {website_id}")
        website_url = body["RegeneratedWebsiteUrl"]
        print(f"Website URL: {website_url}")
        regeneration_theme = body.get("RegenerationTheme")
        print(f"Regeneration theme: {regeneration_theme}")
        # Fetch the original CSS from S3 using the website ID
        response = s3.get_object(
            Bucket="website-regeneration-s3-bucket",
            Key=f"{website_id}/original-styles.css",
        )
        # Read and decode the original CSS content
        content = response["Body"].read().decode("utf-8")
        print(f"Original content for website ID {website_id}:\n{content}")
        if regeneration_theme is None:
            theme_prompt = "No specific theme provided. Please regenerate the CSS and HTML using modern practices while maintaining the original feel and intent."

        else:
            theme_prompt = f"Please regenerate the CSS and HTML using the theme: {regeneration_theme}. Ensure the CSS follows modern practices."
        print(f"Theme prompt for regeneration: {theme_prompt}")
        # Call OpenAI API to regenerate the CSS based on the original content and theme prompt
        call = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a CSS and web design expert."},
                {"role": "user", "content": f"{theme_prompt}\n\n{content}"},
            ],
        )
        # Extract the regenerated CSS from the OpenAI response
        regenerated_css = call.choices[0].message.content
        print(f"Regenerated CSS for website ID {website_id}:\n{regenerated_css}")
        # Save the regenerated CSS back to S3 under a new key
        s3.put_object(
            Bucket="website-regeneration-s3-bucket",
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
