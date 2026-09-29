import anthropic, os, base64
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

client = anthropic.Anthropic(
    api_key=os.environ["ANTHROPIC_API_KEY"]
)

# Option A: URL (fastest)
def describe_image_url(url: str) -> str:
    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=512,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "url",
                        "url": url
                    }
                },
                {
                    "type": "text",
                    "text": "Describe what you see in this image."
                }
            ]
        }]
    )

    return response.content[0].text


# Option B: base64 (for local files)
def describe_image_file(path: str) -> str:
    data = Path(path).read_bytes()
    b64 = base64.standard_b64encode(data).decode()
    ext = Path(path).suffix.lstrip(".").lower()
    media_type = f"image/{ext}"  # image/png, image/jpeg, image/webp, image/gif

    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=512,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": b64
                    }
                },
                {
                    "type": "text",
                    "text": "What is in this image?"
                }
            ]
        }]
    )

    return response.content[0].text


# Usage:
# text = describe_image_url("https://upload.wikimedia.org/wikipedia/commons/thumb/1/1e/Sunrise_over_the_sea.jpg/1280px-Sunrise_over_the_sea.jpg")
# print(text)