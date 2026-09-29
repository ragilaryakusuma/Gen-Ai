import anthropic
import os
from dotenv import load_dotenv

load_dotenv(dotenv_path=r"C:\Users\MSI-PC\Downloads\Generative AI - Copy\.env")

client = anthropic.Anthropic(
    api_key=os.environ["ANTHROPIC_API_KEY"]
)

message = client.messages.create(
    model="claude-sonnet-4-5",
    max_tokens=512,
    system="You are a concise technical writer. Answer in plain English, no jargon.",
    messages=[
        {
            "role": "user",
            "content": "Explain what a vector database does."
        }
    ]
)

print(message.content[0].text)

# Usage stats
print(f"Input tokens: {message.usage.input_tokens}")
print(f"Output tokens: {message.usage.output_tokens}")