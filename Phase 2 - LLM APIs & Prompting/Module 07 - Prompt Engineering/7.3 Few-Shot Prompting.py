import os
import time
from openai import OpenAI, RateLimitError
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1"
)
MODEL = os.getenv("OPENROUTER_MODEL", "openrouter/free")

FEW_SHOT_SYSTEM = """You are a data extractor. Given a raw AI
benchmark result string,
extract: model name, task, and score as a JSON object.
Examples:
Input: "GPT-4o scored 87.3% on the MMLU science subset"
Output: {"model": "gpt-4o", "task": "MMLU science", "score":
87.3}
Input: "Claude Sonnet 4.5 achieved 92.1 on HumanEval"
Output: {"model": "claude-sonnet-4-5", "task": "HumanEval",
"score": 92.1}
Input: "Gemini 1.5 Pro: 78.9% accuracy on GSM8K math"
Output: {"model": "gemini-1.5-pro", "task": "GSM8K math", "score":
78.9}
Return ONLY the JSON object. No explanation."""

test_inputs = [
    "GPT-4o-mini reached 82.0% on MMLU",
    "Llama 3.1 70B: 86.4 on TruthfulQA",
    "Claude Opus 4.5 scored 96.7% on SWE-bench Verified",
]

def create_completion(text: str, max_retries: int = 3):
    """Retry transient OpenRouter provider rate limits with backoff."""
    messages = [
        {"role": "system", "content": FEW_SHOT_SYSTEM},
        {"role": "user", "content": text},
    ]

    for attempt in range(max_retries):
        try:
            return client.chat.completions.create(
                model=MODEL,
                max_tokens=128,
                messages=messages,
            )
        except RateLimitError:
            if attempt == max_retries - 1:
                raise

            wait_seconds = 2 ** attempt
            print(
                f"Rate limited for model {MODEL}. "
                f"Retrying in {wait_seconds}s..."
            )
            time.sleep(wait_seconds)


for text in test_inputs:
    resp = create_completion(text)

    print(f"Input:{text}")
    print(f"Output:{resp.choices[0].message.content}\n")