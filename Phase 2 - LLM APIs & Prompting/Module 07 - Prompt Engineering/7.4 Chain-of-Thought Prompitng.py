import os
import sys
import time
from openai import OpenAI, RateLimitError
from dotenv import load_dotenv

if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

load_dotenv()

client = OpenAI(
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1"
)
MODEL = os.getenv("OPENROUTER_MODEL", "openrouter/free")

# Without CoT - model jumps to answer, more likely to be wrong
DIRECT_PROMPT = "If a model costs $3.00 per million input tokens and $15.00 per million output tokens, and a request uses 2,400 input tokens and 800 output tokens, what is the total cost in USD?"

# With CoT - model reasons through each step
COT_PROMPT = """If a model costs $3.00 per million input tokens and $15.00 per million output tokens,
and a request uses 2,400 input tokens and 800 output tokens,
what is the total cost in USD?
Think through this step by step before giving the final answer."""

# Zero-shot CoT: just adding "think step by step"
ZERO_SHOT_COT = """Solve this problem. Think step by step, showing each calculation.
Finally, state: ANSWER: $X.XXXXXX
Problem: A pipeline makes 50 API calls per hour. Each call uses an average of 1,200 input tokens
and 400 output tokens. The model costs $3.00/M input and $15.00/M output.
What is the daily cost?"""

for label, prompt in [
    ("Direct", DIRECT_PROMPT),
    ("CoT", COT_PROMPT),
    ("Zero-shot CoT", ZERO_SHOT_COT)
]:
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                max_tokens=768,
                messages=[{"role": "user", "content": prompt}],
            )
            msg = resp.choices[0].message
            content = msg.content or getattr(msg, "reasoning", "") or ""
            print(f"=== {label} ===")
            print(content.strip())
            print()
            break
        except RateLimitError:
            if attempt == 2:
                raise
            wait = 2 ** attempt
            print(f"Rate limited on {label}. Retrying in {wait}s...")
            time.sleep(wait)