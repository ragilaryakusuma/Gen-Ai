import anthropic, os
from dotenv import load_dotenv

load_dotenv()

client = anthropic.Anthropic(
    api_key=os.environ["ANTHROPIC_API_KEY"]
)

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
    resp = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}],
    )

    print(f"==={label} ===")
    print(resp.content[0].text[:300])
    print()