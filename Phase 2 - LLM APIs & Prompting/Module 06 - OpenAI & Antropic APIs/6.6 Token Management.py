import anthropic, os
from dotenv import load_dotenv

load_dotenv()

client = anthropic.Anthropic(
    api_key=os.environ["ANTHROPIC_API_KEY"]
)

# Count tokens before sending (no API call made, no charge)
response = client.messages.count_tokens(
    model="claude-sonnet-4-5",
    system="You are a concise assistant.",
    messages=[
        {
            "role": "user",
            "content": "Explain the transformer architecture."
        }
    ]
)

print(f"Estimated input tokens: {response.input_tokens}")

# Cost estimator
PRICING = {
    "claude-sonnet-4-5": {"input": 3.00, "output": 15.00},
    # per 1M tokens
    "claude-opus-4-5": {"input": 15.00, "output": 75.00},
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
}


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Return estimated cost in USD."""
    if model not in PRICING:
        raise ValueError(f"Unknown model: {model}")

    p = PRICING[model]

    return (
        input_tokens * p["input"] +
        output_tokens * p["output"]
    ) / 1_000_000


cost = estimate_cost(
    "claude-sonnet-4-5",
    input_tokens=500,
    output_tokens=300
)

print(f"Estimated cost: ${cost:.6f}")

# Context window limits (always check before sending long documents)
CONTEXT_LIMITS = {
    "claude-sonnet-4-5": 200_000,
    "claude-opus-4-5": 200_000,
    "gpt-4o": 128_000,
    "gpt-4o-mini": 128_000,
    "gemini-1.5-pro": 1_000_000,
}


def fits_in_context(
    model: str,
    token_count: int,
    reserve_for_output: int = 2048
) -> bool:
    limit = CONTEXT_LIMITS.get(model, 128_000)
    return token_count + reserve_for_output <= limit