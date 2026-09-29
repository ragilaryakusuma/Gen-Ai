import anthropic
import os
import asyncio
import time
import random
import pandas as pd

from dotenv import load_dotenv

load_dotenv()


# ============================================================
# Exercise 1
# Retry on Rate Limit with Exponential Backoff
# ============================================================

def retry_on_rate_limit(
    client,
    messages,
    max_retries=5,
    system="",
    max_tokens=512
):
    """
    Retry an Anthropic API call when RateLimitError occurs.
    Uses exponential backoff.
    """

    for attempt in range(max_retries):
        try:
            response = client.messages.create(
                model="claude-sonnet-4-5",
                max_tokens=max_tokens,
                system=system,
                messages=messages,
            )

            return response

        except anthropic.RateLimitError:
            if attempt == max_retries - 1:
                raise

            wait_time = 2 ** attempt
            print(
                f"Rate limit reached. "
                f"Retrying in {wait_time} seconds..."
            )

            time.sleep(wait_time)

    raise RuntimeError("Maximum retries exceeded.")


# Mock client for testing
class MockRateLimitClient:

    def __init__(self, failures_before_success=3):
        self.failures_before_success = failures_before_success
        self.attempts = 0

    class Messages:

        def __init__(self, parent):
            self.parent = parent

        def create(self, **kwargs):
            self.parent.attempts += 1

            if (
                self.parent.attempts
                <= self.parent.failures_before_success
            ):
                raise anthropic.RateLimitError(
                    message="Mock rate limit",
                    response=None
                )

            return MockResponse()

    @property
    def messages(self):
        return self.Messages(self)


class MockResponse:

    class Content:

        text = "Mock response after retry."

    content = [Content()]

    class Usage:
        input_tokens = 10
        output_tokens = 20

    usage = Usage()


# Test Exercise 1
# Uncomment to test the retry mechanism.

# mock_client = MockRateLimitClient(
#     failures_before_success=3
# )
#
# response = retry_on_rate_limit(
#     mock_client,
#     messages=[
#         {
#             "role": "user",
#             "content": "Hello"
#         }
#     ],
#     max_retries=5
# )
#
# print("Exercise 1 result:")
# print(response.content[0].text)


# ============================================================
# Exercise 2
# Token Budget Manager
# ============================================================

class BudgetExceeded(Exception):
    """Raised when the token budget has been exceeded."""
    pass


class TokenBudgetManager:

    def __init__(self, max_tokens: int):
        self.max_tokens = max_tokens
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    @property
    def total_tokens(self):
        return (
            self.total_input_tokens
            + self.total_output_tokens
        )

    def add_usage(
        self,
        input_tokens: int,
        output_tokens: int
    ):
        new_total = (
            self.total_tokens
            + input_tokens
            + output_tokens
        )

        if new_total > self.max_tokens:
            raise BudgetExceeded(
                f"Token budget exceeded. "
                f"Limit: {self.max_tokens}, "
                f"requested total: {new_total}"
            )

        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens

    def remaining_tokens(self):
        return self.max_tokens - self.total_tokens

    def reset(self):
        self.total_input_tokens = 0
        self.total_output_tokens = 0


# Test Exercise 2

budget = TokenBudgetManager(max_tokens=1000)

try:
    budget.add_usage(
        input_tokens=200,
        output_tokens=150
    )

    budget.add_usage(
        input_tokens=300,
        output_tokens=200
    )

    print("\nExercise 2 result:")
    print(
        f"Total tokens: {budget.total_tokens}"
    )
    print(
        f"Remaining tokens: {budget.remaining_tokens()}"
    )

    budget.add_usage(
        input_tokens=200,
        output_tokens=100
    )

except BudgetExceeded as error:
    print(f"Budget error: {error}")


# ============================================================
# Exercise 3
# Compare Multiple Models Concurrently
# ============================================================

async def call_model(
    client,
    model: str,
    prompt: str
):
    """
    Call one model and measure latency.
    """

    start_time = time.perf_counter()

    try:
        response = await asyncio.to_thread(
            client.messages.create,
            model=model,
            max_tokens=512,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        latency_ms = (
            time.perf_counter() - start_time
        ) * 1000

        return {
            "model": model,
            "response_text": response.content[0].text,
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "latency_ms": round(latency_ms, 2)
        }

    except Exception as error:

        latency_ms = (
            time.perf_counter() - start_time
        ) * 1000

        return {
            "model": model,
            "response_text": f"ERROR: {error}",
            "input_tokens": 0,
            "output_tokens": 0,
            "latency_ms": round(latency_ms, 2)
        }


async def compare_models_async(
    client,
    prompt: str,
    models: list[str]
) -> pd.DataFrame:

    tasks = [
        call_model(
            client,
            model,
            prompt
        )
        for model in models
    ]

    results = await asyncio.gather(*tasks)

    return pd.DataFrame(
        results,
        columns=[
            "model",
            "response_text",
            "input_tokens",
            "output_tokens",
            "latency_ms"
        ]
    )


def compare_models(
    prompt: str,
    models: list[str]
) -> pd.DataFrame:

    client = anthropic.Anthropic(
        api_key=os.environ["ANTHROPIC_API_KEY"]
    )

    return asyncio.run(
        compare_models_async(
            client,
            prompt,
            models
        )
    )


# Example Exercise 3
#
# models = [
#     "claude-sonnet-4-5",
#     "claude-opus-4-5"
# ]
#
# results = compare_models(
#     "Explain what a vector database is.",
#     models
# )
#
# print("\nExercise 3 result:")
# print(results.to_string(index=False))


# ============================================================
# Exercise 4
# Stream Anthropic Response Directly to a File
# ============================================================

def stream_to_file(
    prompt: str,
    output_path: str
):

    client = anthropic.Anthropic(
        api_key=os.environ["ANTHROPIC_API_KEY"]
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:

        with client.messages.stream(
            model="claude-sonnet-4-5",
            max_tokens=512,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        ) as stream:

            for text in stream.text_stream:

                file.write(text)
                file.flush()

                print(
                    text,
                    end="",
                    flush=True
                )

    print(
        f"\n\nResponse saved to: {output_path}"
    )


# Example Exercise 4
#
# stream_to_file(
#     "Explain retrieval-augmented generation.",
#     "exercise_4_output.txt"
# )