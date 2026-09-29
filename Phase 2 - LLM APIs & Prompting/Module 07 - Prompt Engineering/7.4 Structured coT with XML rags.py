import anthropic, os, re
from dotenv import load_dotenv

load_dotenv()

client = anthropic.Anthropic(
    api_key=os.environ["ANTHROPIC_API_KEY"]
)

SYSTEM = """Solve problems using this exact format:
<thinking>
Step-by-step reasoning here.
</thinking>
<answer>
The final answer only, no reasoning.
</answer>"""

resp = client.messages.create(
    model="claude-sonnet-4-5",
    max_tokens=512,
    system=SYSTEM,
    messages=[{
        "role": "user",
        "content": "A RAG pipeline retrieves 5 documents, each 400 tokens. The query is 50 tokens. The model has a 4096 token limit for context. How many tokens remain for the response?"
    }],
)

text = resp.content[0].text

# Extract sections
thinking = re.search(
    r"<thinking>(.*?)</thinking>",
    text,
    re.DOTALL
)

answer = re.search(
    r"<answer>(.*?)</answer>",
    text,
    re.DOTALL
)

print(
    "Reasoning:",
    thinking.group(1).strip() if thinking else "not found"
)

print(
    "Answer: ",
    answer.group(1).strip() if answer else "not found"
)