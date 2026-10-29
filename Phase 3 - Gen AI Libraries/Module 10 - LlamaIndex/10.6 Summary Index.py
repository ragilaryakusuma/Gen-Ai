# 10.6 Summary Index (Halaman 120-122)
from llama_index.core import SummaryIndex, Document, Settings
from llama_index.llms.openrouter import OpenRouter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import os
from dotenv import load_dotenv

load_dotenv()

Settings.llm = OpenRouter(
    model="recursive/flint-v1.0:free",
    api_key=os.environ["OPENROUTER_API_KEY"],
    temperature=0.1,
)
Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)

# Simulate a long document split into multiple sections
sections = [
    Document(text="Introduction: This report analyses the performance of five LLMs on coding, reasoning, and language tasks."),
    Document(text="Section 1 - Coding: Claude Sonnet 4.5 scored highest on HumanEval with 94.2%. GPT-4o followed at 90.1%."),
    Document(text="Section 2 - Reasoning: GPT-4o led on GSM8K with 97.1%. Claude Sonnet 4.5 achieved 96.8%."),
    Document(text="Section 3 - Language: Both models performed similarly on MMLU, within 1% of each other."),
    Document(text="Conclusion: For coding tasks, prefer Claude Sonnet 4.5. For balanced performance, GPT-4o is the safer choice."),
]

index = SummaryIndex.from_documents(sections)
query_engine = index.as_query_engine()

if __name__ == "__main__":
    # SummaryIndex reads ALL nodes, not just top-k - good for document-wide questions
    response = query_engine.query("Which model should I use for a coding project?")
    print(response)
