# 10.2 Setup and Settings (Halaman 116)
from llama_index.core import Settings
from llama_index.llms.openrouter import OpenRouter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import os
from dotenv import load_dotenv

load_dotenv()

# Configure global settings - applied to all indexes and query engines
Settings.llm = OpenRouter(
    model="recursive/flint-v1.0:free",
    api_key=os.environ["OPENROUTER_API_KEY"],
    temperature=0.1,
    max_tokens=1024,
)
Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)
Settings.chunk_size = 512       # tokens per chunk
Settings.chunk_overlap = 64     # token overlap between chunks

if __name__ == "__main__":
    print("LlamaIndex Global Settings configured successfully.")
