# 10.7 Persistent Storage (Halaman 122-123)
from llama_index.core import VectorStoreIndex, Document, StorageContext, load_index_from_storage, Settings
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

PERSIST_DIR = os.path.join(os.path.dirname(__file__), "storage")

documents = [
    Document(text="LlamaIndex persists indexes to disk as JSON files."),
    Document(text="Loading a persisted index avoids re-embedding on every run."),
]

if __name__ == "__main__":
    # Build and save index
    index = VectorStoreIndex.from_documents(documents)
    index.storage_context.persist(persist_dir=PERSIST_DIR)
    print(f"Index saved to {PERSIST_DIR}")

    # Load index back - no re-embedding needed
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    loaded_index = load_index_from_storage(storage_context)
    query_engine = loaded_index.as_query_engine()
    response = query_engine.query("How does LlamaIndex handle persistence?")
    print("After reload:", response)
