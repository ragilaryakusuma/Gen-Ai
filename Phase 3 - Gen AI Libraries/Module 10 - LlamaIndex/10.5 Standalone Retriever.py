# 10.5 Retriever - Used Independently (Halaman 119-120)
from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.llms.openrouter import OpenRouter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import os
from dotenv import load_dotenv

load_dotenv()

# Gunakan OpenRouter dengan model recursive/flint-v1.0:free
Settings.llm = OpenRouter(
    model="recursive/flint-v1.0:free",
    api_key=os.environ["OPENROUTER_API_KEY"],
    temperature=0.1,
)

# Gunakan HuggingFace Embedding lokal (gratis & offline, tanpa bergantung kuota OpenAI)
Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)

documents = [
    Document(text="ChromaDB is an open-source embedding database with a simple Python API.", metadata={"tag": "vector-db"}),
    Document(text="Pinecone is a managed vector database designed for production workloads.", metadata={"tag": "vector-db"}),
    Document(text="FAISS is a library by Meta for efficient similarity search on dense vectors.", metadata={"tag": "vector-search"}),
    Document(text="pgvector adds vector similarity search to PostgreSQL.", metadata={"tag": "vector-db"}),
    Document(text="Weaviate is an open-source vector database with a GraphQL API.", metadata={"tag": "vector-db"}),
]

index = VectorStoreIndex.from_documents(documents)
retriever = VectorIndexRetriever(index=index, similarity_top_k=3)

if __name__ == "__main__":
    # Retrieve without synthesising an answer
    nodes = retriever.retrieve("open source vector databases")
    print("Retrieved nodes:")
    for node in nodes:
        print(f" [{node.score:.4f}] {node.text}")
        print(f"  tag: {node.metadata.get('tag')}")
