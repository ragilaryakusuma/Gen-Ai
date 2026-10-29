# 11.3 Pinecone (Halaman 128-130)
# pip install pinecone
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

# Gunakan model embedding lokal HuggingFace
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

def embed(texts: list[str]) -> list[list[float]]:
    return embed_model.encode(texts).tolist()

if __name__ == "__main__":
    if "PINECONE_API_KEY" in os.environ and os.environ["PINECONE_API_KEY"].strip():
        from pinecone import Pinecone, ServerlessSpec
        pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
        INDEX_NAME = "ai-knowledge-base"

        # Create index if it doesn't exist
        if INDEX_NAME not in pc.list_indexes().names():
            pc.create_index(
                name=INDEX_NAME,
                dimension=384, # all-MiniLM-L6-v2 dimension
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
        index = pc.Index(INDEX_NAME)

        documents = [
            {"id": "d1", "text": "RAG connects LLMs to external knowledge bases."},
            {"id": "d2", "text": "Fine-tuning updates model weights on task-specific data."},
            {"id": "d3", "text": "ChromaDB is an open-source local vector database."},
        ]

        vectors = [
            {
                "id": doc["id"],
                "values": embed([doc["text"]])[0],
                "metadata": {"text": doc["text"], "source": "guide"},
            }
            for doc in documents
        ]
        index.upsert(vectors=vectors)
        print(index.describe_index_stats())

        # Query
        q_vector = embed(["how to add external knowledge to an LLM?"])[0]
        results = index.query(
            vector=q_vector,
            top_k=3,
            include_metadata=True,
        )
        for match in results["matches"]:
            print(f"[{match['score']:.4f}] {match['metadata']['text']}")
    else:
        print("PINECONE_API_KEY not configured. Set it in .env to run live Pinecone calls.")
