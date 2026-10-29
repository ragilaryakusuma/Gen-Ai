# 11.6 Module 11 Exercises (Halaman 133)
import time
import numpy as np
import chromadb


# 1. VectorDBBenchmark
class VectorDBBenchmark:
    def __init__(self, collection_name: str = "benchmark_col", dim: int = 64):
        self.client = chromadb.Client()
        self.collection = self.client.create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        self.dim = dim

    def insert_mock_documents(self, count: int = 1000):
        # Generate random normalized embeddings
        embeddings = np.random.randn(count, self.dim)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        ids = [f"id_{i}" for i in range(count)]
        documents = [f"Mock document content number {i}" for i in range(count)]
        metadatas = [{"idx": i, "category": f"cat_{i % 5}"} for i in range(count)]

        start_time = time.perf_counter()
        self.collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings.tolist(),
            metadatas=metadatas,
        )
        insert_duration = time.perf_counter() - start_time
        return insert_duration

    def benchmark_queries(self, num_queries: int = 100, top_k: int = 5):
        query_vectors = np.random.randn(num_queries, self.dim)
        query_vectors = query_vectors / np.linalg.norm(query_vectors, axis=1, keepdims=True)
        
        latencies = []
        start_total = time.perf_counter()
        for q_vec in query_vectors:
            t0 = time.perf_counter()
            self.collection.query(
                query_embeddings=[q_vec.tolist()],
                n_results=top_k,
            )
            latencies.append((time.perf_counter() - t0) * 1000) # ms

        total_time = time.perf_counter() - start_total
        avg_latency = np.mean(latencies)
        p95_latency = np.percentile(latencies, 95)
        throughput = num_queries / total_time

        return {
            "avg_latency_ms": round(float(avg_latency), 3),
            "p95_latency_ms": round(float(p95_latency), 3),
            "throughput_qps": round(float(throughput), 1),
            "total_queries": num_queries,
        }


# 2. MultiCollectionSearch for ChromaDB
class MultiCollectionSearch:
    def __init__(self, client):
        self.client = client

    def search(self, collection_names: list[str], query_embedding: list[float], top_k: int = 5) -> list[dict]:
        combined_results = []
        for name in collection_names:
            try:
                col = self.client.get_collection(name)
                res = col.query(
                    query_embeddings=[query_embedding],
                    n_results=top_k,
                    include=["documents", "distances", "metadatas"],
                )
                if res["documents"] and res["documents"][0]:
                    for doc, dist, meta in zip(res["documents"][0], res["distances"][0], res["metadatas"][0]):
                        combined_results.append({
                            "collection": name,
                            "document": doc,
                            "distance": round(float(dist), 4),
                            "metadata": meta,
                        })
            except Exception:
                continue

        # Sort combined results by distance ascending
        combined_results.sort(key=lambda x: x["distance"])
        return combined_results[:top_k]


if __name__ == "__main__":
    # --- Exercise 1 ---
    print("[Exercise 1] VectorDBBenchmark")
    bench = VectorDBBenchmark()
    insert_sec = bench.insert_mock_documents(count=1000)
    print(f"Inserted 1,000 documents in {insert_sec:.4f} seconds.")

    metrics = bench.benchmark_queries(num_queries=100)
    print(f"Queries benchmarked: {metrics['total_queries']}")
    print(f"Average latency   : {metrics['avg_latency_ms']} ms")
    print(f"95th percentile   : {metrics['p95_latency_ms']} ms")
    print(f"Throughput        : {metrics['throughput_qps']} queries/sec")

    # --- Exercise 2 ---
    print("\n[Exercise 2] MultiCollectionSearch")
    client = chromadb.Client()
    col_research = client.create_collection("research_papers")
    col_blogs = client.create_collection("tech_blogs")

    # Seed mock data
    col_research.add(
        ids=["r1", "r2"],
        documents=["Attention is All You Need (Vaswani et al.)", "BERT: Pre-training of Deep Bidirectional Transformers"],
        embeddings=[[0.1, 0.9, 0.2], [0.2, 0.8, 0.3]],
        metadatas=[{"type": "paper"}, {"type": "paper"}],
    )
    col_blogs.add(
        ids=["b1", "b2"],
        documents=["Understanding Attention Mechanism in Simple Terms", "Getting Started with Vector Databases in Python"],
        embeddings=[[0.15, 0.85, 0.25], [0.8, 0.1, 0.4]],
        metadatas=[{"type": "blog"}, {"type": "blog"}],
    )

    multi_search = MultiCollectionSearch(client)
    q_vec = [0.12, 0.88, 0.22]
    top_results = multi_search.search(["research_papers", "tech_blogs"], query_embedding=q_vec, top_k=3)

    print("Unified ranked results across multiple collections:")
    for rank, item in enumerate(top_results, 1):
        print(f"{rank}. [{item['collection']}] (dist: {item['distance']}) {item['document']}")
