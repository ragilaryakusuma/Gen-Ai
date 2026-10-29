# 10.8 Custom Query Pipeline (Halaman 123-124)
from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.response_synthesizers import get_response_synthesizer, ResponseMode
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

documents = [
    Document(text="HNSW (Hierarchical Navigable Small World) is the graph algorithm powering most modern vector search."),
    Document(text="IVF (Inverted File Index) partitions vectors into clusters for fast approximate search."),
    Document(text="PQ (Product Quantisation) compresses vectors to reduce memory usage at the cost of recall."),
    Document(text="Exact nearest-neighbour search guarantees correctness but is O(n) - unusable at scale."),
]

index = VectorStoreIndex.from_documents(documents)
retriever = VectorIndexRetriever(index=index, similarity_top_k=2)

if __name__ == "__main__":
    # Different response modes
    for mode in [ResponseMode.COMPACT, ResponseMode.TREE_SUMMARIZE]:
        synthesiser = get_response_synthesizer(response_mode=mode)
        engine = RetrieverQueryEngine(retriever=retriever, response_synthesizer=synthesiser)
        resp = engine.query("What algorithms are used in vector search?")
        print(f"[{mode}] {str(resp)[:120]}\n")
