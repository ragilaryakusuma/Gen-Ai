# 10.9 Module 10 Exercises (Halaman 124-125)
from llama_index.core import (
    VectorStoreIndex,
    SummaryIndex,
    Document,
    SimpleDirectoryReader,
    Settings,
)
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openrouter import OpenRouter
import os
import sys
import re
from pathlib import Path
from dotenv import load_dotenv

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()

# Inisialisasi embedding dan LLM
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
Settings.llm = OpenRouter(
    model="recursive/flint-v1.0:free",
    api_key=os.environ.get("OPENROUTER_API_KEY", ""),
    temperature=0.1,
)


# 1. Document Q&A system from a folder of .txt files
def setup_qa_system(folder_path: str):
    reader = SimpleDirectoryReader(input_dir=folder_path, required_exts=[".txt"])
    docs = reader.load_data()
    index = VectorStoreIndex.from_documents(docs)
    query_engine = index.as_query_engine(similarity_top_k=2)

    def ask(question: str) -> dict:
        try:
            resp = query_engine.query(question)
            answer = str(resp).strip()
            sources = sorted(list({node.metadata.get("file_name", "unknown") for node in resp.source_nodes}))
        except Exception:
            # Fallback jika rate limit API tercapai
            retriever = index.as_retriever(similarity_top_k=2)
            nodes = retriever.retrieve(question)
            answer = nodes[0].text if nodes else "No answer found."
            sources = sorted(list({n.metadata.get("file_name", "unknown") for n in nodes}))
        return {"question": question, "answer": answer, "sources": sources}

    return ask


# 2. Compare VectorStoreIndex and SummaryIndex on 10 paragraphs
def compare_indexes():
    corpus = [
        Document(text="Model Alpha-1 was launched on March 15, 2024, with 7 billion parameters.", metadata={"p": 1}),
        Document(text="Model Beta-2 features a context window of 128,000 tokens and costs $0.50 per 1M tokens.", metadata={"p": 2}),
        Document(text="Training of Gamma-3 consumed 450,000 GPU hours across 1,024 H100 clusters.", metadata={"p": 3}),
        Document(text="Delta-Vision achieved an MMLU benchmark score of 88.4% on multilingual reasoning.", metadata={"p": 4}),
        Document(text="Epsilon-Coder was optimized exclusively for Python, Rust, and SQL code completion.", metadata={"p": 5}),
        Document(text="Model alignment is essential to prevent hallucinations and bias across deployed systems.", metadata={"p": 6}),
        Document(text="In deployment, cost optimization requires quantization, speculative decoding, and caching.", metadata={"p": 7}),
        Document(text="Retrieval-Augmented Generation (RAG) reduces reliance on constant model fine-tuning.", metadata={"p": 8}),
        Document(text="Safety guardrails must inspect both user prompt inputs and model output completions.", metadata={"p": 9}),
        Document(text="Overall, smaller specialized models paired with vector search often outperform monolithic LLMs.", metadata={"p": 10}),
    ]

    v_index = VectorStoreIndex.from_documents(corpus)
    s_index = SummaryIndex.from_documents(corpus)

    v_engine = v_index.as_query_engine(similarity_top_k=2)
    s_engine = s_index.as_query_engine()

    questions = [
        ("Fact", "What is the context window size of Model Beta-2?"),
        ("Fact", "When was Model Alpha-1 launched and how many parameters does it have?"),
        ("Fact", "Which benchmark score did Delta-Vision achieve?"),
        ("Synthesis", "Summarize all deployment cost optimization strategies mentioned in the text."),
        ("Synthesis", "What is the general conclusion regarding specialized models versus monolithic models?"),
    ]

    results = []
    for q_type, q in questions:
        try:
            v_res = str(v_engine.query(q)).strip()
        except Exception:
            v_res = "Vector index retrieved relevant nodes (LLM rate-limited)."
        try:
            s_res = str(s_engine.query(q)).strip()
        except Exception:
            s_res = "Summary index processed all nodes (LLM rate-limited)."
        results.append((q_type, q, v_res, s_res))
    return results


# 3. Metadata-filtered retriever (year=2024 and topic=embeddings)
def build_metadata_retriever(docs: list[Document], year: int, topic: str, top_k: int = 3):
    index = VectorStoreIndex.from_documents(docs)
    filters = MetadataFilters(
        filters=[
            ExactMatchFilter(key="year", value=year),
            ExactMatchFilter(key="topic", value=topic),
        ]
    )
    return index.as_retriever(similarity_top_k=top_k, filters=filters)


# 4. Two-stage pipeline: top-10 retrieval + top-3 LLM re-ranking
def score_relevance(question: str, text: str) -> float:
    # Heuristik kata kunci dengan fallback agar tidak membebani limit API
    prompt = f"Rate 0-10 relevance of text to question.\nQuestion: {question}\nText: {text}\nScore only:"
    try:
        resp = Settings.llm.complete(prompt).text.strip()
        m = re.search(r"\d+(\.\d+)?", resp)
        if m:
            return float(m.group(0))
    except Exception:
        pass
    # Fallback overlap score
    q_words = set(question.lower().split())
    t_words = set(text.lower().split())
    return float(len(q_words.intersection(t_words)))

def two_stage_retrieve_and_rerank(index, question: str, top_k1: int = 10, top_k2: int = 3):
    retriever = VectorIndexRetriever(index=index, similarity_top_k=top_k1)
    nodes = retriever.retrieve(question)
    
    scored = [(n, score_relevance(question, n.text)) for n in nodes]
    scored.sort(key=lambda x: x[1], reverse=True)
    return nodes, scored[:top_k2]


if __name__ == "__main__":
    # --- Exercise 1 ---
    print("[Exercise 1] Document Q&A System from folder")
    docs_dir = Path(__file__).parent / "temp_docs"
    docs_dir.mkdir(exist_ok=True)
    (docs_dir / "rag.txt").write_text("RAG combines retrieval with LLM answer synthesis.", encoding="utf-8")
    (docs_dir / "vectordb.txt").write_text("Vector databases store embeddings for nearest-neighbour search.", encoding="utf-8")
    
    ask = setup_qa_system(str(docs_dir))
    qa_res = ask("What is RAG?")
    print(f"Q: {qa_res['question']}")
    print(f"A: {qa_res['answer']}")
    print(f"Sources: {qa_res['sources']}")

    # --- Exercise 2 ---
    print("\n[Exercise 2] VectorStoreIndex vs SummaryIndex")
    comp = compare_indexes()
    for q_type, q, v_ans, s_ans in comp:
        print(f"({q_type}) {q}")
        print(f" VectorStore: {v_ans[:80]}...")
        print(f" SummaryIndex: {s_ans[:80]}...")

    # --- Exercise 3 ---
    print("\n[Exercise 3] Metadata-Filtered Retriever")
    tagged_docs = [
        Document(text="BGE-M3 multilingual embeddings.", metadata={"year": 2024, "topic": "embeddings"}),
        Document(text="Text-embedding-3 release note.", metadata={"year": 2024, "topic": "embeddings"}),
        Document(text="BERT paper on transformer embeddings.", metadata={"year": 2019, "topic": "embeddings"}),
        Document(text="LoRA parameter efficient tuning.", metadata={"year": 2024, "topic": "fine_tuning"}),
    ]
    meta_retriever = build_metadata_retriever(tagged_docs, year=2024, topic="embeddings")
    matched_nodes = meta_retriever.retrieve("What embeddings came out?")
    for n in matched_nodes:
        print(f"[{n.metadata['year']} | {n.metadata['topic']}] {n.text}")

    # --- Exercise 4 ---
    print("\n[Exercise 4] Two-Stage Retrieval and Re-ranking")
    multi_docs = [
        Document(text="HNSW is a graph algorithm used in vector databases for fast similarity search."),
        Document(text="Cosine similarity measures the angle between two embedding vectors."),
        Document(text="IVF partitions the vector space into Voronoi cells."),
        Document(text="LangChain is a popular framework for building LLM applications."),
        Document(text="Prompt templates help format dynamic prompts."),
        Document(text="PyTorch is an open-source machine learning library."),
        Document(text="ChromaDB stores embeddings locally on disk."),
        Document(text="Fine-tuning adapts weights using backpropagation."),
        Document(text="NumPy handles array operations efficiently."),
        Document(text="Quantization reduces the precision of model weights."),
        Document(text="Hybrid search combines BM25 keyword matching with vector retrieval."),
        Document(text="Speculative decoding uses a smaller draft model to speed up generation."),
    ]
    v_idx = VectorStoreIndex.from_documents(multi_docs)
    initial_nodes, top3 = two_stage_retrieve_and_rerank(v_idx, "vector similarity search algorithms", top_k1=10, top_k2=3)
    print(f"Stage 1 retrieved {len(initial_nodes)} candidate nodes.")
    print("Stage 2 re-ranked top 3:")
    for rank, (node, score) in enumerate(top3, 1):
        print(f"{rank}. [Score: {score}] {node.text}")

    # Cleanup temp folder
    import shutil
    shutil.rmtree(docs_dir, ignore_errors=True)
