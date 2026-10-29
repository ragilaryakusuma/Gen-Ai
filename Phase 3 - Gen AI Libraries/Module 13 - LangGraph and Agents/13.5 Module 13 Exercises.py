# 13.5 Module 13 Exercises (Halaman 152)
from typing import TypedDict, Annotated, List
import operator
import sys
import os
from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()

# Setup LLM dengan safe fallback
llm_client = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    model="openrouter/free",
    api_key=os.environ.get("OPENROUTER_API_KEY", "mock-key"),
)

def safe_llm(prompt: str, fallback: str) -> str:
    try:
        res = llm_client.invoke(prompt)
        return res.content.strip()
    except Exception:
        return fallback


# ----------------------------------------------------------------------
# 1. Research Agent with 3 Tools: web_search, summarise, save_note
# ----------------------------------------------------------------------
class ResearchState(TypedDict):
    topic: str
    search_results: str
    summary: str
    saved_notes: Annotated[List[str], operator.add]

def node_web_search(state: ResearchState) -> dict:
    topic = state["topic"]
    results = f"Mock search results for '{topic}': Modern RAG uses hybrid search and dense vector re-ranking for higher precision."
    return {"search_results": results}

def node_summarise(state: ResearchState) -> dict:
    prompt = f"Summarize this in one sentence: {state['search_results']}"
    summary = safe_llm(prompt, "Hybrid search combined with vector re-ranking significantly improves retrieval precision.")
    return {"summary": summary}

def node_save_note(state: ResearchState) -> dict:
    note = f"[{state['topic'].upper()}] {state['summary']}"
    return {"saved_notes": [note]}

def build_research_agent():
    graph = StateGraph(ResearchState)
    graph.add_node("search", node_web_search)
    graph.add_node("summarise", node_summarise)
    graph.add_node("save_note", node_save_note)

    graph.add_edge(START, "search")
    graph.add_edge("search", "summarise")
    graph.add_edge("summarise", "save_note")
    graph.add_edge("save_note", END)
    return graph.compile()


# ----------------------------------------------------------------------
# 2. ReAct Agent with Maximum Step Limit (8 steps)
# ----------------------------------------------------------------------
class StepLimitState(TypedDict):
    step_count: int
    final_output: str

def react_step_node(state: StepLimitState) -> dict:
    new_step = state["step_count"] + 1
    return {"step_count": new_step}

def check_step_limit(state: StepLimitState) -> str:
    if state["step_count"] >= 8:
        return "limit_reached"
    return "continue_step"

def limit_reached_node(state: StepLimitState) -> dict:
    return {"final_output": "Reached step limit."}

def build_step_limited_agent():
    graph = StateGraph(StepLimitState)
    graph.add_node("step", react_step_node)
    graph.add_node("limit", limit_reached_node)

    graph.add_edge(START, "step")
    graph.add_conditional_edges(
        "step",
        check_step_limit,
        {"limit_reached": "limit", "continue_step": "step"}
    )
    graph.add_edge("limit", END)
    return graph.compile()


# ----------------------------------------------------------------------
# 3. PlanAndExecute Agent (Planner + Executor)
# ----------------------------------------------------------------------
class PlanExecuteState(TypedDict):
    goal: str
    plan: List[str]
    current_step: int
    execution_log: Annotated[List[str], operator.add]

def planner_node(state: PlanExecuteState) -> dict:
    plan = [
        "1. Ingest documents and generate chunks",
        "2. Compute vector embeddings with local model",
        "3. Store vectors in ChromaDB and run query test",
    ]
    return {"plan": plan, "current_step": 0}

def executor_node(state: PlanExecuteState) -> dict:
    idx = state["current_step"]
    step_desc = state["plan"][idx]
    log_entry = f"Executed step {idx + 1}: {step_desc}"
    return {"execution_log": [log_entry], "current_step": idx + 1}

def should_continue_executing(state: PlanExecuteState) -> str:
    if state["current_step"] < len(state["plan"]):
        return "executor"
    return END

def build_plan_execute_agent():
    graph = StateGraph(PlanExecuteState)
    graph.add_node("planner", planner_node)
    graph.add_node("executor", executor_node)

    graph.add_edge(START, "planner")
    graph.add_edge("planner", "executor")
    graph.add_conditional_edges("executor", should_continue_executing, {"executor": "executor", END: END})
    return graph.compile()


# ----------------------------------------------------------------------
# 4. Multi-turn Agent with MemorySaver (threads)
# ----------------------------------------------------------------------
class ConversationState(TypedDict):
    messages: Annotated[List[str], operator.add]

def chatbot_node(state: ConversationState) -> dict:
    last_msg = state["messages"][-1]
    response = f"Acknowledged: {last_msg}"
    return {"messages": [response]}

def build_memory_agent():
    memory = MemorySaver()
    graph = StateGraph(ConversationState)
    graph.add_node("chat", chatbot_node)
    graph.add_edge(START, "chat")
    graph.add_edge("chat", END)
    return graph.compile(checkpointer=memory)


if __name__ == "__main__":
    # --- Exercise 1 ---
    print("[Exercise 1] LangGraph Research Agent")
    research_agent = build_research_agent()
    r_state = research_agent.invoke({"topic": "Hybrid Search", "saved_notes": []})
    print(f"Summary    : {r_state['summary']}")
    print(f"Saved Notes: {r_state['saved_notes']}")

    # --- Exercise 2 ---
    print("\n[Exercise 2] ReAct Step Limit Guardrail")
    step_agent = build_step_limited_agent()
    step_result = step_agent.invoke({"step_count": 0, "final_output": ""})
    print(f"Final status after {step_result['step_count']} steps: {step_result['final_output']}")

    # --- Exercise 3 ---
    print("\n[Exercise 3] Plan-and-Execute Agent")
    plan_agent = build_plan_execute_agent()
    plan_result = plan_agent.invoke({"goal": "Deploy RAG pipeline", "execution_log": []})
    print(f"Plan steps planned: {len(plan_result['plan'])}")
    for log in plan_result["execution_log"]:
        print(f" - {log}")

    # --- Exercise 4 ---
    print("\n[Exercise 4] Multi-Turn Agent with MemorySaver")
    mem_agent = build_memory_agent()
    config = {"configurable": {"thread_id": "session_user_42"}}

    # Turn 1
    res1 = mem_agent.invoke({"messages": ["My name is Alice and I like Python."]}, config=config)
    print("Turn 1 response:", res1["messages"][-1])

    # Turn 2 in same thread
    res2 = mem_agent.invoke({"messages": ["What language did I mention?"]}, config=config)
    print("Turn 2 response:", res2["messages"][-1])
    print(f"Total message history in thread: {len(res2['messages'])} entries")
