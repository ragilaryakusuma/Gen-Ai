# 14.7 Module 14 Exercises (Halaman 163)
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type
import os
import sys
import time
from dotenv import load_dotenv

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()

# Setup default LLM
llm = LLM(
    model="openrouter/openrouter/free",
    api_key=os.environ.get("OPENROUTER_API_KEY", "mock-key"),
)


# ----------------------------------------------------------------------
# 3. Custom WebSearchTool with Mock Search
# ----------------------------------------------------------------------
class WebSearchInput(BaseModel):
    query: str = Field(description="Search query string")

class WebSearchTool(BaseTool):
    name: str = "Web Search Tool"
    description: str = "Search technical documentation and articles."
    args_schema: Type[BaseModel] = WebSearchInput

    def _run(self, query: str) -> str:
        mock_database = {
            "rag": "RAG connects an external vector database (e.g. ChromaDB) to an LLM for real-time factual knowledge retrieval.",
            "fine-tuning": "Fine-tuning modifies the model weights on a curated domain dataset for specialized style or tasks.",
        }
        for k, val in mock_database.items():
            if k in query.lower():
                return f"Search result: {val}"
        return f"Search result for '{query}': RAG retrieves dynamic data, while fine-tuning bakes static knowledge into model weights."


# ----------------------------------------------------------------------
# 1. Three-Agent Content Pipeline (Researcher, Outline Writer, Full Writer)
# ----------------------------------------------------------------------
def build_content_pipeline():
    researcher = Agent(
        role="Technical Researcher",
        goal="Find key technical differences between RAG and Fine-Tuning.",
        backstory="Senior AI researcher who identifies architectural trade-offs.",
        llm=llm,
        verbose=False,
    )
    outline_writer = Agent(
        role="Outline Architect",
        goal="Structure the findings into a clear, logical blog outline.",
        backstory="Technical editor specialized in developer-friendly content structures.",
        llm=llm,
        verbose=False,
    )
    full_writer = Agent(
        role="Technical Content Writer",
        goal="Draft the final complete post from the outline.",
        backstory="Experienced engineering author known for clean and practical guides.",
        llm=llm,
        verbose=False,
    )

    t1 = Task(
        description="Research key differences, costs, and trade-offs between Fine-Tuning and RAG.",
        expected_output="Bulleted list of key comparison points.",
        agent=researcher,
    )
    t2 = Task(
        description="Create a 3-section blog outline based on the research.",
        expected_output="Structured Markdown outline with headings.",
        agent=outline_writer,
        context=[t1],
    )
    t3 = Task(
        description="Write the practical guide blog post based on the outline.",
        expected_output="Final Markdown blog post of 200 words.",
        agent=full_writer,
        context=[t2],
    )

    crew = Crew(
        agents=[researcher, outline_writer, full_writer],
        tasks=[t1, t2, t3],
        process=Process.sequential,
        verbose=False,
    )
    return crew


# ----------------------------------------------------------------------
# 2. Code Review Crew (Developer, Security, Performance, Lead)
# ----------------------------------------------------------------------
def build_code_review_crew():
    developer = Agent(role="Developer", goal="Write initial code snippet.", backstory="Python dev.", llm=llm, verbose=False)
    sec_reviewer = Agent(role="Security Reviewer", goal="Audit vulnerabilities.", backstory="AppSec engineer.", llm=llm, verbose=False)
    perf_reviewer = Agent(role="Performance Reviewer", goal="Identify bottlenecks.", backstory="Perf engineer.", llm=llm, verbose=False)
    lead = Agent(role="Tech Lead", goal="Merge review feedback.", backstory="Engineering manager.", llm=llm, verbose=False)

    task_sec = Task(description="Review code for SQL injection, eval(), or unsanitized inputs.", expected_output="Security report.", agent=sec_reviewer)
    task_perf = Task(description="Check for O(n^2) loops or excessive DB queries.", expected_output="Performance report.", agent=perf_reviewer)
    task_lead = Task(description="Compile final review decision.", expected_output="Final decision.", agent=lead, context=[task_sec, task_perf])

    return Crew(
        agents=[developer, sec_reviewer, perf_reviewer, lead],
        tasks=[task_sec, task_perf, task_lead],
        process=Process.sequential,
        verbose=False,
    )


# ----------------------------------------------------------------------
# 4. Compare Sequential vs Hierarchical Process
# ----------------------------------------------------------------------
def explain_process_comparison():
    comparison = {
        "Process.sequential": {
            "execution": "Tasks run in a strict linear order (A -> B -> C).",
            "overhead": "Minimal token overhead, no extra coordinator calls.",
            "best_for": "Predictable workflows with clear handoffs (e.g., Research -> Draft -> Edit).",
        },
        "Process.hierarchical": {
            "execution": "A manager agent plans, assigns, and evaluates sub-tasks dynamically.",
            "overhead": "Higher token cost and latency due to manager delegation loops.",
            "best_for": "Complex open-ended goals where sub-tasks cannot be fully predefined.",
        },
    }
    return comparison


if __name__ == "__main__":
    # --- Exercise 3 ---
    print("[Exercise 3] Custom WebSearchTool Verification")
    search_tool = WebSearchTool()
    tool_out = search_tool._run("fine-tuning vs RAG")
    print(tool_out)

    # --- Exercise 1 ---
    print("\n[Exercise 1] Content Pipeline Crew Assembly")
    content_crew = build_content_pipeline()
    print(f"Content pipeline assembled with {len(content_crew.agents)} agents and {len(content_crew.tasks)} sequential tasks.")

    # --- Exercise 2 ---
    print("\n[Exercise 2] Code Review Crew Assembly")
    review_crew = build_code_review_crew()
    print(f"Code review crew assembled with {len(review_crew.agents)} specialist roles.")

    # --- Exercise 4 ---
    print("\n[Exercise 4] Process.sequential vs Process.hierarchical Comparison")
    comp = explain_process_comparison()
    for proc_name, details in comp.items():
        print(f"\n{proc_name}:")
        print(f" - Execution: {details['execution']}")
        print(f" - Overhead : {details['overhead']}")
        print(f" - Best for : {details['best_for']}")
