# 14.5 Crew - Putting It Together (Halaman 159-161)
from crewai import Crew, Process, Agent, LLM
import os
import importlib.util
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

if __name__ == "__main__":
    module_dir = Path(__file__).parent
    
    def load_module_from_file(module_name, file_path):
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    tools_mod = load_module_from_file("tools_14_3", module_dir / "14.3 Tools.py")
    tasks_mod = load_module_from_file("tasks_14_4", module_dir / "14.4 Tasks.py")

    cost_tool = tools_mod.cost_tool
    context_tool = tools_mod.context_tool
    make_tasks = tasks_mod.make_tasks

    llm = LLM(
        model="openrouter/openrouter/free",
        api_key=os.environ.get("OPENROUTER_API_KEY"),
    )

    researcher = Agent(
        role="AI Research Analyst",
        goal="Find accurate information on AI topics.",
        backstory="Senior AI research analyst with expertise in LLM systems.",
        llm=llm,
        tools=[context_tool],
        verbose=False,
        max_iter=5,
    )
    writer = Agent(
        role="Technical Content Writer",
        goal="Write clear technical content for developers.",
        backstory="Technical writer with a software engineering background.",
        llm=llm,
        tools=[cost_tool],
        verbose=False,
        max_iter=5,
    )
    reviewer = Agent(
        role="Content Quality Reviewer",
        goal="Ensure content is accurate and publication-ready.",
        backstory="Meticulous editor who reviews technical AI content.",
        llm=llm,
        verbose=False,
        max_iter=5,
    )

    research_task, writing_task, review_task = make_tasks(researcher, writer, reviewer)

    crew = Crew(
        agents=[researcher, writer, reviewer],
        tasks=[research_task, writing_task, review_task],
        process=Process.sequential,
        verbose=False,
        memory=False,
    )

    print("Sequential Crew assembled successfully.")
