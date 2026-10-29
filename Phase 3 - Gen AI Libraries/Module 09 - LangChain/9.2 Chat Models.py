# 9.2 Chat Models (Halaman 101-103)
from langchain_openai import ChatOpenAI
#from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
import os
from dotenv import load_dotenv

load_dotenv()

# OpenAI model via OpenRouter
llm_oai = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    model="inclusionai/ling-3.0-flash-sante:free",
    temperature=0.7,
    max_tokens=1024,
    api_key=os.environ.get("OPENROUTER_API_KEY", "your-openrouter-key"),
)

# Anthropic model via OpenRouter - same interface
llm_ant = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    model="inclusionai/ling-3.0-flash-sante:free",
    temperature=0.7,
    max_tokens=1024,
    api_key=os.environ.get("OPENROUTER_API_KEY", "your-openrouter-key"),
)

# Both accept the same message types
messages = [
    SystemMessage(content="You are a concise technical assistant."),
    HumanMessage(content="What is LangChain Expression Language?"),
]

if __name__ == "__main__":
    # Invoke (blocking)
    print("=== Invoke Anthropic ===")
    response = llm_ant.invoke(messages)
    print(response.content)

    # Stream
    print("\n=== Stream Anthropic ===")
    for chunk in llm_ant.stream(messages):
        print(chunk.content, end="", flush=True)
    print()

    # Batch - run multiple inputs in parallel
    print("\n=== Batch Anthropic ===")
    responses = llm_ant.batch([
        [HumanMessage(content="What is RAG?")],
        [HumanMessage(content="What is a vector database?")],
    ])
    for r in responses:
        print(r.content[:80])
