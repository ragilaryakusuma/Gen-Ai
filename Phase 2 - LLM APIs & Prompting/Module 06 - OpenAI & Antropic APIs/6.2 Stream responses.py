import ollama

response = ollama.chat(
    model="qwen3:8b",
    messages=[
        {
            "role": "user",
            "content": "List 5 use cases for vector databases."
        }
    ],
    stream=True
)

for chunk in response:
    print(chunk["message"]["content"], end="", flush=True)

print()