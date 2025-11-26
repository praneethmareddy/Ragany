import os
from dotenv import load_dotenv
from raganything import RAGAnything

os.environ["LLM_BINDING"] = "ollama"
os.environ["LLM_BINDING_HOST"] = "http://localhost:11434"
os.environ["LLM_MODEL"] = "gpt-oss:20b"  # ← EXACT NAME from your screenshot
os.environ["EMBEDDING_MODEL"] = "text-embedding-nomic-embed-text-v1.5"
os.environ["CHUNK_SIZE"] = "500"
os.environ["CHUNK_OVERLAP"] = "50"

load_dotenv()

rag = RAGAnything()

rag.index("docs")

while True:
    question = input("\nYou: ")
    if question.lower() in ["exit", "quit"]:
        break

    print("\nAssistant:")
    for chunk in rag.stream(question):
        print(chunk, end="", flush=True)
    print("\n")
