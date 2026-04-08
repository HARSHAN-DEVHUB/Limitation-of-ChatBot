from __future__ import annotations

from pathlib import Path
import sys
import uuid

from rich import print

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import settings
from src.db.sqlite import SQLiteStore
from src.index.retrieval import Retriever
from src.llm.generator import generate_grounded_response
from src.memory.short_term import ShortTermMemory


def run() -> None:
    db = SQLiteStore()
    memory = ShortTermMemory()
    session_id = str(uuid.uuid4())
    retriever = Retriever(top_k=settings.top_k)

    print("Offline University Support Chatbot")
    print("Type 'exit' to stop.\n")

    while True:
        user_message = input("You: ").strip()
        if user_message.lower() in {"exit", "quit"}:
            print("Goodbye")
            break

        memory.add("user", user_message)

        retrieved = retriever.search(user_message)
        retrieved = [r for r in retrieved if r.get("score", 0.0) >= settings.min_retrieval_score]

        answer, citations = generate_grounded_response(user_message, retrieved)
        print(f"\nAssistant: {answer}\n")
        if citations:
            print(f"Citations: {', '.join(citations)}\n")

        db.log_chat(session_id, "user", user_message)
        db.log_chat(session_id, "assistant", answer, citations)
        memory.add("assistant", answer)


if __name__ == "__main__":
    run()
