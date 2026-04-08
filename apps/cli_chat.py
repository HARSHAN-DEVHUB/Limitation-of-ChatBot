from __future__ import annotations

from pathlib import Path
import re
import sys
import uuid

from rich import print

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import settings
from src.db.sqlite import SQLiteStore
from src.index.retrieval import Retriever
from src.learning.filters import is_high_signal_turn
from src.llm.generator import generate_grounded_response
from src.memory.short_term import ShortTermMemory


GREETING_WORDS = {"hi", "hello", "hey", "good morning", "good evening", "good afternoon"}
THANKS_WORDS = {"thanks", "thank you", "thx"}


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _is_greeting(text: str) -> bool:
    msg = _normalize(text)
    return msg in GREETING_WORDS


def _is_thanks(text: str) -> bool:
    msg = _normalize(text)
    return msg in THANKS_WORDS


def _social_reply(text: str) -> str:
    if _is_greeting(text):
        return (
            "Hello. I am your offline university support bot. "
            "Ask me about registration, portal access, deadlines, or support procedures."
        )
    if _is_thanks(text):
        return "You are welcome. If you want, ask your next university support question."
    return "I am here and ready to help. Please ask your university support question."


def _is_refusal_answer(text: str) -> bool:
    msg = _normalize(text)
    return "do not have enough evidence" in msg or "i do not know" in msg or "i don't know" in msg


def _token_set(text: str) -> set[str]:
    return {tok for tok in re.findall(r"[a-z0-9]+", text.lower()) if len(tok) > 2}


def _keyword_memory_search(chunks: list[dict], query: str, limit: int = 2) -> list[dict]:
    q_tokens = _token_set(query)
    if not q_tokens:
        return []

    scored: list[dict] = []
    for chunk in chunks:
        metadata = chunk.get("metadata", {})
        if metadata.get("source") == "continuous_learning" and _is_refusal_answer(
            chunk.get("text", "")
        ):
            continue

        c_tokens = _token_set(chunk.get("text", ""))
        if not c_tokens:
            continue
        overlap = len(q_tokens.intersection(c_tokens))
        score = overlap / max(len(q_tokens), 1)
        if metadata.get("source") == "continuous_learning":
            score *= 0.6
        if score > 0:
            scored.append(
                {
                    "score": float(score),
                    "chunk_id": chunk["chunk_id"],
                    "text": chunk["text"],
                    "metadata": chunk.get("metadata", {}),
                }
            )

    scored.sort(key=lambda r: r["score"], reverse=True)
    return scored[:limit]


def _merge_results(primary: list[dict], secondary: list[dict], top_k: int) -> list[dict]:
    merged: dict[str, dict] = {}
    for row in primary + secondary:
        cid = row["chunk_id"]
        prev = merged.get(cid)
        if prev is None or row["score"] > prev["score"]:
            merged[cid] = row

    out = sorted(merged.values(), key=lambda r: r["score"], reverse=True)
    return out[:top_k]


def _focus_results(retrieved: list[dict], ratio: float = 0.8) -> list[dict]:
    if not retrieved:
        return []
    best = retrieved[0]["score"]
    cutoff = best * ratio
    focused = [row for row in retrieved if row["score"] >= cutoff]
    return focused or retrieved[:1]


def _has_agent_guidance(text: str) -> bool:
    for line in text.splitlines():
        if line.strip().lower().startswith("agent:"):
            return True
    return False


def run() -> None:
    db = SQLiteStore()
    memory = ShortTermMemory()
    session_id = str(uuid.uuid4())
    learned_idx = 0
    retriever = Retriever(top_k=settings.top_k)

    print("Offline University Support Chatbot")
    print("Type 'exit' to stop.\n")

    while True:
        user_message = input("You: ").strip()
        if user_message.lower() in {"exit", "quit"}:
            print("Goodbye")
            break

        memory.add("user", user_message)

        if _is_greeting(user_message) or _is_thanks(user_message):
            answer = _social_reply(user_message)
            citations: list[str] = []
            print(f"\nAssistant: {answer}\n")
            db.log_chat(session_id, "user", user_message)
            db.log_chat(session_id, "assistant", answer, citations)
            memory.add("assistant", answer)
            continue

        semantic_results = retriever.search(user_message)
        semantic_results = [
            r for r in semantic_results if r.get("score", 0.0) >= settings.min_retrieval_score
        ]

        # Always include a lexical pass over local DB memory so newly learned turns can be reused
        # without requiring any external API or cloud service.
        memory_results = _keyword_memory_search(db.fetch_chunks(), user_message, limit=2)
        retrieved = _merge_results(semantic_results, memory_results, top_k=settings.top_k)
        retrieved = _focus_results(retrieved, ratio=0.8)
        guidance_rows = [row for row in retrieved if _has_agent_guidance(row.get("text", ""))]
        if guidance_rows:
            retrieved = guidance_rows
        elif semantic_results:
            retrieved = semantic_results[:1]

        answer, citations = generate_grounded_response(user_message, retrieved)
        print(f"\nAssistant: {answer}\n")
        if citations:
            print(f"Citations: {', '.join(citations)}\n")

        db.log_chat(session_id, "user", user_message)
        db.log_chat(session_id, "assistant", answer, citations)
        memory.add("assistant", answer)

        # Continuous learning: store useful assistant answers as new local knowledge chunks.
        if is_high_signal_turn(answer) and citations and not _is_refusal_answer(answer):
            compact_answer = " ".join(answer.split())
            learned_chunk = {
                "chunk_id": f"learned:{session_id}:{learned_idx}",
                "doc_id": "chat_learning",
                "conv_id": session_id,
                "turn_start": learned_idx,
                "turn_end": learned_idx,
                "text": f"user: {user_message}\nagent: {compact_answer}",
                "metadata": {"source": "continuous_learning"},
            }
            db.upsert_chunks([learned_chunk])
            learned_idx += 1


if __name__ == "__main__":
    run()
