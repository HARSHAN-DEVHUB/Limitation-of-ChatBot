from __future__ import annotations

from src.learning.filters import is_high_signal_turn


def chat_rows_to_chunks(rows: list[dict], session_doc_prefix: str = "chatlog") -> list[dict]:
    chunks = []
    idx = 0
    for row in rows:
        if row["role"] != "assistant":
            continue
        if not is_high_signal_turn(row["text"]):
            continue
        chunk_id = f"{session_doc_prefix}:{row['session_id']}:{idx}"
        chunks.append(
            {
                "chunk_id": chunk_id,
                "doc_id": session_doc_prefix,
                "conv_id": row["session_id"],
                "turn_start": row["id"],
                "turn_end": row["id"],
                "text": row["text"],
                "metadata": {"source": "chatlog", "created_at": row["created_at"]},
            }
        )
        idx += 1
    return chunks
