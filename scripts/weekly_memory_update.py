from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.db.sqlite import SQLiteStore
from src.embedding.embedder import LocalEmbedder
from src.index.faiss_index import FaissStore
from src.learning.filters import is_high_signal_turn


def _load_state(path: Path) -> dict:
    if not path.exists():
        return {"last_chat_id": 0, "last_run_utc": None}
    return json.loads(path.read_text(encoding="utf-8"))


def _save_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2), encoding="utf-8")


def _rows_to_chunks(rows: list[dict]) -> list[dict]:
    chunks: list[dict] = []
    for row in rows:
        if row["role"] != "assistant":
            continue
        if not is_high_signal_turn(row["text"]):
            continue

        chunk_id = f"chatlog:{row['session_id']}:{row['id']}"
        chunks.append(
            {
                "chunk_id": chunk_id,
                "doc_id": "chatlog",
                "conv_id": row["session_id"],
                "turn_start": row["id"],
                "turn_end": row["id"],
                "text": row["text"],
                "metadata": {
                    "source": "weekly_memory_update",
                    "created_at": row["created_at"],
                },
            }
        )
    return chunks


def _rebuild_index(db: SQLiteStore) -> int:
    chunks = db.fetch_chunks()
    if not chunks:
        return 0

    embedder = LocalEmbedder()
    vectors = embedder.encode([chunk["text"] for chunk in chunks])
    FaissStore().save(vectors, chunks)
    return len(chunks)


def main(state_path: Path, min_new_chats: int) -> None:
    db = SQLiteStore()
    state = _load_state(state_path)
    last_chat_id = int(state.get("last_chat_id", 0))

    all_rows = db.fetch_chat_rows()
    new_rows = [row for row in all_rows if row["id"] > last_chat_id]

    if len(new_rows) < min_new_chats:
        print(
            f"Skipped weekly update: only {len(new_rows)} new chat rows (min required: {min_new_chats})"
        )
        return

    new_chunks = _rows_to_chunks(new_rows)
    upserted = db.upsert_chunks(new_chunks)
    indexed_count = _rebuild_index(db)

    max_chat_id = max((row["id"] for row in all_rows), default=last_chat_id)
    next_state = {
        "last_chat_id": max_chat_id,
        "last_run_utc": datetime.now(timezone.utc).isoformat(),
        "new_chat_rows": len(new_rows),
        "new_chunks_upserted": upserted,
        "total_indexed_chunks": indexed_count,
    }
    _save_state(state_path, next_state)

    db.log_ingestion_run(
        source_path="weekly_memory_update",
        input_rows=len(new_rows),
        output_chunks=upserted,
        notes=f"indexed_total={indexed_count}",
    )

    print(f"Weekly memory update complete: {upserted} chunks upserted, {indexed_count} chunks indexed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--state-path",
        type=Path,
        default=Path("data/processed/weekly_memory_state.json"),
    )
    parser.add_argument("--min-new-chats", type=int, default=1)
    args = parser.parse_args()
    main(state_path=args.state_path, min_new_chats=args.min_new_chats)
