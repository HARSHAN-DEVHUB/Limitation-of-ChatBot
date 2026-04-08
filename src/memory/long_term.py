from __future__ import annotations

from src.db.sqlite import SQLiteStore


def fetch_chat_history(session_id: str) -> list[dict]:
    rows = SQLiteStore().fetch_chat_rows()
    return [row for row in rows if row["session_id"] == session_id]
