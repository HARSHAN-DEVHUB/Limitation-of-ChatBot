from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Iterable

from src.config import settings


class SQLiteStore:
    def __init__(self, db_path: Path | None = None) -> None:
        self.db_path = db_path or settings.sqlite_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        schema_path = Path(__file__).with_name("schema.sql")
        schema = schema_path.read_text(encoding="utf-8")
        with self._connect() as conn:
            conn.executescript(schema)

    def upsert_chunks(self, rows: Iterable[dict]) -> int:
        payload = [
            (
                row["chunk_id"],
                row["doc_id"],
                row.get("conv_id"),
                row.get("turn_start"),
                row.get("turn_end"),
                row["text"],
                json.dumps(row.get("metadata", {})),
            )
            for row in rows
        ]
        if not payload:
            return 0

        with self._connect() as conn:
            conn.executemany(
                """
                INSERT INTO chunks (chunk_id, doc_id, conv_id, turn_start, turn_end, text, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(chunk_id) DO UPDATE SET
                    text = excluded.text,
                    metadata_json = excluded.metadata_json
                """,
                payload,
            )
        return len(payload)

    def fetch_chunks(self) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT chunk_id, doc_id, conv_id, turn_start, turn_end, text, metadata_json FROM chunks"
            ).fetchall()
        out = []
        for row in rows:
            out.append(
                {
                    "chunk_id": row["chunk_id"],
                    "doc_id": row["doc_id"],
                    "conv_id": row["conv_id"],
                    "turn_start": row["turn_start"],
                    "turn_end": row["turn_end"],
                    "text": row["text"],
                    "metadata": json.loads(row["metadata_json"] or "{}"),
                }
            )
        return out

    def log_chat(self, session_id: str, role: str, text: str, citations: list[str] | None = None) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO chats (session_id, role, text, citations_json) VALUES (?, ?, ?, ?)",
                (session_id, role, text, json.dumps(citations or [])),
            )

    def fetch_chat_rows(self) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, session_id, role, text, citations_json, created_at FROM chats ORDER BY id"
            ).fetchall()
        return [
            {
                "id": row["id"],
                "session_id": row["session_id"],
                "role": row["role"],
                "text": row["text"],
                "citations": json.loads(row["citations_json"] or "[]"),
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    def log_ingestion_run(
        self, source_path: str, input_rows: int, output_chunks: int, notes: str = ""
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO ingestion_runs (source_path, input_rows, output_chunks, notes)
                VALUES (?, ?, ?, ?)
                """,
                (source_path, input_rows, output_chunks, notes),
            )
