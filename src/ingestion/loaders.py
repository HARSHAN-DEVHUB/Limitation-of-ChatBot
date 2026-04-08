from __future__ import annotations

import json
from pathlib import Path


def iter_jsonl_rows(input_path: Path):
    with input_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def normalize_turn(row: dict, fallback_doc_id: str) -> dict:
    return {
        "doc_id": str(row.get("doc_id", fallback_doc_id)),
        "conv_id": str(row.get("conv_id", "unknown")),
        "turn_id": int(row.get("turn_id", 0)),
        "speaker": str(row.get("speaker", "unknown")),
        "text": str(row.get("text", "")).strip(),
        "metadata": row.get("metadata", {}),
    }


def load_and_normalize_jsonl(input_path: Path) -> list[dict]:
    fallback_doc_id = input_path.stem
    rows = [normalize_turn(row, fallback_doc_id) for row in iter_jsonl_rows(input_path)]
    rows = [r for r in rows if r["text"]]
    rows.sort(key=lambda r: (r["conv_id"], r["turn_id"]))
    return rows
