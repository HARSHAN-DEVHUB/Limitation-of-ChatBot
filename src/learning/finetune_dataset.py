from __future__ import annotations

import json
import random
from pathlib import Path


def _build_sample(user_text: str, assistant_text: str) -> dict:
    prompt = (
        "You are an offline university support assistant. "
        "Answer with concise, actionable steps grounded in available policy memory.\n\n"
        f"User: {user_text}\nAssistant:"
    )
    return {"prompt": prompt, "response": assistant_text}


def build_dataset_from_chat_rows(rows: list[dict]) -> list[dict]:
    samples: list[dict] = []
    current_user: str | None = None

    for row in rows:
        role = row.get("role", "")
        text = str(row.get("text", "")).strip()
        if not text:
            continue

        if role == "user":
            current_user = text
            continue

        if role == "assistant" and current_user:
            if "do not have enough evidence" in text.lower():
                current_user = None
                continue
            samples.append(_build_sample(current_user, text))
            current_user = None

    return samples


def build_dataset_from_transcript_jsonl(path: Path, max_samples: int = 4000) -> list[dict]:
    grouped: dict[str, list[dict]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            grouped.setdefault(row["conv_id"], []).append(row)

    samples: list[dict] = []
    for conv_rows in grouped.values():
        conv_rows.sort(key=lambda r: r.get("turn_id", 0))
        current_user: str | None = None
        for row in conv_rows:
            speaker = str(row.get("speaker", "")).lower()
            text = str(row.get("text", "")).strip()
            if not text:
                continue

            if speaker == "student":
                current_user = text
                continue

            if speaker == "agent" and current_user:
                samples.append(_build_sample(current_user, text))
                current_user = None

            if len(samples) >= max_samples:
                return samples

    return samples


def split_train_val(samples: list[dict], val_ratio: float = 0.1, seed: int = 42) -> tuple[list[dict], list[dict]]:
    if not samples:
        return [], []

    rng = random.Random(seed)
    pool = samples[:]
    rng.shuffle(pool)
    val_size = max(1, int(len(pool) * val_ratio))
    val = pool[:val_size]
    train = pool[val_size:]
    return train, val


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
