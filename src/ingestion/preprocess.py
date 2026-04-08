from __future__ import annotations

import re


_WS_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    text = text.replace("\u00a0", " ")
    text = _WS_RE.sub(" ", text).strip()
    return text


def preprocess_rows(rows: list[dict], min_chars: int = 5) -> list[dict]:
    cleaned = []
    for row in rows:
        text = clean_text(row["text"])
        if len(text) < min_chars:
            continue
        out = dict(row)
        out["text"] = text
        cleaned.append(out)
    return cleaned
