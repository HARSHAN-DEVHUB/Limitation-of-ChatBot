from __future__ import annotations

from collections import defaultdict


def chunk_conversations(rows: list[dict], max_chars: int, overlap_chars: int) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["doc_id"], row["conv_id"])].append(row)

    chunks: list[dict] = []
    for (doc_id, conv_id), conv_rows in grouped.items():
        buffer: list[dict] = []
        buffer_chars = 0
        chunk_idx = 0

        for row in conv_rows:
            text = f"{row['speaker']}: {row['text']}"
            row_len = len(text)

            if buffer and buffer_chars + row_len > max_chars:
                chunks.append(_build_chunk(doc_id, conv_id, chunk_idx, buffer))
                chunk_idx += 1

                if overlap_chars > 0:
                    overlap_buffer = []
                    overlap_len = 0
                    for prev in reversed(buffer):
                        prev_text = f"{prev['speaker']}: {prev['text']}"
                        overlap_len += len(prev_text)
                        overlap_buffer.append(prev)
                        if overlap_len >= overlap_chars:
                            break
                    buffer = list(reversed(overlap_buffer))
                    buffer_chars = sum(len(f"{r['speaker']}: {r['text']}") for r in buffer)
                else:
                    buffer = []
                    buffer_chars = 0

            buffer.append(row)
            buffer_chars += row_len

        if buffer:
            chunks.append(_build_chunk(doc_id, conv_id, chunk_idx, buffer))

    return chunks


def _build_chunk(doc_id: str, conv_id: str, chunk_idx: int, rows: list[dict]) -> dict:
    text = "\n".join(f"{row['speaker']}: {row['text']}" for row in rows)
    turn_start = rows[0]["turn_id"]
    turn_end = rows[-1]["turn_id"]
    chunk_id = f"{doc_id}:{conv_id}:{chunk_idx}"
    return {
        "chunk_id": chunk_id,
        "doc_id": doc_id,
        "conv_id": conv_id,
        "turn_start": turn_start,
        "turn_end": turn_end,
        "text": text,
        "metadata": {
            "chunk_idx": chunk_idx,
            "row_count": len(rows),
        },
    }
