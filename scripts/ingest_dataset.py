from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import settings
from src.db.sqlite import SQLiteStore
from src.ingestion.chunking import chunk_conversations
from src.ingestion.loaders import load_and_normalize_jsonl
from src.ingestion.preprocess import preprocess_rows


def main(input: Path = settings.raw_data_dir, output: Path = settings.processed_data_dir) -> None:
    output.mkdir(parents=True, exist_ok=True)
    db = SQLiteStore()

    total_rows = 0
    total_chunks = 0

    for input_file in sorted(input.glob("*.jsonl")):
        rows = load_and_normalize_jsonl(input_file)
        clean_rows = preprocess_rows(rows)
        chunks = chunk_conversations(
            clean_rows,
            max_chars=settings.chunk_size,
            overlap_chars=settings.chunk_overlap,
        )

        total_rows += len(clean_rows)
        total_chunks += db.upsert_chunks(chunks)
        db.log_ingestion_run(str(input_file), len(clean_rows), len(chunks), "ingested")

        out_file = output / f"{input_file.stem}.chunks.jsonl"
        with out_file.open("w", encoding="utf-8") as handle:
            for chunk in chunks:
                handle.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    print(f"Rows processed: {total_rows}")
    print(f"Chunks stored: {total_chunks}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=settings.raw_data_dir)
    parser.add_argument("--output", type=Path, default=settings.processed_data_dir)
    args = parser.parse_args()
    main(input=args.input, output=args.output)
