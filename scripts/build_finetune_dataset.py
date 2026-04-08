from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.db.sqlite import SQLiteStore
from src.learning.finetune_dataset import (
    build_dataset_from_chat_rows,
    build_dataset_from_transcript_jsonl,
    split_train_val,
    write_jsonl,
)


def main(transcript_jsonl: Path, output_dir: Path, max_transcript_samples: int) -> None:
    db = SQLiteStore()
    chat_rows = db.fetch_chat_rows()

    chat_samples = build_dataset_from_chat_rows(chat_rows)
    transcript_samples = []
    if transcript_jsonl.exists():
        transcript_samples = build_dataset_from_transcript_jsonl(
            transcript_jsonl,
            max_samples=max_transcript_samples,
        )

    merged = chat_samples + transcript_samples
    train, val = split_train_val(merged, val_ratio=0.1, seed=42)

    train_path = output_dir / "train.jsonl"
    val_path = output_dir / "val.jsonl"
    write_jsonl(train_path, train)
    write_jsonl(val_path, val)

    print(f"Built finetune dataset: train={len(train)}, val={len(val)}")
    print(f"Train path: {train_path}")
    print(f"Val path: {val_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--transcript-jsonl",
        type=Path,
        default=Path("data/raw/synthetic_university_calls.jsonl"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/finetune"),
    )
    parser.add_argument("--max-transcript-samples", type=int, default=4000)
    args = parser.parse_args()
    main(
        transcript_jsonl=args.transcript_jsonl,
        output_dir=args.output_dir,
        max_transcript_samples=args.max_transcript_samples,
    )
