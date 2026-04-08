from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from datasets import load_dataset


def _iter_daily_dialog_rows(split: str, limit: int):
    ds = load_dataset("daily_dialog", split=split)
    count = 0
    for idx, row in enumerate(ds):
        utts = row.get("dialog", [])
        conv_id = f"daily_dialog_{split}_{idx:06d}"
        for tid, utter in enumerate(utts, start=1):
            speaker = "student" if tid % 2 == 1 else "agent"
            yield {
                "doc_id": f"daily_dialog_{split}",
                "conv_id": conv_id,
                "turn_id": tid,
                "speaker": speaker,
                "text": str(utter).strip(),
            }
        count += 1
        if count >= limit:
            break


def main(dataset: str, split: str, limit: int, output: Path) -> None:
    if dataset != "daily_dialog":
        raise ValueError("Currently supported dataset values: daily_dialog")

    output.parent.mkdir(parents=True, exist_ok=True)
    rows = _iter_daily_dialog_rows(split=split, limit=limit)

    written = 0
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            if not row["text"]:
                continue
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1

    print(f"Imported {written} turns from {dataset}/{split} to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="daily_dialog")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--limit", type=int, default=2000)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/raw/hf_daily_dialog_train.jsonl"),
    )
    args = parser.parse_args()
    main(dataset=args.dataset, split=args.split, limit=args.limit, output=args.output)
