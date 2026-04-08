from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.db.sqlite import SQLiteStore


def main() -> None:
    rows = SQLiteStore().fetch_chat_rows()
    out_path = Path("data/processed/chatlogs.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Exported {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
