from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import settings

from scripts.weekly_memory_update import main as weekly_memory_main
from scripts.weekly_weight_update import main as weekly_weight_main


def main(min_new_chats_memory: int, min_new_chats_weight: int) -> None:
    weekly_memory_main(
        state_path=Path("data/processed/weekly_memory_state.json"),
        min_new_chats=min_new_chats_memory,
    )
    weekly_weight_main(
        state_path=settings.weight_update_state_path,
        min_new_chats=min_new_chats_weight,
    )
    print("Full weekly pipeline completed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-new-chats-memory", type=int, default=1)
    parser.add_argument("--min-new-chats-weight", type=int, default=settings.weight_update_min_new_chats)
    args = parser.parse_args()
    main(
        min_new_chats_memory=args.min_new_chats_memory,
        min_new_chats_weight=args.min_new_chats_weight,
    )
