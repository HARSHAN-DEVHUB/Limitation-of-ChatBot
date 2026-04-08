from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.evaluation.runner import run_eval_suite, summarize_results


def main(suite: Path, mode: str) -> None:
    results = run_eval_suite(suite, mode=mode)
    summary = summarize_results(results)
    print(json.dumps({"mode": mode, "summary": summary}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--mode", type=str, default="improved", choices=["improved", "baseline"])
    args = parser.parse_args()
    main(args.suite, args.mode)
