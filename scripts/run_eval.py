from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.evaluation.runner import run_eval_suite, summarize_results


def main(suite: Path) -> None:
    baseline_results = run_eval_suite(suite, mode="baseline")
    improved_results = run_eval_suite(suite, mode="improved")

    out_dir = Path("data/eval/results")
    out_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    baseline_path = out_dir / f"eval_baseline_{ts}.json"
    improved_path = out_dir / f"eval_improved_{ts}.json"
    summary_path = out_dir / f"eval_summary_{ts}.csv"

    baseline_path.write_text(json.dumps(baseline_results, indent=2), encoding="utf-8")
    improved_path.write_text(json.dumps(improved_results, indent=2), encoding="utf-8")

    baseline_summary = summarize_results(baseline_results)
    improved_summary = summarize_results(improved_results)

    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "mode",
                "count",
                "avg_specificity",
                "avg_refusal_quality",
                "avg_citation_count",
            ],
        )
        writer.writeheader()
        writer.writerow({"mode": "baseline", **baseline_summary})
        writer.writerow({"mode": "improved", **improved_summary})

    print(f"Saved baseline results to {baseline_path}")
    print(f"Saved improved results to {improved_path}")
    print(f"Saved summary to {summary_path}")
    print("Summary:")
    print(f"  baseline avg_specificity={baseline_summary['avg_specificity']}")
    print(f"  improved avg_specificity={improved_summary['avg_specificity']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", type=Path, required=True)
    args = parser.parse_args()
    main(suite=args.suite)
