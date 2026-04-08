from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import settings
from src.db.sqlite import SQLiteStore

from scripts.build_finetune_dataset import main as build_finetune_dataset_main
from scripts.train_lora import train_lora


def _load_state(path: Path) -> dict:
    if not path.exists():
        return {
            "last_chat_id": 0,
            "active_adapter_path": "",
            "best_eval_loss": None,
            "last_run_utc": None,
        }
    return json.loads(path.read_text(encoding="utf-8"))


def _save_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2), encoding="utf-8")


def _run_summary_for_adapter(adapter_path: str) -> dict:
    env = os.environ.copy()
    env["LLM_BACKEND"] = "hf_local"
    env["HF_BASE_MODEL"] = settings.hf_base_model
    env["HF_ACTIVE_ADAPTER_PATH"] = adapter_path

    cmd = [
        sys.executable,
        "scripts/eval_backend_summary.py",
        "--suite",
        str(settings.weight_eval_suite_path),
        "--mode",
        "improved",
    ]
    proc = subprocess.run(
        cmd,
        cwd=ROOT_DIR,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    data = json.loads(proc.stdout)
    return data["summary"]


def _passes_metric_gate(summary: dict) -> tuple[bool, list[str]]:
    checks = [
        (
            summary.get("avg_specificity", 0.0) >= settings.weight_gate_min_specificity,
            f"specificity<{settings.weight_gate_min_specificity}",
        ),
        (
            summary.get("avg_refusal_quality", 0.0) >= settings.weight_gate_min_refusal_quality,
            f"refusal<{settings.weight_gate_min_refusal_quality}",
        ),
        (
            summary.get("avg_citation_count", 0.0) >= settings.weight_gate_min_citation_count,
            f"citation<{settings.weight_gate_min_citation_count}",
        ),
    ]
    failed = [label for ok, label in checks if not ok]
    return len(failed) == 0, failed


def main(state_path: Path, min_new_chats: int) -> None:
    db = SQLiteStore()
    rows = db.fetch_chat_rows()

    state = _load_state(state_path)
    last_chat_id = int(state.get("last_chat_id", 0))
    new_rows = [row for row in rows if row["id"] > last_chat_id]

    if len(new_rows) < min_new_chats:
        print(
            f"Skipped weight update: only {len(new_rows)} new chats (min required: {min_new_chats})"
        )
        return

    build_finetune_dataset_main(
        transcript_jsonl=Path("data/raw/synthetic_university_calls.jsonl"),
        output_dir=Path("data/processed/finetune"),
        max_transcript_samples=4000,
    )

    train_path = Path("data/processed/finetune/train.jsonl")
    val_path = Path("data/processed/finetune/val.jsonl")

    result = train_lora(
        train_path=train_path,
        val_path=val_path,
        output_dir=settings.lora_output_dir,
        base_model=settings.hf_base_model,
        max_steps=settings.weight_update_max_steps,
        learning_rate=settings.weight_update_learning_rate,
        batch_size=settings.weight_update_batch_size,
    )

    prev_best = state.get("best_eval_loss")
    prev_best_float = float(prev_best) if prev_best is not None else None
    if prev_best_float is None:
        loss_ok = True
    else:
        allowed = prev_best_float * (1.0 + settings.weight_gate_max_eval_loss_increase)
        loss_ok = result["eval_loss"] <= allowed

    current_adapter = str(state.get("active_adapter_path", "") or "")
    candidate_summary = _run_summary_for_adapter(result["adapter_path"])
    if current_adapter:
        current_summary = _run_summary_for_adapter(current_adapter)
    else:
        current_summary = {"avg_specificity": 0.0, "avg_refusal_quality": 0.0, "avg_citation_count": 0.0}

    metric_ok, failed_metric_reasons = _passes_metric_gate(candidate_summary)
    better_or_equal = (
        candidate_summary.get("avg_specificity", 0.0) >= current_summary.get("avg_specificity", 0.0)
        and candidate_summary.get("avg_refusal_quality", 0.0)
        >= current_summary.get("avg_refusal_quality", 0.0)
        and candidate_summary.get("avg_citation_count", 0.0)
        >= current_summary.get("avg_citation_count", 0.0)
    )

    improved = loss_ok and metric_ok and better_or_equal
    status = "promoted" if improved else "rejected"
    reject_reasons = []
    if not loss_ok:
        reject_reasons.append("eval_loss_gate_failed")
    if not metric_ok:
        reject_reasons.extend(failed_metric_reasons)
    if not better_or_equal:
        reject_reasons.append("candidate_not_better_or_equal_than_current")

    db.log_model_run(
        run_name=result["run_name"],
        base_model=settings.hf_base_model,
        adapter_path=result["adapter_path"],
        train_rows=result["train_rows"],
        val_rows=result["val_rows"],
        train_loss=result["train_loss"],
        eval_loss=result["eval_loss"],
        status=status,
        notes=json.dumps(
            {
                "prev_best_eval_loss": prev_best,
                "loss_ok": loss_ok,
                "current_summary": current_summary,
                "candidate_summary": candidate_summary,
                "reject_reasons": reject_reasons,
            }
        ),
    )

    max_chat_id = max((row["id"] for row in rows), default=last_chat_id)
    next_state = {
        "last_chat_id": max_chat_id,
        "active_adapter_path": result["adapter_path"] if improved else state.get("active_adapter_path", ""),
        "best_eval_loss": result["eval_loss"] if improved else prev_best,
        "last_run_utc": datetime.now(timezone.utc).isoformat(),
        "new_chat_rows": len(new_rows),
        "last_status": status,
        "last_candidate_summary": candidate_summary,
        "last_current_summary": current_summary,
        "last_reject_reasons": reject_reasons,
    }
    _save_state(state_path, next_state)

    print(
        json.dumps(
            {
                "status": status,
                **result,
                "state_path": str(state_path),
                "loss_ok": loss_ok,
                "candidate_summary": candidate_summary,
                "current_summary": current_summary,
                "reject_reasons": reject_reasons,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-path", type=Path, default=settings.weight_update_state_path)
    parser.add_argument("--min-new-chats", type=int, default=settings.weight_update_min_new_chats)
    args = parser.parse_args()
    main(state_path=args.state_path, min_new_chats=args.min_new_chats)
