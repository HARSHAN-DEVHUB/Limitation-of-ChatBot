from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from src.config import settings
from src.db.sqlite import SQLiteStore


def _select_lora_target_modules(model) -> list[str]:
    preferred = ["q_proj", "v_proj", "k_proj", "o_proj", "c_attn", "c_proj"]
    names = set()
    for module_name, _ in model.named_modules():
        leaf = module_name.split(".")[-1]
        if leaf in preferred:
            names.add(leaf)

    ordered = [name for name in preferred if name in names]
    if ordered:
        return ordered[:4]

    # Fallback: target common projection-like layers by suffix.
    fallback = []
    for module_name, _ in model.named_modules():
        leaf = module_name.split(".")[-1]
        if any(key in leaf for key in ["proj", "attn", "fc"]):
            fallback.append(leaf)
    uniq = []
    seen = set()
    for name in fallback:
        if name in seen:
            continue
        seen.add(name)
        uniq.append(name)
        if len(uniq) >= 4:
            break
    return uniq or ["c_attn"]


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _to_text_rows(rows: list[dict]) -> list[dict]:
    out = []
    for row in rows:
        prompt = row.get("prompt", "")
        response = row.get("response", "")
        full_text = f"{prompt} {response}".strip()
        if full_text:
            out.append({"text": full_text})
    return out


def train_lora(
    train_path: Path,
    val_path: Path,
    output_dir: Path,
    base_model: str,
    max_steps: int,
    learning_rate: float,
    batch_size: int,
) -> dict:
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(base_model)

    target_modules = _select_lora_target_modules(model)
    lora_cfg = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=target_modules,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)

    train_rows = _to_text_rows(_read_jsonl(train_path))
    val_rows = _to_text_rows(_read_jsonl(val_path))
    train_ds = Dataset.from_list(train_rows)
    val_ds = Dataset.from_list(val_rows)

    def tokenize_fn(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=512,
            padding="max_length",
        )

    train_ds = train_ds.map(tokenize_fn, batched=True, remove_columns=["text"])
    val_ds = val_ds.map(tokenize_fn, batched=True, remove_columns=["text"])

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    run_ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_output = output_dir / f"lora_run_{run_ts}"
    run_output.mkdir(parents=True, exist_ok=True)

    args = TrainingArguments(
        output_dir=str(run_output),
        overwrite_output_dir=True,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        max_steps=max_steps,
        eval_strategy="steps",
        eval_steps=max(10, max_steps // 4),
        save_steps=max(10, max_steps // 4),
        logging_steps=max(5, max_steps // 10),
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
    )

    train_result = trainer.train()
    eval_result = trainer.evaluate()
    trainer.save_model(str(run_output))
    tokenizer.save_pretrained(str(run_output))

    return {
        "run_name": run_output.name,
        "adapter_path": str(run_output),
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "train_loss": float(train_result.training_loss),
        "eval_loss": float(eval_result.get("eval_loss", 0.0)),
    }


def main(
    train_path: Path,
    val_path: Path,
    output_dir: Path,
    base_model: str,
    max_steps: int,
    learning_rate: float,
    batch_size: int,
) -> None:
    result = train_lora(
        train_path=train_path,
        val_path=val_path,
        output_dir=output_dir,
        base_model=base_model,
        max_steps=max_steps,
        learning_rate=learning_rate,
        batch_size=batch_size,
    )

    SQLiteStore().log_model_run(
        run_name=result["run_name"],
        base_model=base_model,
        adapter_path=result["adapter_path"],
        train_rows=result["train_rows"],
        val_rows=result["val_rows"],
        train_loss=result["train_loss"],
        eval_loss=result["eval_loss"],
        status="trained",
        notes="manual_train_lora",
    )

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-path", type=Path, default=Path("data/processed/finetune/train.jsonl"))
    parser.add_argument("--val-path", type=Path, default=Path("data/processed/finetune/val.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=settings.lora_output_dir)
    parser.add_argument("--base-model", type=str, default=settings.hf_base_model)
    parser.add_argument("--max-steps", type=int, default=settings.weight_update_max_steps)
    parser.add_argument("--learning-rate", type=float, default=settings.weight_update_learning_rate)
    parser.add_argument("--batch-size", type=int, default=settings.weight_update_batch_size)
    args = parser.parse_args()
    main(
        train_path=args.train_path,
        val_path=args.val_path,
        output_dir=args.output_dir,
        base_model=args.base_model,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
    )
