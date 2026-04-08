# Offline University Support Chatbot (No External AI APIs)

An offline MSc dissertation project that builds a university-support chatbot with:

- no hosted AI APIs
- retrieval-grounded answers with citations
- continuous learning from local chats and transcript corpora
- optional weekly LoRA weight updates with promotion gates

The dissertation focus is to measure chatbot limitations (generic responses, weak refusal behavior, retrieval mistakes, and drift), then show what improves with offline RAG + memory + controlled model updates.

---

## Current Project Status

What is implemented now:

- offline retrieval + response generation
- local storage and searchable memory in SQLite + FAISS
- continuous memory ingestion from chat logs
- synthetic transcript generation and HF transcript import
- finetune dataset builder + local LoRA training
- weekly weight-update workflow with promotion/rejection gate
- evaluation scripts and saved result artifacts

Important current caveat:

- Full, heavy evaluation and live runs may be terminated in low-resource environments (exit code 143) when large model loading is attempted. This is an environment/resource issue, not a missing feature in the code.

---

## Table Of Contents

- [What This Bot Is And Is Not](#what-this-bot-is-and-is-not)
- [Architecture](#architecture)
- [Repository Layout](#repository-layout)
- [Quick Start](#quick-start)
- [Data Ingestion And Indexing](#data-ingestion-and-indexing)
- [Run The Bot](#run-the-bot)
- [Continuous Learning Pipelines](#continuous-learning-pipelines)
- [Evaluation](#evaluation)
- [Backends And Config](#backends-and-config)
- [Artifacts And State Files](#artifacts-and-state-files)
- [Troubleshooting](#troubleshooting)
- [Dissertation Notes](#dissertation-notes)
- [Roadmap](#roadmap)

---

## What This Bot Is And Is Not

### Included

- Offline local operation with no OpenAI/Gemini/Anthropic API calls
- Grounded answers from retrieved evidence
- Citation output for non-refusal answers
- Weekly memory update pipeline
- Weekly optional weight update (LoRA) pipeline

### Not Included

- Real private call recordings (public transcript data is used instead)
- Guaranteed real-time training after each single chat turn
- Cloud-hosted inference

---

## Architecture

```text
Public transcript JSONL + chat logs
            -> preprocess + chunk
            -> store chunks in SQLite
            -> embed chunks
            -> save vectors in FAISS

User question
            -> retrieve relevant chunks
            -> grounded prompt
            -> local backend generation
            -> answer + citations
            -> log chat to SQLite
            -> optional high-signal write-back chunk
```

Core modules:

- ingestion: normalization, cleaning, chunking
- retrieval: semantic + lexical memory support
- generation: backend routing with strict/fallback behavior
- learning: weekly memory and weekly weight update scripts
- evaluation: baseline vs improved scoring

---

## Repository Layout

```text
apps/
   cli_chat.py
data/
   raw/
   processed/
   index/
   eval/
docs/
   transcript-datasets.md
models/
   adapters/
scripts/
   ingest_dataset.py
   build_index.py
   run_eval.py
   export_chatlogs.py
   generate_synthetic_transcripts.py
   import_hf_transcripts.py
   weekly_memory_update.py
   build_finetune_dataset.py
   train_lora.py
   weekly_weight_update.py
   run_full_weekly_pipeline.py
src/
   config.py
   db/
   embedding/
   index/
   learning/
   llm/
   evaluation/
```

---

## Quick Start

### 1) Create and activate venv

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

### 3) Configure environment

```bash
cp .env.example .env
```

Edit `.env` with your preferred backend and model paths.

---

## Data Ingestion And Indexing

### Minimal local dataset flow

1. Place transcript JSONL files in `data/raw/`
2. Run ingestion
3. Build vector index

```bash
python scripts/ingest_dataset.py --input data/raw --output data/processed
python scripts/build_index.py
```

Note: `scripts/build_index.py` has no CLI args and always rebuilds from DB chunks.

### Generate synthetic transcripts (large offline corpus)

```bash
python scripts/generate_synthetic_transcripts.py --conversations 3000
python scripts/ingest_dataset.py --input data/raw --output data/processed
python scripts/build_index.py
```

### Import public transcripts from Hugging Face

```bash
python scripts/import_hf_transcripts.py --dataset daily_dialog --split train --limit 2000 --output data/raw/hf_daily_dialog_train.jsonl
python scripts/ingest_dataset.py --input data/raw --output data/processed
python scripts/build_index.py
```

---

## Run The Bot

```bash
python apps/cli_chat.py
```

The CLI supports:

- greeting/thanks social handling
- retrieval-grounded answers
- refusal when evidence is insufficient
- chat logging
- high-signal continuous memory write-back

---

## Continuous Learning Pipelines

### 1) Weekly memory update (knowledge-level learning)

```bash
python scripts/weekly_memory_update.py --min-new-chats 1
```

This script:

- reads new chat rows from SQLite
- filters high-signal assistant turns
- upserts memory chunks
- rebuilds FAISS index
- updates state in `data/processed/weekly_memory_state.json`

### 2) Build finetune dataset

```bash
python scripts/build_finetune_dataset.py --transcript-jsonl data/raw/synthetic_university_calls.jsonl --output-dir data/processed/finetune --max-transcript-samples 4000
```

### 3) Train LoRA adapter

```bash
python scripts/train_lora.py --train-path data/processed/finetune/train.jsonl --val-path data/processed/finetune/val.jsonl
```

### 4) Weekly weight update (model-level learning)

```bash
python scripts/weekly_weight_update.py --min-new-chats 20
```

This workflow:

- checks minimum new chats
- builds finetune data
- trains a candidate LoRA adapter
- evaluates candidate
- promotes or rejects candidate based on configured gates
- updates state in `data/processed/weekly_weight_state.json`

### 5) Run both weekly jobs together

```bash
python scripts/run_full_weekly_pipeline.py --min-new-chats-memory 1 --min-new-chats-weight 20
```

---

## Evaluation

Run full evaluation (baseline and improved):

```bash
python scripts/run_eval.py --suite data/eval/university_support_questions.json
```

Output files are written under `data/eval/results/`:

- `eval_baseline_*.json`
- `eval_improved_*.json`
- `eval_summary_*.csv`

Metric columns:

- `avg_specificity`
- `avg_refusal_quality`
- `avg_citation_count`

Run single-mode summary helper:

```bash
python scripts/eval_backend_summary.py --suite data/eval/university_support_questions.json --mode improved
```

---

## Backends And Config

Main config is in `src/config.py`, with values loaded from `.env`.

### Backend selection

- `LLM_BACKEND=auto`:
   tries `llama_cpp`, then `ollama`, then `hf_local`, then fallback formatter
- `LLM_BACKEND=llama_cpp`:
   requires `llama-cpp-python` and a valid local GGUF path
- `LLM_BACKEND=ollama`:
   requires running local Ollama service
- `LLM_BACKEND=hf_local`:
   uses HuggingFace model (and optional active adapter)
- `LLM_BACKEND=mock`:
   skips heavy generation and uses deterministic fallback behavior

### Strict backend behavior

- `LLM_STRICT_BACKEND=true` forces an explicit backend-unavailable message.
- `LLM_STRICT_BACKEND=false` allows fallback behavior.

### Important variables

- `LLM_MODEL_PATH`
- `HF_BASE_MODEL`
- `HF_ACTIVE_ADAPTER_PATH`
- `WEIGHT_UPDATE_STATE_PATH`
- `WEIGHT_EVAL_SUITE_PATH`
- `WEIGHT_GATE_MIN_SPECIFICITY`
- `WEIGHT_GATE_MIN_REFUSAL_QUALITY`
- `WEIGHT_GATE_MIN_CITATION_COUNT`
- `WEIGHT_GATE_MAX_EVAL_LOSS_INCREASE`
- `TOP_K`
- `MIN_RETRIEVAL_SCORE`

---

## Artifacts And State Files

Key operational files:

- `data/chatbot.db`: SQLite store for chat logs, chunks, run logs
- `data/index/faiss.index`: vector index
- `data/index/chunks.jsonl`: index metadata
- `data/processed/weekly_memory_state.json`: memory update state
- `data/processed/weekly_weight_state.json`: active adapter + promotion state
- `models/adapters/*`: LoRA training outputs

---

## Troubleshooting

### Process terminated (exit code 143)

Symptoms:

- evaluation or chat run exits with code 143

Typical causes:

- environment resource limits while loading larger models

Actions:

1. Use lighter backend for checks:

```bash
export LLM_BACKEND=mock
python apps/cli_chat.py
```

2. Reduce heavy runs and test one script at a time.
3. If using `hf_local`, use a smaller base model.
4. Run training/eval on a higher-memory GPU machine when possible.

### No chunks found

Run:

```bash
python scripts/ingest_dataset.py --input data/raw --output data/processed
python scripts/build_index.py
```

### Weekly jobs skip with low chat count

This is expected when `--min-new-chats` is not met. Lower the threshold for testing.

---

## Dissertation Notes

Use this wording for methodology clarity:

- “Call records are represented by public transcript corpora in text form.”
- “Continuous learning is implemented at two levels: memory-level KB updates and optional periodic weight-level LoRA updates.”
- “Promotion gates prevent degraded adapters from becoming active.”

Recommended experiment set:

1. Baseline vs improved on fixed suite.
2. Before/after weekly memory update comparison.
3. Promotion/rejection examples from weekly weight update logs.
4. Failure analysis on out-of-domain and ambiguous questions.

---

## Roadmap

- [x] MVP: offline ingestion + retrieval + citations
- [x] Continuous memory updates from chats
- [x] Evaluation artifacts (JSON/CSV)
- [x] LoRA training + weekly weight update orchestration
- [ ] Final dissertation report package (figures/tables/analysis write-up)

---
