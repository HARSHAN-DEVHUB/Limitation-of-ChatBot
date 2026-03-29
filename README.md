# Offline University Support Chatbot (No External AI APIs)
An MSc dissertation-ready project that builds a **self-contained, offline AI chatbot** that:
- **Does not call ChatGPT/Gemini/any hosted LLM APIs**
- Learns from **public “call record” transcripts** (already text) and **user chats**
- Improves over time via a **continuous learning pipeline** (knowledge-base updates + optional fine-tuning)
- Produces **grounded, non-generic answers** by retrieving relevant evidence from its own data (“brain”)

> Dissertation theme: **Finding the limitations of chatbots** (generic responses, hallucinations, lack of domain grounding, memory failures) and quantifying improvements/limitations using an offline RAG + memory system.

---

## Table of Contents
- [Key Idea](#key-idea)
- [Non-goals / Constraints](#non-goals--constraints)
- [System Architecture](#system-architecture)
- [Data & “Call Records”](#data--call-records)
- [Continuous Learning Definition](#continuous-learning-definition)
- [Tech Stack](#tech-stack)
- [Repository Structure](#repository-structure)
- [Pipelines](#pipelines)
  - [1) Ingestion Pipeline](#1-ingestion-pipeline)
  - [2) Retrieval + Generation (Chat)](#2-retrieval--generation-chat)
  - [3) Continuous Learning Loop](#3-continuous-learning-loop)
  - [4) Optional Fine-tuning (LoRA)](#4-optional-fine-tuning-lora)
- [Evaluation Plan (Dissertation)](#evaluation-plan-dissertation)
- [Setup](#setup)
- [Run](#run)
- [Configuration](#configuration)
- [Reproducibility](#reproducibility)
- [Ethics & Privacy](#ethics--privacy)
- [Limitations](#limitations)
- [Roadmap](#roadmap)

---

## Key Idea
Most chatbots answer **generically** because they lack **domain-specific knowledge** and cannot reliably ground answers in evidence.

This project implements an **offline “mini-LLM system”**:

**Local LLM** + **Knowledge Base (public transcripts + chats)** + **Retriever** + **Memory**  
→ grounded answers with citations  
→ “learning” via continuously updated knowledge base  
→ optional periodic fine-tuning for weight updates

---

## Non-goals / Constraints
- ❌ No OpenAI / Gemini / Anthropic APIs
- ❌ No cloud-hosted inference endpoints (the model runs locally on your machine)
- ✅ Public datasets only (no real private call recordings)
- ✅ “Call records” are represented by **public call-style transcript datasets** (already in text form)
- ✅ Final deliverable must include a working chatbot app (CLI or web)

> Note: Development in GitHub Codespaces is fine for coding, but GPU-heavy training may need a local GPU machine.

---

## System Architecture

### High-level
```text
                ┌───────────────────────────────────────────┐
                │         Public Transcript Datasets          │
                └───────────────────────────────────────────┘
                                   │
                                   ▼
┌────────────────────────────────────────────────────────────────┐
│                     Ingestion + Preprocessing                    │
│  - clean text                                                   │
│  - split into chunks                                            │
│  - add metadata (dataset id, conversation id, turn index, etc.)  │
└────────────────────────────────────────────────────────────────┘
                                   │
                     embeddings    │
                (local model)      ▼
                          ┌─────────────────────────┐
                          │   Vector Index (FAISS)  │
                          └─────────────────────────┘
                                   ▲
                                   │
                          ┌─────────────────────────┐
                          │ SQLite (metadata + logs)│
                          └─────────────────────────┘
                                   ▲
                                   │
┌────────────────────────────────────────────────────────────────┐
│                           Chat Application                       │
│  User message → Retrieve relevant chunks → Build prompt → LLM     │
│  Response + citations → Save chat → Add to KB (continuous update) │
└────────────────────────────────────────────────────────────────┘
```

### Components
1. **Local LLM (Generator)**  
   Generates the final response. Runs offline on CPU/GPU.

2. **Embedding Model**  
   Turns chunks of text into vectors for similarity search.

3. **Vector Index (FAISS)**  
   Fast retrieval of relevant chunks.

4. **SQLite “Brain Store”**  
   Stores:
   - documents and chunks metadata
   - chat logs (user + assistant turns)
   - ingestion runs + evaluation runs

5. **Continuous Learning Loop**
   - Chat logs are periodically re-ingested as additional knowledge
   - Index gets updated, enabling improved future answers

---

## Data & “Call Records”
Because real call recordings are unavailable, this project uses **public conversation transcript datasets** as a stand-in for “call records”.

### Data requirements
Each dataset should provide:
- conversational turns (`speaker`, `text`)
- conversation/session id
- domain or intent labels (optional)

### Example dataset categories (choose 1–2)
- customer-support / helpdesk dialogue corpora
- spoken dialogue system corpora (transcribed)
- task-oriented dialogue corpora (booking, admin tasks, troubleshooting)

> The dissertation should explicitly state: “call records are simulated using public corpora.”

---

## Continuous Learning Definition
“Continuous learning” can mean two different things:

### Level 1 (Required): Continuous Knowledge Learning (RAG)
- Every new chat is saved.
- Useful information is extracted and indexed.
- Future responses improve because the retriever finds newly-added evidence.

✅ Works with limited compute  
✅ Safe + explainable (citations)  
✅ Easy to evaluate over time

### Level 2 (Optional): Periodic Weight Updates (LoRA)
- After N chats or weekly, run LoRA fine-tuning on curated Q/A pairs.
- Evaluate before/after on a fixed test set.

✅ Demonstrates “model updates”  
⚠ Requires careful filtering to avoid degradation

---

## Tech Stack
**Language:** Python 3.11+

**Local LLM runtime (pick one):**
- `llama.cpp` (recommended for portability)
- or Ollama (local service; still offline)

**Embeddings:**
- `sentence-transformers` (local CPU)
- optional GPU acceleration on RTX 4060

**Vector store:**
- FAISS (local index)

**DB:**
- SQLite (simple, reproducible)

**App UI:**
- CLI (fastest)
- or FastAPI + minimal web UI

---

## Repository Structure
```text
.
├── apps/
│   ├── cli_chat.py
│   └── web/                      # optional FastAPI app
├── data/
│   ├── raw/                      # downloaded public datasets (not committed)
│   ├── processed/                # cleaned + normalized JSONL (not committed)
│   └── eval/                     # evaluation question sets (committed)
├── docs/
│   ├── architecture.md
│   ├── dissertation-metrics.md
│   └── dataset-cards.md
├── src/
│   ├── config.py
│   ├── db/
│   │   ├── schema.sql
│   │   └── sqlite.py
│   ├── ingestion/
│   │   ├── loaders.py
│   │   ├── preprocess.py
│   │   └── chunking.py
│   ├── embedding/
│   │   └── embedder.py
│   ├── index/
│   │   ├── faiss_index.py
│   │   └── retrieval.py
│   ├── llm/
│   │   ├── prompt.py
│   │   └── generator.py
│   ├── memory/
│   │   ├── short_term.py
│   │   └── long_term.py
│   ├── learning/
│   │   ├── chatlog_to_kb.py
│   │   └── filters.py
│   └── evaluation/
│       ├── metrics.py
│       └── runner.py
├── scripts/
│   ├── ingest_dataset.py
│   ├── build_index.py
│   ├── run_eval.py
│   └── export_chatlogs.py
├── .env.example
├── requirements.txt
└── README.md
```

---

## Pipelines

### 1) Ingestion Pipeline
**Goal:** Convert transcripts into clean, searchable knowledge chunks.

Steps:
1. Load dataset → normalize to JSONL:
   ```json
   {"doc_id":"...", "conv_id":"...", "turn_id":12, "speaker":"agent", "text":"..."}
   ```
2. Clean:
   - remove timestamps/noise markers
   - normalize whitespace
   - drop extremely short/empty turns
3. Chunk:
   - join turns into windows (e.g., 6–12 turns) or chunk by token length
4. Embed chunks (local embedding model)
5. Store:
   - SQLite: chunk metadata + text
   - FAISS: vectors keyed by `chunk_id`

### 2) Retrieval + Generation (Chat)
**Goal:** Ground the LLM response using retrieved evidence.

Steps:
1. User message → embed query
2. FAISS search → top-k chunks
3. Build prompt:
   - system rules:
     - answer using only provided evidence when possible
     - if evidence is missing: say “I don’t know”
     - provide citations to chunk ids
4. LLM generates answer
5. Return:
   - response text
   - citations: `[chunk_123, chunk_981]`
6. Save conversation to SQLite

### 3) Continuous Learning Loop
**Goal:** The chatbot improves as it chats.

After each session (or nightly job):
1. Export new chat logs
2. Filter:
   - remove low-quality turns (empty, abusive, purely social)
   - keep high-signal Q/A pairs and factual statements
3. Transform to KB chunks
4. Embed + index them
5. Now future retrieval includes user-derived knowledge

> This mimics “learning from user chats” without risky always-on fine-tuning.

### 4) Optional Fine-tuning (LoRA)
**Goal:** Show weight-updating learning (optional).

Workflow:
1. Curate a training set from chat logs + transcripts
2. Run LoRA fine-tuning on a small open model using RTX 4060
3. Evaluate on fixed test set (before/after)
4. If quality drops, revert and report findings (important limitation!)

---

## Evaluation Plan (Dissertation)
Design an evaluation that measures “generic vs specific” and “hallucination vs grounded”.

### Baselines
- **Baseline A:** LLM only (no retrieval)
- **Baseline B:** LLM + retrieval (RAG)
- **System C:** RAG + long-term memory (chatlog ingestion)

### Example metrics
1. **Specificity score (rubric-based)**
   - 0: generic advice
   - 1: partially specific, vague steps
   - 2: concrete steps grounded in evidence + citations

2. **Hallucination rate**
   - % of answers that contain claims not supported by retrieved evidence

3. **Answer correctness**
   - human annotation on a test set (20–100 questions)

4. **“I don’t know” quality**
   - does the bot refuse when evidence is absent?

5. **Latency**
   - compare MacBook Air vs RTX 4060 machine

### Suggested experiments
- Evaluate with the same question set:
  - before any chat learning
  - after ingesting 50/100/200 chats
- Evaluate what happens when transcripts contain conflicting info

---

## Setup

### 1) Create venv
```bash
python -m venv .venv
source .venv/bin/activate
```

### 2) Install dependencies
```bash
pip install -r requirements.txt
```

### 3) Configure environment
Copy `.env.example` to `.env` and set:
- paths to your local LLM model
- embedding model name
- data directories

---

## Run

### Ingest datasets
```bash
python scripts/ingest_dataset.py --input data/raw --output data/processed
python scripts/build_index.py --input data/processed
```

### Start CLI chatbot
```bash
python apps/cli_chat.py
```

### Run evaluation
```bash
python scripts/run_eval.py --suite data/eval/university_support_questions.json
```

---

## Configuration
Common config options:
- `LLM_MODEL_PATH`: path to local GGUF model
- `LLM_CTX_SIZE`: context window (e.g., 4096)
- `TOP_K`: number of retrieved chunks (e.g., 5–10)
- `CHUNK_SIZE`: tokens/characters per chunk
- `INDEX_PATH`: FAISS index path
- `SQLITE_PATH`: DB path

---

## Reproducibility
- All datasets must be referenced with:
  - name, version, source link (in `docs/dataset-cards.md`)
  - license
- Fix random seeds where possible.
- Save:
  - ingestion run parameters
  - model versions
  - evaluation results in timestamped files (CSV/JSON)

---

## Ethics & Privacy
- Use only public datasets with clear licenses.
- If collecting user chats during testing, add a consent notice:
  - what is stored
  - how it is used (learning/indexing)
  - how to delete it

---

## Limitations
This system is designed to highlight real chatbot limitations:
- Domain mismatch → generic outputs
- Retrieval failure → wrong grounding
- Conflicting evidence → inconsistent answers
- Continuous ingestion can reinforce mistakes if filtering is weak
- Fine-tuning on noisy chat logs can cause catastrophic forgetting

---

## Roadmap
- [ ] MVP: Offline LLM + ingestion + FAISS retrieval + citations
- [ ] Continuous learning: ingest chat logs into KB
- [ ] Evaluation runner + metrics + plots
- [ ] (Optional) LoRA fine-tuning experiment
- [ ] Final report: architecture + experiments + limitations + discussion

---
