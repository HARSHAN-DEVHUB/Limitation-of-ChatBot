from __future__ import annotations

import json
from pathlib import Path

from src.evaluation.baselines import generate_baseline_response
from src.evaluation.metrics import citation_count, refusal_quality, specificity_score
from src.llm.generator import generate_grounded_response


def run_eval_suite(suite_path: Path, mode: str = "improved") -> list[dict]:
    rows = json.loads(suite_path.read_text(encoding="utf-8"))
    results = []

    for row in rows:
        retrieved_chunks = row.get("retrieved_chunks", [])
        if mode == "baseline":
            answer, citations = generate_baseline_response(retrieved_chunks)
        else:
            answer, citations = generate_grounded_response(row["question"], retrieved_chunks)

        results.append(
            {
                "id": row["id"],
                "mode": mode,
                "question": row["question"],
                "answer": answer,
                "citations": citations,
                "specificity": specificity_score(answer, citations),
                "refusal_quality": refusal_quality(answer),
                "citation_count": citation_count(citations),
            }
        )

    return results


def summarize_results(results: list[dict]) -> dict:
    if not results:
        return {
            "count": 0,
            "avg_specificity": 0.0,
            "avg_refusal_quality": 0.0,
            "avg_citation_count": 0.0,
        }

    n = len(results)
    return {
        "count": n,
        "avg_specificity": round(sum(r["specificity"] for r in results) / n, 3),
        "avg_refusal_quality": round(sum(r["refusal_quality"] for r in results) / n, 3),
        "avg_citation_count": round(sum(r["citation_count"] for r in results) / n, 3),
    }
