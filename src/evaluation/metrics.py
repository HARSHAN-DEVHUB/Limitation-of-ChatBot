from __future__ import annotations


def specificity_score(answer: str, citations: list[str]) -> int:
    answer_low = answer.lower()
    has_numbered_steps = "1." in answer and "2." in answer
    has_guidance_words = any(
        token in answer_low for token in ["follow", "steps", "upload", "contact", "verify"]
    )
    has_citations = len(citations) > 0

    if has_numbered_steps and has_guidance_words and has_citations:
        return 2
    if has_guidance_words and has_citations:
        return 1
    return 0


def refusal_quality(answer: str) -> int:
    answer_low = answer.lower()
    if "do not have enough evidence" in answer_low or "i don't know" in answer_low:
        return 1
    return 0


def citation_count(citations: list[str]) -> int:
    return len(citations)
