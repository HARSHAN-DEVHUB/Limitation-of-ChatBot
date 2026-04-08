from __future__ import annotations


def generate_baseline_response(retrieved_chunks: list[dict]) -> tuple[str, list[str]]:
    if not retrieved_chunks:
        return "I do not know.", []

    top = retrieved_chunks[0]
    citations = [top["chunk_id"]]
    response = (
        "Here is the most relevant information I found:\n\n"
        f"{top['text']}\n\n"
        "This may not exactly match your case."
    )
    return response, citations
