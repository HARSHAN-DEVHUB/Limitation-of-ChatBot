from __future__ import annotations


def build_grounded_prompt(user_message: str, retrieved_chunks: list[dict]) -> str:
    evidence = "\n\n".join(
        f"[{chunk['chunk_id']}]\n{chunk['text']}" for chunk in retrieved_chunks
    )
    return (
        "You are an offline university support assistant.\n"
        "Rules:\n"
        "1) Use only the provided evidence.\n"
        "2) If evidence is insufficient, reply: 'I do not know based on current evidence.'\n"
        "3) Give concise, practical steps.\n"
        "4) Do not invent policies, deadlines, or links.\n\n"
        f"User message:\n{user_message}\n\n"
        f"Evidence:\n{evidence}"
    )
