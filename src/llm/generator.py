from __future__ import annotations

import json
from pathlib import Path
from urllib import error, request

from src.config import settings
from src.llm.prompt import build_grounded_prompt


_llama_cpp_model = None
_ollama_unavailable = False


def _extract_citations(retrieved_chunks: list[dict]) -> list[str]:
    return [chunk["chunk_id"] for chunk in retrieved_chunks]


def _is_refusal(text: str) -> bool:
    low = text.lower()
    return "do not know" in low or "not enough evidence" in low


def _build_llama_cpp_client():
    global _llama_cpp_model
    if _llama_cpp_model is not None:
        return _llama_cpp_model

    model_path = Path(settings.llm_model_path)
    if not model_path.exists():
        return None

    try:
        from llama_cpp import Llama  # type: ignore
    except Exception:
        return None

    _llama_cpp_model = Llama(
        model_path=str(model_path),
        n_ctx=settings.llm_ctx_size,
        verbose=False,
    )
    return _llama_cpp_model


def _generate_with_llama_cpp(user_message: str, retrieved_chunks: list[dict]) -> str | None:
    client = _build_llama_cpp_client()
    if client is None:
        return None

    prompt = build_grounded_prompt(user_message, retrieved_chunks)
    out = client(
        prompt,
        max_tokens=settings.llm_max_tokens,
        temperature=settings.llm_temperature,
        stop=["\n\nUser:", "\nYou:"],
    )
    text = out["choices"][0]["text"].strip()
    return text or None


def _generate_with_ollama(user_message: str, retrieved_chunks: list[dict]) -> str | None:
    global _ollama_unavailable
    if _ollama_unavailable:
        return None

    prompt = build_grounded_prompt(user_message, retrieved_chunks)
    payload = {
        "model": settings.ollama_model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": settings.llm_temperature,
            "num_predict": settings.llm_max_tokens,
        },
    }
    body = json.dumps(payload).encode("utf-8")
    url = f"{settings.ollama_host.rstrip('/')}/api/generate"
    req = request.Request(url, data=body, headers={"Content-Type": "application/json"})

    try:
        with request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except (error.URLError, TimeoutError, json.JSONDecodeError):
        _ollama_unavailable = True
        return None

    text = str(data.get("response", "")).strip()
    return text or None


def _extract_agent_actions(retrieved_chunks: list[dict]) -> list[str]:
    actions: list[str] = []
    seen: set[str] = set()
    for chunk in retrieved_chunks:
        for line in chunk["text"].splitlines():
            line = line.strip()
            if not line.lower().startswith("agent:"):
                continue
            action = line.split(":", 1)[1].strip()
            key = action.lower()
            if action and key not in seen:
                seen.add(key)
                actions.append(action)
    return actions


def _format_step_response(actions: list[str]) -> str:
    if not actions:
        return "I do not have enough evidence in my knowledge base to answer that yet."

    step_lines = [f"{idx}. {action}" for idx, action in enumerate(actions, start=1)]
    return (
        "Based on the available records, follow these steps:\n\n"
        + "\n".join(step_lines)
        + "\n\n"
        + "If your case is different, tell me your exact situation and I will refine the steps."
    )


def _generate_fallback(retrieved_chunks: list[dict]) -> str:
    actions = _extract_agent_actions(retrieved_chunks)
    return _format_step_response(actions)


def _generate_with_backend(user_message: str, retrieved_chunks: list[dict]) -> str | None:
    backend = settings.llm_backend.lower().strip()

    if backend == "llama_cpp":
        return _generate_with_llama_cpp(user_message, retrieved_chunks)
    if backend == "ollama":
        return _generate_with_ollama(user_message, retrieved_chunks)
    if backend == "mock":
        return None

    # auto mode: try llama.cpp first, then Ollama.
    text = _generate_with_llama_cpp(user_message, retrieved_chunks)
    if text:
        return text
    return _generate_with_ollama(user_message, retrieved_chunks)


def generate_grounded_response(user_message: str, retrieved_chunks: list[dict]) -> tuple[str, list[str]]:
    if not retrieved_chunks:
        return (
            "I do not have enough evidence in my knowledge base to answer that yet.",
            [],
        )

    citations = _extract_citations(retrieved_chunks)
    llm_response = _generate_with_backend(user_message, retrieved_chunks)
    response = llm_response if llm_response else _generate_fallback(retrieved_chunks)

    if _is_refusal(response):
        return response, []

    return response, citations
