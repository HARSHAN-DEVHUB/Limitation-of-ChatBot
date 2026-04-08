from __future__ import annotations

import json
from pathlib import Path
from urllib import error, request

from src.config import settings
from src.llm.prompt import build_grounded_prompt


_llama_cpp_model = None
_ollama_unavailable = False
_hf_generator = None


def _extract_citations(retrieved_chunks: list[dict]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for chunk in retrieved_chunks:
        cid = chunk["chunk_id"]
        if cid in seen:
            continue
        seen.add(cid)
        out.append(cid)
        if len(out) >= 3:
            break
    return out


def _is_refusal(text: str) -> bool:
    low = text.lower()
    return (
        "do not know" in low
        or "not enough evidence" in low
        or "do not have enough evidence" in low
        or "i don't know" in low
    )


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


def _get_active_adapter_path() -> str:
    if settings.hf_active_adapter_path:
        return settings.hf_active_adapter_path

    state_path = settings.weight_update_state_path
    if not state_path.exists():
        return ""

    try:
        data = json.loads(state_path.read_text(encoding="utf-8"))
    except Exception:
        return ""
    return str(data.get("active_adapter_path", "") or "")


def _build_hf_generator():
    global _hf_generator
    if _hf_generator is not None:
        return _hf_generator

    try:
        import torch  # type: ignore
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline  # type: ignore
    except Exception:
        return None

    base_model = settings.hf_base_model
    try:
        model = AutoModelForCausalLM.from_pretrained(base_model)
        tokenizer = AutoTokenizer.from_pretrained(base_model)
    except Exception:
        return None

    adapter_path = _get_active_adapter_path()
    if adapter_path:
        try:
            from peft import PeftModel  # type: ignore

            model = PeftModel.from_pretrained(model, adapter_path)
        except Exception:
            pass

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = 0 if torch.cuda.is_available() else -1
    _hf_generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=device)
    return _hf_generator


def _generate_with_hf_local(user_message: str, retrieved_chunks: list[dict]) -> str | None:
    generator = _build_hf_generator()
    if generator is None:
        return None

    prompt = build_grounded_prompt(user_message, retrieved_chunks)
    try:
        out = generator(
            prompt,
            max_new_tokens=settings.llm_max_tokens,
            temperature=settings.llm_temperature,
            do_sample=settings.llm_temperature > 0,
        )
    except Exception:
        return None

    if not out:
        return None
    generated = str(out[0].get("generated_text", "")).strip()
    if generated.startswith(prompt):
        generated = generated[len(prompt) :].strip()
    return generated or None


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
    if backend == "hf_local":
        return _generate_with_hf_local(user_message, retrieved_chunks)
    if backend == "mock":
        return None

    # auto mode: try llama.cpp, then Ollama, then local HF.
    text = _generate_with_llama_cpp(user_message, retrieved_chunks)
    if text:
        return text
    text = _generate_with_ollama(user_message, retrieved_chunks)
    if text:
        return text
    return _generate_with_hf_local(user_message, retrieved_chunks)


def generate_grounded_response(user_message: str, retrieved_chunks: list[dict]) -> tuple[str, list[str]]:
    if not retrieved_chunks:
        return (
            "I do not have enough evidence in my knowledge base to answer that yet.",
            [],
        )

    citations = _extract_citations(retrieved_chunks)
    llm_response = _generate_with_backend(user_message, retrieved_chunks)

    if llm_response is None and settings.llm_strict_backend and settings.llm_backend != "mock":
        return (
            f"Configured backend '{settings.llm_backend}' is unavailable. "
            "Please start the backend service or switch backend mode.",
            [],
        )

    response = llm_response if llm_response else _generate_fallback(retrieved_chunks)

    if _is_refusal(response):
        return response, []

    return response, citations
