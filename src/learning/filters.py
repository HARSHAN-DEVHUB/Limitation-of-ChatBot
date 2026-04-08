from __future__ import annotations


BLOCKLIST = {"hi", "hello", "ok", "thanks", "thank you"}


def is_high_signal_turn(text: str) -> bool:
    text_norm = text.strip().lower()
    if not text_norm:
        return False
    if text_norm in BLOCKLIST:
        return False
    return len(text_norm) > 25
