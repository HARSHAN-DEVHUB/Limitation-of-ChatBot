from __future__ import annotations


class ShortTermMemory:
    def __init__(self, max_turns: int = 8) -> None:
        self.max_turns = max_turns
        self.turns: list[dict] = []

    def add(self, role: str, text: str) -> None:
        self.turns.append({"role": role, "text": text})
        if len(self.turns) > self.max_turns:
            self.turns = self.turns[-self.max_turns :]

    def get_context_text(self) -> str:
        return "\n".join(f"{t['role']}: {t['text']}" for t in self.turns)
