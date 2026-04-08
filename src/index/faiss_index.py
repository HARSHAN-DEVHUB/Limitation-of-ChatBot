from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np

from src.config import settings


class FaissStore:
    def __init__(self, index_path: Path | None = None, meta_path: Path | None = None) -> None:
        self.index_path = index_path or settings.index_path
        self.meta_path = meta_path or settings.index_meta_path

    def save(self, vectors: np.ndarray, chunks: list[dict]) -> None:
        if len(vectors) == 0:
            raise ValueError("No vectors to index")

        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        faiss.write_index(index, str(self.index_path))

        with self.meta_path.open("w", encoding="utf-8") as handle:
            for chunk in chunks:
                handle.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    def load(self) -> tuple[faiss.Index, list[dict]]:
        if not self.index_path.exists() or not self.meta_path.exists():
            raise FileNotFoundError(
                f"Missing index files: {self.index_path} and/or {self.meta_path}"
            )

        index = faiss.read_index(str(self.index_path))
        chunks = []
        with self.meta_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    chunks.append(json.loads(line))
        return index, chunks
