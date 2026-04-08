from __future__ import annotations

import numpy as np

from src.config import settings
from src.embedding.embedder import LocalEmbedder
from src.index.faiss_index import FaissStore


class Retriever:
    def __init__(self, top_k: int | None = None) -> None:
        self.top_k = top_k or settings.top_k
        self.embedder = LocalEmbedder()
        self.index, self.chunks = FaissStore().load()

    def search(self, query: str) -> list[dict]:
        qv = self.embedder.encode([query])
        scores, indices = self.index.search(qv.astype(np.float32), self.top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.chunks):
                continue
            chunk = self.chunks[idx]
            results.append(
                {
                    "score": float(score),
                    "chunk_id": chunk["chunk_id"],
                    "text": chunk["text"],
                    "metadata": chunk.get("metadata", {}),
                }
            )
        return results
