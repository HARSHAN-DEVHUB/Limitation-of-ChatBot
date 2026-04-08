from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import settings


class LocalEmbedder:
    def __init__(self, model_name: str | None = None) -> None:
        self.model = SentenceTransformer(model_name or settings.embedding_model_name)

    def encode(self, texts: list[str]) -> np.ndarray:
        vectors = self.model.encode(texts, normalize_embeddings=True)
        return np.asarray(vectors, dtype="float32")
