"""Local embedding via Qwen3-Embedding (sentence-transformers).

Produces 1536-d L2-normalized vectors. Runs on MPS (Apple Silicon) or CPU.
Model loaded lazily on first call to avoid import-time overhead.

Contract §4: both sides MUST use the same model + same dimension.
"""

from __future__ import annotations

import numpy as np

__all__ = ["EmbeddingModel"]

# Model identifier — pinned per contract §4/§6
_MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
_DIMENSION = 1536


class EmbeddingModel:
    """Wrapper for local Qwen3-Embedding inference."""

    def __init__(self, model_name: str = _MODEL_NAME, device: str | None = None):
        from sentence_transformers import SentenceTransformer
        import torch

        if device is None:
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"

        self._model = SentenceTransformer(model_name, device=device)
        self._dimension = _DIMENSION

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, text: str) -> list[float]:
        """Embed a single text. Returns L2-normalized 1536-d vector."""
        vec = self._model.encode(
            text,
            normalize_embeddings=True,
            output_value="sentence_embedding",
        )
        vec = self._ensure_dimension(vec)
        return vec.tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts. Returns list of L2-normalized 1536-d vectors."""
        if not texts:
            return []
        vecs = self._model.encode(
            texts,
            normalize_embeddings=True,
            output_value="sentence_embedding",
            batch_size=32,
        )
        return [self._ensure_dimension(v).tolist() for v in vecs]

    def _ensure_dimension(self, vec: np.ndarray) -> np.ndarray:
        """Truncate or zero-pad to self._dimension, then re-normalize (MRL)."""
        if len(vec) >= self._dimension:
            vec = vec[:self._dimension]
        else:
            vec = np.pad(vec, (0, self._dimension - len(vec)))
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec
