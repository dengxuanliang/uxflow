"""LocalEmbedder — Qwen3-Embedding via sentence-transformers (optional dep).

Produces 1024-d L2-normalized vectors on MPS/CUDA/CPU. Model auto-downloads
from HuggingFace on first use (local_files_only defaults to False; pass
True for air-gapped/offline use). Requires the
[local-embed] extra; heavy imports are lazy so `import uxflow_embed` works
without torch installed.

Contract §6: same model + dimension across all embedding call sites.
"""
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 dengxuanliang

from __future__ import annotations

import numpy as np

__all__ = ["LocalEmbedder"]

_MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
_DIMENSION = 1024


class LocalEmbedder:
    """Wrapper for local Qwen3-Embedding inference."""

    def __init__(self, model_name: str = _MODEL_NAME, device: str | None = None, local_files_only: bool = False):
        try:
            from sentence_transformers import SentenceTransformer
            import torch
        except ImportError as e:
            raise ImportError(
                "LocalEmbedder requires the optional 'local-embed' dependencies. "
                'Install with: pip install -e ".[local-embed]"'
            ) from e

        if device is None:
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"

        # Default local_files_only=False auto-downloads so cloners run out of
        # the box; pass True for air-gapped/offline use.
        self._model = SentenceTransformer(model_name, device=device, local_files_only=local_files_only)
        self._dimension = _DIMENSION

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, text: str) -> list[float]:
        vec = self._model.encode(
            text, normalize_embeddings=True, output_value="sentence_embedding"
        )
        return self._ensure_dimension(vec).tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
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
        """Truncate or zero-pad to dimension, then re-normalize (MRL)."""
        if len(vec) >= self._dimension:
            vec = vec[: self._dimension]
        else:
            vec = np.pad(vec, (0, self._dimension - len(vec)))
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec
