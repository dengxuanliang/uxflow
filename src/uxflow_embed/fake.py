"""FakeEmbedder — deterministic hash-based vectors, zero ML deps.

For unit tests across modules 1/2/3/0.5: they care about how vectors are
ranked/deduped/recalled, not vector semantics. Lets tests run on CI with
no torch and no GPU.
"""
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 dengxuanliang

from __future__ import annotations

import hashlib

import numpy as np

__all__ = ["FakeEmbedder"]


class FakeEmbedder:
    """Stable L2-normalized vector per text, derived from its hash."""

    def __init__(self, dimension: int = 1024):
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def preferred_batch_size(self) -> int:
        """FakeEmbedder is CPU-bound numpy ops; larger batches amortize loop overhead."""
        return 128

    def embed(self, text: str) -> list[float]:
        # Seed a deterministic RNG from the text hash → stable pseudo-vector.
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        seed = int.from_bytes(digest[:8], "big")
        rng = np.random.default_rng(seed)
        vec = rng.standard_normal(self._dimension)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]
