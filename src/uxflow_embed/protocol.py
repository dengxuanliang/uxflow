"""Embedder protocol — the single embedding seam for all UXFlow modules.

Contract §6: all embedding call sites MUST use the same model + dimension.
Injecting one Embedder instance makes that a structural guarantee.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

__all__ = ["Embedder"]


@runtime_checkable
class Embedder(Protocol):
    """Produces L2-normalized vectors. Implementations: Fake / Local / Api."""

    @property
    def dimension(self) -> int: ...

    def embed(self, text: str) -> list[float]: ...

    def embed_batch(self, texts: list[str]) -> list[list[float]]: ...
