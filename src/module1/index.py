"""Phase 2+3: Memory-based index with multi-path recall.

V1 strategy: in-memory list + numpy. No ES/Qdrant dependency.
Scales to ~10k signatures. For millions, swap to ES+Qdrant (same interface).

NOTE: When migrating to a persistent index, query and stored vectors are no
longer guaranteed to be same-source. Validate embedding dimension at
add/load time (fail-loud on cross-dim mixing) and skip mismatched query
vectors in _vector_score. Not needed for V1: vectors are computed on the fly
from a single EmbeddingModel, so query and index dimensions always match.

Recall pipeline:
  1. Structured filters (languages, tools_used, has_verification_step)
  2. BM25 scoring (TF-IDF on bm25_tokens)
  3. Vector cosine similarity (numpy)
  4. RRF fusion → top-N
"""
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 dengxuanliang

from __future__ import annotations

from module1 import recall_core
from module1.models import Slice, TrajectorySignature
from module1.store import RecallHit

__all__ = ["MemoryIndex"]


class MemoryIndex:
    """In-memory index for trajectory signatures with multi-path recall."""

    def __init__(self):
        self._signatures: list[TrajectorySignature] = []
        self._slice_source: dict[tuple[str, int], Slice] = {}

    @property
    def size(self) -> int:
        return len(self._signatures)

    def add(self, sig: TrajectorySignature) -> None:
        self._signatures.append(sig)

    def add_batch(self, sigs: list[TrajectorySignature]) -> None:
        self._signatures.extend(sigs)

    def recall(
        self,
        *,
        structured_filters: dict,
        keywords: list[str],
        query_embeddings: list[list[float]],
        top_n: int = 20,
    ) -> list[RecallHit]:
        """Multi-path recall: filter → BM25 + vector → RRF → top-N.

        Args:
            structured_filters: dict with optional keys: languages, tools_used,
                has_verification_step. None values are skipped.
            keywords: BM25 query terms.
            query_embeddings: list of query vectors (one per hyde_positive segment).
                Cosine similarity uses max across segments.
            top_n: max results to return.

        Returns:
            Ranked list of RecallHit, best first.
        """
        return recall_core.recall(
            self._signatures,
            structured_filters=structured_filters,
            keywords=keywords,
            query_embeddings=query_embeddings,
            top_n=top_n,
        )

    def _get_signature(
        self, trajectory_id: str, slice_index: int
    ) -> TrajectorySignature | None:
        for sig in self._signatures:
            if sig.trajectory_id == trajectory_id and sig.slice_index == slice_index:
                return sig
        return None

    def update_labels(
        self, trajectory_id: str, slice_index: int, labels: list[str]
    ) -> None:
        """Attach capability labels on the matching signature."""
        sig = self._get_signature(trajectory_id, slice_index)
        if sig is not None:
            existing = sig.capability_labels or []
            sig.capability_labels = list(dict.fromkeys([*existing, *labels]))

    def set_slice_source(
        self, trajectory_id: str, slice_index: int, slice_obj: Slice
    ) -> None:
        self._slice_source[(trajectory_id, slice_index)] = slice_obj

    def get_slice(self, trajectory_id: str, slice_index: int) -> Slice | None:
        """Return the source slice object for (trajectory_id, slice_index)."""
        return self._slice_source.get((trajectory_id, slice_index))

    def get_slice_obj(self, trajectory_id: str, slice_index: int) -> Slice | None:
        return self.get_slice(trajectory_id, slice_index)

    @property
    def slice_source_count(self) -> int:
        return len(self._slice_source)

    def _apply_filters(self, filters: dict) -> list[TrajectorySignature]:
        """Thin wrapper over recall_core.apply_filters (kept for e2e introspection)."""
        return recall_core.apply_filters(self._signatures, filters)

    def _bm25_score(
        self, candidates: list[TrajectorySignature], keywords: list[str]
    ) -> dict[int, float]:
        """Thin wrapper over recall_core.bm25_score (kept for e2e introspection)."""
        return recall_core.bm25_score(self._signatures, candidates, keywords)

    def _vector_score(
        self, candidates: list[TrajectorySignature], query_embeddings: list[list[float]]
    ) -> dict[int, float]:
        """Thin wrapper over recall_core.vector_score (kept for e2e introspection)."""
        return recall_core.vector_score(candidates, query_embeddings)
