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

import math
from collections import Counter

import numpy as np

from module1.models import Slice, TrajectorySignature
from module1.store import RecallHit

__all__ = ["MemoryIndex"]

_RRF_K = 60  # RRF constant (standard value)


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
        if not self._signatures:
            return []

        # Phase 1: structured filtering
        candidates = self._apply_filters(structured_filters)
        if not candidates:
            return []

        # Phase 2: scoring
        bm25_scores = self._bm25_score(candidates, keywords) if keywords else {}
        vector_scores = self._vector_score(candidates, query_embeddings) if query_embeddings else {}

        # Phase 3: RRF fusion
        fused = self._rrf_fuse(candidates, bm25_scores, vector_scores)

        # Sort by fused score descending
        fused.sort(key=lambda x: x[1], reverse=True)

        return [
            RecallHit(signature=sig, rrf_score=score)
            for sig, score in fused[:top_n]
        ]

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
        """Apply structured filters. None/missing values skip that filter."""
        result = []
        languages = filters.get("languages")
        tools_used = filters.get("tools_used")
        has_verify = filters.get("has_verification_step")

        for sig in self._signatures:
            # languages: intersection non-empty (OR)
            if languages is not None:
                if not set(sig.languages) & set(languages):
                    continue
            # tools_used: intersection non-empty (OR)
            if tools_used is not None:
                if not set(sig.tools_used) & set(tools_used):
                    continue
            # has_verification_step: exact match
            if has_verify is not None:
                if sig.has_verification_step != has_verify:
                    continue
            result.append(sig)
        return result

    def _bm25_score(
        self, candidates: list[TrajectorySignature], keywords: list[str]
    ) -> dict[int, float]:
        """Simple TF-IDF BM25 scoring over bm25_tokens.

        Returns {index_in_candidates: score}.
        """
        # Document frequency (across full index for IDF)
        n_docs = len(self._signatures)
        df: Counter = Counter()
        for sig in self._signatures:
            for token in set(t.lower() for t in sig.bm25_tokens):
                df[token] += 1

        scores: dict[int, float] = {}
        query_lower = [k.lower() for k in keywords]

        for i, sig in enumerate(candidates):
            score = 0.0
            token_counts = Counter(t.lower() for t in sig.bm25_tokens)
            for term in query_lower:
                tf = token_counts.get(term, 0)
                if tf == 0:
                    continue
                doc_freq = df.get(term, 0)
                # IDF: log((N - df + 0.5) / (df + 0.5) + 1)
                idf = math.log((n_docs - doc_freq + 0.5) / (doc_freq + 0.5) + 1.0)
                # BM25 TF saturation (k1=1.2, b=0.0 since doc lengths are similar)
                k1 = 1.2
                tf_norm = (tf * (k1 + 1)) / (tf + k1)
                score += idf * tf_norm
            if score > 0:
                scores[i] = score
        return scores

    def _vector_score(
        self, candidates: list[TrajectorySignature], query_embeddings: list[list[float]]
    ) -> dict[int, float]:
        """Cosine similarity between query embeddings and candidate embeddings.

        Uses max similarity across multiple query vectors (one per hyde segment).
        Returns {index_in_candidates: score}.
        """
        if not query_embeddings:
            return {}

        # Build candidate embedding matrix
        cand_embs = []
        valid_indices = []
        for i, sig in enumerate(candidates):
            if sig.embedding and any(v != 0.0 for v in sig.embedding):
                cand_embs.append(sig.embedding)
                valid_indices.append(i)

        if not cand_embs:
            return {}

        cand_matrix = np.array(cand_embs, dtype=np.float32)  # (M, dim)
        # Normalize candidate embeddings
        norms = np.linalg.norm(cand_matrix, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        cand_matrix = cand_matrix / norms

        # Compute max cosine across all query embeddings
        max_scores = np.zeros(len(cand_embs), dtype=np.float32)
        for qe in query_embeddings:
            q_vec = np.array(qe, dtype=np.float32)
            q_norm = np.linalg.norm(q_vec)
            if q_norm > 0:
                q_vec = q_vec / q_norm
            cosines = cand_matrix @ q_vec  # (M,)
            max_scores = np.maximum(max_scores, cosines)

        scores: dict[int, float] = {}
        for idx, cand_idx in enumerate(valid_indices):
            if max_scores[idx] > 0:
                scores[cand_idx] = float(max_scores[idx])
        return scores

    def _rrf_fuse(
        self,
        candidates: list[TrajectorySignature],
        bm25_scores: dict[int, float],
        vector_scores: dict[int, float],
    ) -> list[tuple[TrajectorySignature, float]]:
        """Reciprocal Rank Fusion across BM25 and vector channels.

        RRF(d) = sum over channels: 1 / (K + rank_in_channel(d))
        """
        # If neither channel has scores, return all candidates with equal score
        if not bm25_scores and not vector_scores:
            return [(sig, 1.0) for sig in candidates]

        # Rank each channel
        bm25_ranked = sorted(bm25_scores.keys(), key=lambda i: bm25_scores[i], reverse=True)
        vector_ranked = sorted(vector_scores.keys(), key=lambda i: vector_scores[i], reverse=True)

        # Compute RRF scores
        rrf_scores: dict[int, float] = {}
        for rank, idx in enumerate(bm25_ranked):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (_RRF_K + rank + 1)
        for rank, idx in enumerate(vector_ranked):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (_RRF_K + rank + 1)

        # Include candidates that passed filters but have no score (very low priority)
        for i in range(len(candidates)):
            if i not in rrf_scores:
                rrf_scores[i] = 0.0

        return [(candidates[i], score) for i, score in rrf_scores.items()]
