"""Module 3 semantic and token-level deduplication."""
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 dengxuanliang

from __future__ import annotations

from typing import Any

import numpy as np
from datasketch import MinHash

__all__ = ["deduplicate"]


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    if a_norm == 0 or b_norm == 0:
        return 0.0
    return float(a @ b / (a_norm * b_norm))


def _tokens_of(candidate: Any) -> set[str]:
    tokens = getattr(candidate, "_tokens", None) or getattr(candidate, "bm25_tokens", None) or []
    return {str(token) for token in tokens}


def _minhash(tokens: set[str], num_perm: int = 64) -> MinHash | None:
    if not tokens:
        return None
    mh = MinHash(num_perm=num_perm)
    for token in sorted(tokens):
        mh.update(token.encode("utf-8"))
    return mh


def deduplicate(
    candidates: list[Any],
    *,
    cosine_threshold: float = 0.95,
    minhash_threshold: float = 0.9,
) -> list[Any]:
    """Collapse near-duplicate MergedCandidates, absorbing attribution into survivor."""
    if not candidates:
        return []
    from module3.merge import absorb

    ordered = sorted(candidates, key=lambda c: c.relevance_score, reverse=True)
    kept: list[Any] = []
    kept_vectors: list[np.ndarray] = []
    kept_minhashes: list[MinHash | None] = []

    for candidate in ordered:
        vec = np.asarray(candidate.embedding, dtype=np.float32)
        minhash = _minhash(_tokens_of(candidate))
        dup_of = None
        for i, kv in enumerate(kept_vectors):
            if _cosine(vec, kv) > cosine_threshold:
                dup_of = i
                break
            existing = kept_minhashes[i]
            if (minhash is not None and existing is not None
                    and minhash.jaccard(existing) >= minhash_threshold):
                dup_of = i
                break
        if dup_of is not None:
            absorb(kept[dup_of], candidate)   # 吸收归属/掩码，保覆盖
            continue
        kept.append(candidate)
        kept_vectors.append(vec)
        kept_minhashes.append(minhash)

    return kept
