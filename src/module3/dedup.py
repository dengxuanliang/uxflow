"""Module 3 semantic and token-level deduplication."""

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
    """Collapse duplicate candidates, keeping highest relevance per group."""
    if not candidates:
        return []

    ordered = sorted(candidates, key=lambda c: c.relevance_score, reverse=True)
    kept: list[Any] = []
    kept_vectors: list[np.ndarray] = []
    kept_minhashes: list[MinHash | None] = []

    for candidate in ordered:
        vec = np.asarray(candidate.embedding, dtype=np.float32)
        minhash = _minhash(_tokens_of(candidate))
        cosine_duplicate = any(
            candidate.sub_problem_id == kept_candidate.sub_problem_id
            and _cosine(vec, kept_vec) > cosine_threshold
            for kept_candidate, kept_vec in zip(kept, kept_vectors)
        )
        token_duplicate = any(
            candidate.sub_problem_id == kept_candidate.sub_problem_id
            and
            minhash is not None
            and existing is not None
            and minhash.jaccard(existing) >= minhash_threshold
            for kept_candidate, existing in zip(kept, kept_minhashes)
        )
        if cosine_duplicate or token_duplicate:
            continue
        kept.append(candidate)
        kept_vectors.append(vec)
        kept_minhashes.append(minhash)

    return kept
