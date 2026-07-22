"""Question normalization + content-addressed problem ids + cosine similarity."""
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 dengxuanliang

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections import OrderedDict

import numpy as np

__all__ = ["normalize_question", "problem_id_of", "cosine", "ProblemCompiler"]


def normalize_question(q: str) -> str:
    """NFC-normalize, strip, and collapse internal whitespace. Case preserved."""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", q).strip())


def problem_id_of(q: str) -> str:
    """Content-addressed id: first 16 hex chars of sha1 over the normalized question."""
    return hashlib.sha1(normalize_question(q).encode("utf-8")).hexdigest()[:16]


def cosine(a, b) -> float:
    """Cosine similarity; returns 0.0 if either side is a zero vector."""
    va = np.asarray(a, dtype=np.float64)
    vb = np.asarray(b, dtype=np.float64)
    na = float(np.linalg.norm(va))
    nb = float(np.linalg.norm(vb))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(va, vb) / (na * nb))


class ProblemCompiler:
    """只读编译收口 (D2): 持久 nearest → 进程 LRU → compile。绝不写 problem_store。"""

    def __init__(self, compiler, problem_store, embedder, *, tau_q,
                 serialize_fn, lru_cap=128):
        self._compiler = compiler
        self._problem_store = problem_store
        self._embedder = embedder
        self._tau_q = tau_q
        self._serialize_fn = serialize_fn
        self._lru_cap = lru_cap
        self._lru = OrderedDict()  # normalized question -> spec dict

    async def get_or_compile(self, question):
        """返回 (spec_dict, dedup_info|None)。dedup_info 命中已有问题时为
        {"matched_problem_id","matched_question","similarity"}，否则 None。
        只读：命中持久库/LRU 都不写；未命中 compile 后只写 LRU，不写 problem_store。"""
        emb = self._embedder.embed(question)
        hit = self._problem_store.nearest(emb)
        if hit is not None and hit[1] >= self._tau_q:
            pid, sim, spec = hit
            rec = self._problem_store.get(pid)
            dedup = {"matched_problem_id": pid,
                     "matched_question": rec["raw_question"] if rec else question,
                     "similarity": sim}
            return spec, dedup
        key = normalize_question(question)
        if key in self._lru:
            self._lru.move_to_end(key)
            return self._lru[key], None
        # search（读预览）路径无失败轨迹源 → 显式 failure_evidence=None，走纯文字打标。
        spec_obj = await self._compiler.compile(question, failure_evidence=None)
        spec = self._serialize_fn(spec_obj, question)
        self._lru[key] = spec
        self._lru.move_to_end(key)
        if len(self._lru) > self._lru_cap:
            self._lru.popitem(last=False)
        return spec, None
