# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 dengxuanliang
"""End-to-end check of the api embedding backend against a live endpoint.

Skipped without credentials so CI stays hermetic. Run it deliberately:
    uv run pytest tests/integration/test_api_embedder_e2e.py -v -s
"""

import os
import pathlib

import numpy as np
import pytest
from dotenv import load_dotenv

load_dotenv(pathlib.Path(__file__).parent.parent.parent / ".env")

from uxflow_runtime import make_embedder  # noqa: E402

pytestmark = pytest.mark.skipif(
    not (os.environ.get("LITELLM_KEY") or os.environ.get("OPENAI_API_KEY")),
    reason="no LITELLM_KEY/OPENAI_API_KEY in environment; skipping live-endpoint check",
)


def _cos(a, b) -> float:
    va, vb = np.asarray(a), np.asarray(b)
    return float(va @ vb / (np.linalg.norm(va) * np.linalg.norm(vb)))


def test_api_embedder_against_live_endpoint(monkeypatch):
    monkeypatch.setenv("UXFLOW_EMBED_BACKEND", "api")
    emb = make_embedder()

    assert emb.dimension == 3072
    assert emb.preferred_batch_size == 32

    texts = [
        "修复运行时抛出的异常，如 TypeError/KeyError/ValueError",
        "为函数补充单元测试",
        "把咖啡豆磨成粉后手冲",
    ]
    vectors = emb.embed_batch(texts)

    assert len(vectors) == 3
    for vec in vectors:
        assert len(vec) == emb.dimension
        norm = float(np.linalg.norm(np.asarray(vec)))
        assert abs(norm - 1.0) < 1e-3

    # Single-embed path must agree with the batch path per-component — this is
    # what would catch a base64-vs-list decoding divergence between the two.
    single = emb.embed(texts[0])
    batch_first = np.asarray(vectors[0])
    single_arr = np.asarray(single)
    assert single_arr.shape == batch_first.shape
    assert np.max(np.abs(single_arr - batch_first)) < 1e-5

    # Two unrelated texts must not collapse to near-identical vectors — this
    # is what would catch an endpoint returning a constant vector.
    sim = _cos(vectors[0], vectors[2])
    assert sim < 0.99
