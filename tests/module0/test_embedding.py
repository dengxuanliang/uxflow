"""Embedding model tests.

These tests require sentence-transformers + torch installed, plus
the Qwen3-Embedding model downloaded (~1.2GB). Skip if unavailable.
"""

import pytest

try:
    import torch
    import sentence_transformers
    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False

pytestmark = pytest.mark.skipif(not HAS_DEPS, reason="sentence-transformers/torch not installed")


@pytest.fixture(scope="module")
def model():
    """Load model once for all tests in this module (expensive init)."""
    from module0.embedding import EmbeddingModel
    return EmbeddingModel()


def test_model_loads(model):
    assert model is not None
    assert model.dimension == 1024


def test_embed_single(model):
    import numpy as np
    vec = model.embed("hello world")
    assert len(vec) == 1024
    norm = np.linalg.norm(vec)
    assert abs(norm - 1.0) < 0.01


def test_embed_batch(model):
    import numpy as np
    texts = ["first text", "second text", "third text"]
    vecs = model.embed_batch(texts)
    assert len(vecs) == 3
    assert all(len(v) == 1024 for v in vecs)
    for v in vecs:
        assert abs(np.linalg.norm(v) - 1.0) < 0.01


def test_embed_hyde_positive(model):
    """Simulate embedding hyde_positive segments."""
    import numpy as np
    segments = [
        "工具正确写入 python 文件，无语法错误，执行结果 exit code 0",
        "python 文件包含合法 import 和函数定义，lint 通过",
    ]
    vecs = model.embed_batch(segments)
    assert len(vecs) == 2
    cos_sim = np.dot(vecs[0], vecs[1])
    assert cos_sim < 0.99  # related but not identical
