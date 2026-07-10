import numpy as np

from uxflow_embed import FakeEmbedder


def test_dimension_default_1024():
    assert FakeEmbedder().dimension == 1024


def test_dimension_configurable():
    assert FakeEmbedder(dimension=8).dimension == 8


def test_embed_is_deterministic():
    e = FakeEmbedder(dimension=16)
    assert e.embed("hello") == e.embed("hello")


def test_embed_is_l2_normalized():
    vec = FakeEmbedder(dimension=16).embed("hello")
    assert abs(np.linalg.norm(vec) - 1.0) < 1e-6


def test_different_text_different_vector():
    e = FakeEmbedder(dimension=16)
    assert e.embed("hello") != e.embed("world")


def test_embed_batch_matches_embed():
    e = FakeEmbedder(dimension=16)
    assert e.embed_batch(["a", "b"]) == [e.embed("a"), e.embed("b")]


def test_empty_batch_returns_empty():
    assert FakeEmbedder().embed_batch([]) == []
