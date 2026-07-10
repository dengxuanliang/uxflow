import httpx
import numpy as np

from uxflow_embed import ApiEmbedder


def _mock_transport(dimension):
    def handler(request: httpx.Request) -> httpx.Response:
        import json
        payload = json.loads(request.content)
        inputs = payload["input"]
        if isinstance(inputs, str):
            inputs = [inputs]
        data = [
            {"embedding": [0.1] * dimension, "index": i} for i in range(len(inputs))
        ]
        return httpx.Response(200, json={"data": data})
    return httpx.MockTransport(handler)


def test_embed_returns_normalized_vector():
    emb = ApiEmbedder(
        api_key="test", model="text-embedding-3-small", dimension=8,
        base_url="https://example.com/v1",
        transport=_mock_transport(8),
    )
    vec = emb.embed("hello")
    assert len(vec) == 8
    assert abs(np.linalg.norm(vec) - 1.0) < 1e-6


def test_embed_batch_count_matches():
    emb = ApiEmbedder(
        api_key="test", model="m", dimension=8,
        base_url="https://example.com/v1",
        transport=_mock_transport(8),
    )
    out = emb.embed_batch(["a", "b", "c"])
    assert len(out) == 3


def test_empty_batch_short_circuits():
    emb = ApiEmbedder(
        api_key="test", model="m", dimension=8,
        base_url="https://example.com/v1",
        transport=_mock_transport(8),
    )
    assert emb.embed_batch([]) == []
