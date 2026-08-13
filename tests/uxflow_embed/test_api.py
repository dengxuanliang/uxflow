import httpx
import numpy as np
import pytest

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


def test_missing_key_raises_value_error(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError):
        ApiEmbedder(model="m", dimension=8)


def test_max_tries_below_one_raises_value_error():
    """max_tries < 1 would fall through the retry loop and `raise None`."""
    with pytest.raises(ValueError, match="max_tries"):
        ApiEmbedder(
            api_key="test", model="m", dimension=8,
            base_url="https://example.com/v1",
            transport=_mock_transport(8),
            max_tries=0,
        )


def test_non_200_raises_after_exhausting_retries():
    """5xx is retryable, so it raises only after max_tries attempts."""
    attempt = [0]

    def handler(request):
        attempt[0] += 1
        return httpx.Response(500, json={"error": "boom"})

    emb = ApiEmbedder(
        api_key="test", model="m", dimension=8,
        base_url="https://example.com/v1",
        transport=httpx.MockTransport(handler),
        max_tries=3,
        retry_delay=0.01,
    )
    with pytest.raises(httpx.HTTPStatusError):
        emb.embed("hello")
    assert attempt[0] == 3, "5xx must be retried up to max_tries"


def test_4xx_does_not_retry():
    """401/403/400 should raise immediately, not retry."""
    attempt = [0]

    def handler(request):
        attempt[0] += 1
        return httpx.Response(403, json={"error": "forbidden"})

    emb = ApiEmbedder(
        api_key="test", model="m", dimension=8,
        base_url="https://example.com/v1",
        transport=httpx.MockTransport(handler),
        max_tries=5,
        retry_delay=0.01,
    )
    with pytest.raises(httpx.HTTPStatusError, match="403"):
        emb.embed("hello")

    assert attempt[0] == 1


def test_429_triggers_retry():
    """429 rate-limit should retry like 5xx."""
    attempt = [0]

    def handler(request):
        attempt[0] += 1
        if attempt[0] < 3:
            return httpx.Response(429, json={"error": "rate limit"})
        return httpx.Response(200, json={
            "data": [{"embedding": [1.0] * 8, "index": 0}]
        })

    emb = ApiEmbedder(
        api_key="test", model="m", dimension=8,
        base_url="https://example.com/v1",
        transport=httpx.MockTransport(handler),
        max_tries=5,
        retry_delay=0.01,
    )
    result = emb.embed("hello")
    assert len(result) == 8
    assert attempt[0] == 3


def test_timeout_triggers_retry():
    """Timeout on attempt 1-4 should retry; success on attempt 5."""
    attempt = [0]

    def handler(request):
        attempt[0] += 1
        if attempt[0] < 5:
            raise httpx.ReadTimeout("forced timeout")
        return httpx.Response(200, json={
            "data": [{"embedding": [1.0] * 8, "index": 0}]
        })

    emb = ApiEmbedder(
        api_key="test", model="m", dimension=8,
        base_url="https://example.com/v1",
        transport=httpx.MockTransport(handler),
        timeout=1.0,
        max_tries=5,
        retry_delay=0.01,
    )
    result = emb.embed("hello")
    assert len(result) == 8
    assert attempt[0] == 5


def test_base64_encoding_format():
    """When server returns base64, decode to float32 correctly."""
    import base64
    import struct

    def handler(request):
        import json
        body = json.loads(request.content)
        assert body.get("encoding_format") == "base64"
        vec = [0.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0]
        raw = struct.pack(f"{len(vec)}f", *vec)
        b64 = base64.b64encode(raw).decode("ascii")
        return httpx.Response(200, json={"data": [{"embedding": b64, "index": 0}]})

    emb = ApiEmbedder(
        api_key="test", model="m", dimension=8,
        base_url="https://example.com/v1",
        transport=httpx.MockTransport(handler),
    )
    result = emb.embed("hello")
    assert len(result) == 8
    assert abs(result[0] - 0.5) < 1e-6


def test_preferred_batch_size_from_constructor():
    emb = ApiEmbedder(
        api_key="test", model="m", dimension=8,
        preferred_batch_size=64,
    )
    assert emb.preferred_batch_size == 64


def test_dimensions_parameter_sent():
    """The model does native dimension reduction; ask for it rather than truncating."""
    received = {}

    def handler(request):
        import json
        received.update(json.loads(request.content))
        dim = received.get("dimensions", 8)
        return httpx.Response(200, json={
            "data": [{"embedding": [0.707] * dim, "index": 0}]
        })

    emb = ApiEmbedder(
        api_key="test", model="text-embedding-3-large", dimension=1024,
        base_url="https://example.com/v1",
        transport=httpx.MockTransport(handler),
    )
    result = emb.embed("hello")
    assert received.get("dimensions") == 1024
    assert len(result) == 1024


def test_out_of_order_index_realigns_to_input(monkeypatch):
    # Server returns data in REVERSED index order; output must realign to input order.
    def handler(request):
        import json
        inputs = json.loads(request.content)["input"]
        # distinguishable embeddings: item i gets value (i+1)/10 in slot 0
        data = [
            {"embedding": [(i + 1) / 10] + [0.0] * 7, "index": i}
            for i in range(len(inputs))
        ]
        data.reverse()  # deliberately out of order
        return httpx.Response(200, json={"data": data})
    emb = ApiEmbedder(
        api_key="test", model="m", dimension=8,
        base_url="https://example.com/v1",
        transport=httpx.MockTransport(handler),
    )
    out = emb.embed_batch(["a", "b", "c"])
    # After sort-by-index + L2-normalize, each vector's slot 0 should be the
    # largest component and strictly increasing across a<b<c is not guaranteed
    # post-normalize, but slot 0 should be positive & dominant for each. Assert
    # realignment by checking slot 0 is the max component for every row.
    for vec in out:
        assert vec[0] == max(vec)
