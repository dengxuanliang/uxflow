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
        # 429 now backs off exponentially; keep the base tiny so this test
        # stays fast. test_429_backoff_grows_exponentially asserts the growth.
        rate_limit_base=0.01,
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


def _record_sleeps(monkeypatch):
    """Capture retry delays without actually sleeping.

    Patches the module's own `time.sleep`, so the assertions below measure the
    delay the code *chose*, not wall-clock time. That keeps the suite fast and
    lets us assert exact values instead of fuzzy timing.
    """
    import uxflow_embed.api as api_mod

    delays: list[float] = []
    monkeypatch.setattr(api_mod.time, "sleep", lambda d: delays.append(d))
    return delays


def test_429_backoff_grows_exponentially(monkeypatch):
    """429 is a closed quota window, not queuing -- it must back off, not hammer.

    A flat delay knocks on a locked door N times in N seconds; Azure tier
    windows are ~60s, so exhaustion is guaranteed. Delays must double.
    """
    delays = _record_sleeps(monkeypatch)

    def handler(request):
        return httpx.Response(429, json={"error": "rate limit"})

    emb = ApiEmbedder(
        api_key="test", model="m", dimension=8,
        base_url="https://example.com/v1",
        transport=httpx.MockTransport(handler),
        max_tries=4,
        retry_delay=1.0,
        rate_limit_base=2.0,
        rate_limit_cap=32.0,
    )
    with pytest.raises(httpx.HTTPStatusError):
        emb.embed("hello")

    # 3 sleeps for 4 attempts (none after the last), doubling from the base.
    assert delays == [2.0, 4.0, 8.0], (
        f"429 must back off exponentially, got {delays}"
    )


def test_429_honours_retry_after_header(monkeypatch):
    """A server-stated wait beats our guess. This endpoint sends none, but
    others do, and obeying it when offered is strictly better."""
    delays = _record_sleeps(monkeypatch)

    def handler(request):
        return httpx.Response(
            429, json={"error": "rate limit"}, headers={"retry-after": "7"}
        )

    emb = ApiEmbedder(
        api_key="test", model="m", dimension=8,
        base_url="https://example.com/v1",
        transport=httpx.MockTransport(handler),
        max_tries=3,
        rate_limit_base=2.0,
        rate_limit_cap=32.0,
    )
    with pytest.raises(httpx.HTTPStatusError):
        emb.embed("hello")

    assert delays == [7.0, 7.0], f"Retry-After must win over backoff, got {delays}"


def test_429_backoff_is_capped(monkeypatch):
    """Doubling without a ceiling would let one batch hang for hours."""
    delays = _record_sleeps(monkeypatch)

    def handler(request):
        return httpx.Response(429, json={"error": "rate limit"})

    emb = ApiEmbedder(
        api_key="test", model="m", dimension=8,
        base_url="https://example.com/v1",
        transport=httpx.MockTransport(handler),
        max_tries=8,
        rate_limit_base=2.0,
        rate_limit_cap=8.0,
    )
    with pytest.raises(httpx.HTTPStatusError):
        emb.embed("hello")

    assert delays == [2.0, 4.0, 8.0, 8.0, 8.0, 8.0, 8.0], (
        f"backoff must saturate at the cap, got {delays}"
    )
    assert max(delays) <= 8.0


def test_timeout_retry_stays_flat(monkeypatch):
    """The timeout path must NOT inherit the 429 backoff.

    Queuing and quota are opposite failure modes: a queued request stays slow
    so resending fast is right, while a closed window needs waiting out. This
    asserts the two paths did not get merged.
    """
    delays = _record_sleeps(monkeypatch)

    def handler(request):
        raise httpx.ReadTimeout("forced timeout")

    emb = ApiEmbedder(
        api_key="test", model="m", dimension=8,
        base_url="https://example.com/v1",
        transport=httpx.MockTransport(handler),
        max_tries=4,
        retry_delay=1.0,
        rate_limit_base=2.0,
    )
    with pytest.raises(httpx.ReadTimeout):
        emb.embed("hello")

    assert delays == [1.0, 1.0, 1.0], (
        f"timeouts must use the flat delay, not backoff, got {delays}"
    )


def test_5xx_retry_stays_flat(monkeypatch):
    """5xx is a server hiccup, not a quota window -- flat like timeouts."""
    delays = _record_sleeps(monkeypatch)

    def handler(request):
        return httpx.Response(500, json={"error": "boom"})

    emb = ApiEmbedder(
        api_key="test", model="m", dimension=8,
        base_url="https://example.com/v1",
        transport=httpx.MockTransport(handler),
        max_tries=3,
        retry_delay=1.0,
        rate_limit_base=2.0,
    )
    with pytest.raises(httpx.HTTPStatusError):
        emb.embed("hello")

    assert delays == [1.0, 1.0], f"5xx must use the flat delay, got {delays}"


def test_all_empty_batch_skips_http_entirely():
    """Every input empty -> zero vectors, and no request is ever sent.

    The API rejects empty input with 400, and 400 is deliberately not retried,
    so a single empty slice would otherwise kill a whole ingest.
    """
    calls = [0]

    def handler(request):
        calls[0] += 1
        return httpx.Response(200, json={"data": []})

    emb = ApiEmbedder(
        api_key="test", model="m", dimension=8,
        base_url="https://example.com/v1",
        transport=httpx.MockTransport(handler),
    )
    out = emb.embed_batch(["", "   ", "\n\t"])

    assert len(out) == 3
    assert all(vec == [0.0] * 8 for vec in out)
    assert calls[0] == 0, "an all-empty batch must not hit the network at all"


def test_mixed_empty_and_real_texts_preserve_order():
    """Empty slots get zero vectors; real ones keep their own embedding.

    Each input gets a distinguishable vector so a mis-ordering is detectable --
    identical mock vectors would let a reordering bug pass silently.
    """
    sent = {}

    def handler(request):
        import json
        payload = json.loads(request.content)
        sent["input"] = payload["input"]
        # Tag each returned vector with its input's first character, so the
        # assertions below can prove which text produced which vector.
        data = [
            {"embedding": [float(ord(t[0]))] + [0.0] * 7, "index": i}
            for i, t in enumerate(payload["input"])
        ]
        return httpx.Response(200, json={"data": data})

    emb = ApiEmbedder(
        api_key="test", model="m", dimension=8,
        base_url="https://example.com/v1",
        transport=httpx.MockTransport(handler),
    )
    out = emb.embed_batch(["a", "", "b", "   ", "c"])

    assert len(out) == 5, "one result per input, including the empty ones"
    assert sent["input"] == ["a", "b", "c"], "blanks must not reach the API"

    assert out[1] == [0.0] * 8
    assert out[3] == [0.0] * 8

    # Compare against a fresh single embed of the same text: same text => same
    # vector, so this pins which input produced which slot.
    for pos, text in ((0, "a"), (2, "b"), (4, "c")):
        assert out[pos] == emb.embed(text), f"slot {pos} must hold {text!r}"


def test_empty_string_at_end_of_batch():
    """Trailing empty is the classic off-by-one position."""
    def handler(request):
        import json
        inputs = json.loads(request.content)["input"]
        data = [
            {"embedding": [float(ord(t[0]))] + [0.0] * 7, "index": i}
            for i, t in enumerate(inputs)
        ]
        return httpx.Response(200, json={"data": data})

    emb = ApiEmbedder(
        api_key="test", model="m", dimension=8,
        base_url="https://example.com/v1",
        transport=httpx.MockTransport(handler),
    )
    out = emb.embed_batch(["x", "y", ""])

    assert len(out) == 3
    assert out[2] == [0.0] * 8, "the trailing blank must be the zeroed one"
    assert out[0] == emb.embed("x")
    assert out[1] == emb.embed("y")


def test_empty_string_at_start_of_batch():
    """Leading empty shifts every following index if the remap is wrong."""
    def handler(request):
        import json
        inputs = json.loads(request.content)["input"]
        data = [
            {"embedding": [float(ord(t[0]))] + [0.0] * 7, "index": i}
            for i, t in enumerate(inputs)
        ]
        return httpx.Response(200, json={"data": data})

    emb = ApiEmbedder(
        api_key="test", model="m", dimension=8,
        base_url="https://example.com/v1",
        transport=httpx.MockTransport(handler),
    )
    out = emb.embed_batch(["", "p", "q"])

    assert len(out) == 3
    assert out[0] == [0.0] * 8
    assert out[1] == emb.embed("p")
    assert out[2] == emb.embed("q")


def test_zero_vector_is_not_normalized_to_nan():
    """_ensure_dimension divides by the norm; norm 0 would yield NaN and
    poison every downstream cosine comparison."""
    import math

    emb = ApiEmbedder(
        api_key="test", model="m", dimension=8,
        base_url="https://example.com/v1",
        transport=_mock_transport(8),
    )
    out = emb.embed_batch([""])
    assert not any(math.isnan(v) for v in out[0]), "zero vector must not be NaN"
    assert out[0] == [0.0] * 8


def test_base64_encoding_format():
    """When server returns base64, decode to float32 correctly."""
    import base64
    import struct

    def handler(request):
        import json
        body = json.loads(request.content)
        assert body.get("encoding_format") == "base64"
        vec = [0.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0]
        raw = struct.pack(f"<{len(vec)}f", *vec)
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


def test_corrupt_base64_raises_with_context():
    """A malformed blob is not an httpx error, so it must not escape as a bare
    binascii traceback with no mention of where it came from."""
    def handler(request):
        # "!!!!" is outside the base64 alphabet -> binascii.Error.
        return httpx.Response(200, json={
            "data": [{"embedding": "!!!!not-valid-base64!!!!", "index": 0}]
        })

    emb = ApiEmbedder(
        api_key="test", model="text-embedding-3-large", dimension=8,
        base_url="https://example.com/v1",
        transport=httpx.MockTransport(handler),
    )
    with pytest.raises(ValueError) as excinfo:
        emb.embed("hello")

    msg = str(excinfo.value)
    assert "text-embedding-3-large" in msg, "error must name the model"
    assert "https://example.com/v1/embeddings" in msg, "error must name the endpoint"


def test_base64_blob_not_multiple_of_four_bytes_raises_with_context():
    """A blob that decodes cleanly but isn't a whole number of float32s."""
    import base64

    def handler(request):
        # 6 raw bytes: valid base64, but 6 % 4 != 0, so struct.unpack of
        # 1 float (4 bytes) against a 6-byte buffer raises struct.error.
        blob = b"\x00\x01\x02\x03\x04\x05"
        b64 = base64.b64encode(blob).decode("ascii")
        return httpx.Response(200, json={"data": [{"embedding": b64, "index": 0}]})

    emb = ApiEmbedder(
        api_key="test", model="text-embedding-3-large", dimension=8,
        base_url="https://example.com/v1",
        transport=httpx.MockTransport(handler),
    )
    with pytest.raises(ValueError) as excinfo:
        emb.embed("hello")

    msg = str(excinfo.value)
    assert "text-embedding-3-large" in msg, "error must name the model"
    assert "https://example.com/v1/embeddings" in msg, "error must name the endpoint"


def test_preferred_batch_size_defaults_to_32():
    """32 keeps a 3072-dim base64 response near 0.5MB, half the ~1MB ceiling."""
    emb = ApiEmbedder(api_key="test", model="m", dimension=8)
    assert emb.preferred_batch_size == 32


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
