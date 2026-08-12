# API Embedder Switch Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Switch embedding from local CPU Qwen to API calls with retry, base64 encoding, and batch size tuning for restricted networks.

**Architecture:** Enhance existing `ApiEmbedder` with retry logic and base64 support; add optional `preferred_batch_size` protocol attribute; wire LITELLM credentials as primary, OpenAI as fallback; make batch/timeout/retries tunable via env vars.

**Tech Stack:** httpx, base64, numpy (already in deps)

---

## File Structure

**Modified:**
- `src/uxflow_embed/protocol.py` — document optional `preferred_batch_size`
- `src/uxflow_embed/api.py` — retry, base64, dimensions param, preferred_batch_size
- `src/uxflow_embed/local.py` — add preferred_batch_size=32
- `src/uxflow_embed/fake.py` — add preferred_batch_size=128
- `src/uxflow_runtime.py` — LITELLM credentials priority, read batch/timeout/tries env vars
- `src/module1/pipeline.py` — getattr(emb, "preferred_batch_size", _EMBED_CHUNK)
- `.env.example` — document api backend config, network tuning
- `README.md` / `README.zh-CN.md` — update embedding backends section
- `tests/uxflow_embed/test_api.py` — base64, retry, 4xx-no-retry, 429-retry tests
- `tests/uxflow_embed/test_fake.py` / `test_local.py` / `test_protocol.py` — batch size
- `tests/test_runtime_wiring.py` — LITELLM credential fallback, env tunables
- `docs/superpowers/specs/2026-08-12-api-embedder-switch-design.md` — record §4.8 calibration data

**Created:**
- `scripts/calibrate_thresholds_api.py` — cosine distribution under the active backend
- `tests/integration/test_api_embedder_e2e.py` — real-endpoint smoke test (skipped without a key)

**Environment note:** run every command through `uv run`. Bare `pytest` is not
on PATH in this project. The `dev` extra provides pytest, pytest-asyncio and
ruff only — there is no mypy and no pytest-cov, so do not add steps that call them.

---

### Task 1: Protocol — Add Optional Batch Size Attribute

**Files:**
- Modify: `src/uxflow_embed/protocol.py:16-25`
- Test: `tests/uxflow_embed/test_protocol.py`

- [ ] **Step 1: Write failing test for optional attribute**

```python
# Add to tests/uxflow_embed/test_protocol.py

def test_preferred_batch_size_is_optional():
    """Implementations without preferred_batch_size still satisfy the Protocol."""
    from uxflow_embed.protocol import Embedder
    
    class MinimalEmbedder:
        @property
        def dimension(self) -> int:
            return 8
        def embed(self, text: str) -> list[float]:
            return [0.0] * 8
        def embed_batch(self, texts: list[str]) -> list[list[float]]:
            return [[0.0] * 8] * len(texts)
    
    # Should NOT raise — preferred_batch_size is optional
    assert isinstance(MinimalEmbedder(), Embedder)
    
    # getattr with default should work
    emb = MinimalEmbedder()
    assert getattr(emb, "preferred_batch_size", 99) == 99
```

- [ ] **Step 2: Run test to verify it passes (baseline)**

Run: `uv run pytest tests/uxflow_embed/test_protocol.py::test_preferred_batch_size_is_optional -v`

Expected: PASS (no changes to protocol yet, test validates that optional attrs work with runtime_checkable)

- [ ] **Step 3: Add docstring note about optional attribute**

```python
# In src/uxflow_embed/protocol.py, update class docstring

@runtime_checkable
class Embedder(Protocol):
    """Produces L2-normalized vectors. Implementations: Fake / Local / Api.
    
    Optional attributes that implementations MAY provide:
    - preferred_batch_size: int — optimal batch size for embed_batch calls
    """

    @property
    def dimension(self) -> int: ...

    def embed(self, text: str) -> list[float]: ...

    def embed_batch(self, texts: list[str]) -> list[list[float]]: ...
```

- [ ] **Step 4: Run full protocol tests**

Run: `uv run pytest tests/uxflow_embed/test_protocol.py -v`

Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/uxflow_embed/protocol.py tests/uxflow_embed/test_protocol.py
git commit -m "feat(embed): document optional preferred_batch_size in Protocol

Consumers use getattr(emb, 'preferred_batch_size', default) to respect
implementation preferences without breaking third-party embedders.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 2: FakeEmbedder — Add Batch Size

**Files:**
- Modify: `src/uxflow_embed/fake.py:19-42`
- Test: `tests/uxflow_embed/test_fake.py`

- [ ] **Step 1: Write failing test**

```python
# Add to tests/uxflow_embed/test_fake.py

def test_preferred_batch_size():
    from uxflow_embed import FakeEmbedder
    emb = FakeEmbedder()
    assert emb.preferred_batch_size == 128
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/uxflow_embed/test_fake.py::test_preferred_batch_size -v`

Expected: FAIL with "AttributeError: 'FakeEmbedder' object has no attribute 'preferred_batch_size'"

- [ ] **Step 3: Add property**

```python
# In src/uxflow_embed/fake.py, after dimension property

@property
def preferred_batch_size(self) -> int:
    """FakeEmbedder is CPU-bound numpy ops; larger batches amortize loop overhead."""
    return 128
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/uxflow_embed/test_fake.py::test_preferred_batch_size -v`

Expected: PASS

- [ ] **Step 5: Run full fake tests**

Run: `uv run pytest tests/uxflow_embed/test_fake.py -v`

Expected: All tests PASS

- [ ] **Step 6: Commit**

```bash
git add src/uxflow_embed/fake.py tests/uxflow_embed/test_fake.py
git commit -m "feat(embed): add preferred_batch_size=128 to FakeEmbedder

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 3: LocalEmbedder — Add Batch Size

**Files:**
- Modify: `src/uxflow_embed/local.py:24-80`
- Test: `tests/uxflow_embed/test_local.py`

- [ ] **Step 1: Write failing test**

```python
# Add to tests/uxflow_embed/test_local.py

def test_preferred_batch_size():
    """LocalEmbedder declares batch 32, matching its internal encode batch_size."""
    from uxflow_embed import LocalEmbedder
    try:
        emb = LocalEmbedder()
    except ImportError:
        pytest.skip("local-embed extra not installed")
    assert emb.preferred_batch_size == 32
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/uxflow_embed/test_local.py::test_preferred_batch_size -v`

Expected: FAIL with AttributeError (or SKIP if deps missing)

- [ ] **Step 3: Add property**

```python
# In src/uxflow_embed/local.py, after dimension property (around line 49)

@property
def preferred_batch_size(self) -> int:
    """Internal encode() uses batch_size=32; match it to avoid re-chunking."""
    return 32
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/uxflow_embed/test_local.py::test_preferred_batch_size -v`

Expected: PASS (or SKIP if deps missing)

- [ ] **Step 5: Commit**

```bash
git add src/uxflow_embed/local.py tests/uxflow_embed/test_local.py
git commit -m "feat(embed): add preferred_batch_size=32 to LocalEmbedder

Matches the internal SentenceTransformer encode batch_size.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 4: ApiEmbedder — Add Retry Logic

**Files:**
- Modify: `src/uxflow_embed/api.py:20-80`
- Test: `tests/uxflow_embed/test_api.py`

- [ ] **Step 1: Write failing test for timeout retry**

```python
# Add to tests/uxflow_embed/test_api.py

def test_timeout_triggers_retry():
    """Timeout on attempt 1-4 should retry; success on attempt 5."""
    import httpx
    from uxflow_embed import ApiEmbedder
    
    attempt = [0]
    def handler(request):
        attempt[0] += 1
        if attempt[0] < 5:
            raise httpx.ReadTimeout("forced timeout")
        # 5th attempt succeeds
        return httpx.Response(200, json={
            "data": [{"embedding": [1.0] * 8, "index": 0}]
        })
    
    emb = ApiEmbedder(
        api_key="test", model="m", dimension=8,
        base_url="https://example.com/v1",
        transport=httpx.MockTransport(handler),
        timeout=1.0,
        max_tries=5,
        retry_delay=0.01,  # fast for tests
    )
    result = emb.embed("hello")
    assert len(result) == 8
    assert attempt[0] == 5  # 4 failures + 1 success
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/uxflow_embed/test_api.py::test_timeout_triggers_retry -v`

Expected: FAIL with "TypeError: __init__() got unexpected keyword argument 'max_tries'"

- [ ] **Step 3: Add retry parameters to __init__**

```python
# In src/uxflow_embed/api.py, update __init__ signature and body

def __init__(
    self,
    *,
    model: str,
    dimension: int,
    api_key: str | None = None,
    base_url: str = "https://api.openai.com/v1",
    timeout: float = 25.0,
    max_tries: int = 5,
    retry_delay: float = 1.0,
    transport: httpx.BaseTransport | None = None,
):
    key = api_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise ValueError("ApiEmbedder needs an api_key or the OPENAI_API_KEY env var.")
    self._model = model
    self._dimension = dimension
    self._url = base_url.rstrip("/") + "/embeddings"
    self._timeout = timeout
    self._max_tries = max_tries
    self._retry_delay = retry_delay
    self._client = httpx.Client(
        headers={"Authorization": f"Bearer {key}"},
        timeout=timeout,
        transport=transport,
    )
```

- [ ] **Step 4: Implement retry logic in embed_batch**

```python
# In src/uxflow_embed/api.py, replace embed_batch body

def embed_batch(self, texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    
    import time
    last_exc = None
    for attempt in range(1, self._max_tries + 1):
        try:
            resp = self._client.post(
                self._url,
                json={"model": self._model, "input": texts}
            )
            resp.raise_for_status()
            rows = sorted(resp.json()["data"], key=lambda d: d["index"])
            return [self._ensure_dimension(r["embedding"]) for r in rows]
        except (httpx.TimeoutException, httpx.HTTPStatusError) as e:
            # Retry on timeout, 5xx, 429
            if isinstance(e, httpx.HTTPStatusError):
                if 400 <= e.response.status_code < 500 and e.response.status_code != 429:
                    # 4xx except 429: auth/format error, don't retry
                    raise
            last_exc = e
            if attempt < self._max_tries:
                time.sleep(self._retry_delay)
    
    # All attempts exhausted
    raise last_exc
```

- [ ] **Step 5: Run retry test to verify it passes**

Run: `uv run pytest tests/uxflow_embed/test_api.py::test_timeout_triggers_retry -v`

Expected: PASS

- [ ] **Step 6: Write test for 4xx no-retry**

```python
# Add to tests/uxflow_embed/test_api.py

def test_4xx_does_not_retry():
    """401/403/400 should raise immediately, not retry."""
    import httpx
    from uxflow_embed import ApiEmbedder
    
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
    
    assert attempt[0] == 1  # No retries
```

- [ ] **Step 7: Run 4xx test**

Run: `uv run pytest tests/uxflow_embed/test_api.py::test_4xx_does_not_retry -v`

Expected: PASS

- [ ] **Step 8: Write test for 429 retry**

```python
# Add to tests/uxflow_embed/test_api.py

def test_429_triggers_retry():
    """429 rate-limit should retry like 5xx."""
    import httpx
    from uxflow_embed import ApiEmbedder
    
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
```

- [ ] **Step 9: Run 429 test**

Run: `uv run pytest tests/uxflow_embed/test_api.py::test_429_triggers_retry -v`

Expected: PASS

- [ ] **Step 10: Fix the pre-existing 500 test that retry now slows down**

`test_non_200_raises` already exists and constructs `ApiEmbedder` without
retry args. Once 5xx is retryable it will retry 5 times with a 1s sleep,
turning a fast test into a ~4-second one. Update it to pin the retry knobs
and to assert the retry count explicitly:

```python
# In tests/uxflow_embed/test_api.py, REPLACE the existing test_non_200_raises

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
```

- [ ] **Step 11: Run all api tests**

Run: `uv run pytest tests/uxflow_embed/test_api.py -v`

Expected: All tests PASS, and the suite finishes in well under a second
(no real 1s sleeps — every test pins `retry_delay=0.01`)

- [ ] **Step 12: Commit**

```bash
git add src/uxflow_embed/api.py tests/uxflow_embed/test_api.py
git commit -m "feat(embed): add retry to ApiEmbedder for proxy queuing failures

The proxy times out on roughly a third of batch requests — not because
the connection hangs, but because the request lands in a slow queue.
Abandoning and re-sending usually lands on a free worker and returns in
seconds, so retries use a flat short delay rather than exponential
backoff, which would just wait out a queue that never speeds up.

Retries timeout/5xx/429; 4xx raises immediately, since a bad key or a
malformed request fails identically five times and only turns a
second-long diagnosis into a minute-long one.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 5: ApiEmbedder — Add base64 Encoding Support

**Files:**
- Modify: `src/uxflow_embed/api.py`
- Test: `tests/uxflow_embed/test_api.py`

- [ ] **Step 1: Write failing test for base64**

```python
# Add to tests/uxflow_embed/test_api.py

def test_base64_encoding_format():
    """When server returns base64, decode to float32 correctly."""
    import httpx, base64, struct
    from uxflow_embed import ApiEmbedder
    
    def handler(request):
        import json
        body = json.loads(request.content)
        assert body.get("encoding_format") == "base64"
        
        # Return base64-encoded float32 vector
        vec = [0.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0]
        raw = struct.pack(f"{len(vec)}f", *vec)
        b64 = base64.b64encode(raw).decode("ascii")
        
        return httpx.Response(200, json={
            "data": [{"embedding": b64, "index": 0}]
        })
    
    emb = ApiEmbedder(
        api_key="test", model="m", dimension=8,
        base_url="https://example.com/v1",
        transport=httpx.MockTransport(handler),
    )
    result = emb.embed("hello")
    assert len(result) == 8
    # After L2-normalize, [0.5,0.5,0.5,0.5,0,0,0,0] becomes [0.5,0.5,0.5,0.5,0,0,0,0]
    # (already unit norm)
    assert abs(result[0] - 0.5) < 1e-6
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/uxflow_embed/test_api.py::test_base64_encoding_format -v`

Expected: FAIL (no base64 handling yet)

- [ ] **Step 3: Update embed_batch to request base64 and decode**

```python
# In src/uxflow_embed/api.py, update embed_batch request and response handling

def embed_batch(self, texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    
    import time
    last_exc = None
    for attempt in range(1, self._max_tries + 1):
        try:
            resp = self._client.post(
                self._url,
                json={
                    "model": self._model,
                    "input": texts,
                    "encoding_format": "base64",
                }
            )
            resp.raise_for_status()
            rows = sorted(resp.json()["data"], key=lambda d: d["index"])
            return [self._decode_embedding(r["embedding"]) for r in rows]
        except (httpx.TimeoutException, httpx.HTTPStatusError) as e:
            if isinstance(e, httpx.HTTPStatusError):
                if 400 <= e.response.status_code < 500 and e.response.status_code != 429:
                    raise
            last_exc = e
            if attempt < self._max_tries:
                time.sleep(self._retry_delay)
    
    raise last_exc
```

- [ ] **Step 4: Add _decode_embedding helper**

```python
# In src/uxflow_embed/api.py, add new method after _ensure_dimension

def _decode_embedding(self, raw) -> list[float]:
    """Decode base64 or list embedding, truncate/pad to dimension, L2-normalize."""
    import base64
    import struct
    
    if isinstance(raw, str):
        # base64-encoded float32
        blob = base64.b64decode(raw)
        count = len(blob) // 4
        vec = struct.unpack(f"{count}f", blob)
        return self._ensure_dimension(list(vec))
    else:
        # list of floats (fallback for endpoints that don't support base64)
        return self._ensure_dimension(raw)
```

- [ ] **Step 5: Run base64 test**

Run: `uv run pytest tests/uxflow_embed/test_api.py::test_base64_encoding_format -v`

Expected: PASS

- [ ] **Step 6: Update existing normalization test to handle base64**

```python
# In tests/uxflow_embed/test_api.py, update test_normalization_and_dimension_adjustment

def test_normalization_and_dimension_adjustment():
    """L2-normalize + truncate/pad works for both base64 and list format."""
    import httpx, base64, struct
    from uxflow_embed import ApiEmbedder
    
    def handler(request):
        import json
        body = json.loads(request.content)
        if "base64" in body.get("input", [""])[0]:
            # Return base64 for "base64" input
            vec = [3.0, 4.0] + [0.0] * 6
            raw = struct.pack("8f", *vec)
            return httpx.Response(200, json={
                "data": [{"embedding": base64.b64encode(raw).decode(), "index": 0}]
            })
        else:
            # Return list for other input
            return httpx.Response(200, json={
                "data": [{"embedding": [3.0, 4.0] + [0.0] * 6, "index": 0}]
            })
    
    emb = ApiEmbedder(
        api_key="test", model="m", dimension=8,
        base_url="https://example.com/v1",
        transport=httpx.MockTransport(handler),
    )
    
    # Test base64 path
    result_b64 = emb.embed("base64")
    assert len(result_b64) == 8
    assert abs(result_b64[0] - 0.6) < 1e-6  # 3/5 after normalize
    
    # Test list path
    result_list = emb.embed("list")
    assert len(result_list) == 8
    assert abs(result_list[0] - 0.6) < 1e-6
```

- [ ] **Step 7: Run normalization test**

Run: `uv run pytest tests/uxflow_embed/test_api.py::test_normalization_and_dimension_adjustment -v`

Expected: PASS

- [ ] **Step 8: Run all api tests**

Run: `uv run pytest tests/uxflow_embed/test_api.py -v`

Expected: All tests PASS

- [ ] **Step 9: Commit**

```bash
git add src/uxflow_embed/api.py tests/uxflow_embed/test_api.py
git commit -m "feat(embed): add base64 encoding support to ApiEmbedder

Requests 'encoding_format: base64' for 3.7× size reduction.
Falls back to list format for endpoints that don't support it.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 6: ApiEmbedder — Add Batch Size and Dimension Support

**Files:**
- Modify: `src/uxflow_embed/api.py`
- Test: `tests/uxflow_embed/test_api.py`

- [ ] **Step 1: Write failing test for preferred_batch_size**

```python
# Add to tests/uxflow_embed/test_api.py

def test_preferred_batch_size_from_constructor():
    """ApiEmbedder exposes batch size from constructor."""
    from uxflow_embed import ApiEmbedder
    
    emb = ApiEmbedder(
        api_key="test", model="m", dimension=8,
        preferred_batch_size=64,
    )
    assert emb.preferred_batch_size == 64
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/uxflow_embed/test_api.py::test_preferred_batch_size_from_constructor -v`

Expected: FAIL with "unexpected keyword argument 'preferred_batch_size'"

- [ ] **Step 3: Add preferred_batch_size parameter and property**

```python
# In src/uxflow_embed/api.py, update __init__ and add property

def __init__(
    self,
    *,
    model: str,
    dimension: int,
    api_key: str | None = None,
    base_url: str = "https://api.openai.com/v1",
    timeout: float = 25.0,
    max_tries: int = 5,
    retry_delay: float = 1.0,
    preferred_batch_size: int = 32,
    transport: httpx.BaseTransport | None = None,
):
    key = api_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise ValueError("ApiEmbedder needs an api_key or the OPENAI_API_KEY env var.")
    self._model = model
    self._dimension = dimension
    self._url = base_url.rstrip("/") + "/embeddings"
    self._timeout = timeout
    self._max_tries = max_tries
    self._retry_delay = retry_delay
    self._preferred_batch_size = preferred_batch_size
    self._client = httpx.Client(
        headers={"Authorization": f"Bearer {key}"},
        timeout=timeout,
        transport=transport,
    )

@property
def preferred_batch_size(self) -> int:
    """Optimal batch size balancing throughput vs response size for network constraints."""
    return self._preferred_batch_size
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/uxflow_embed/test_api.py::test_preferred_batch_size_from_constructor -v`

Expected: PASS

- [ ] **Step 5: Write test for dimensions parameter support**

```python
# Add to tests/uxflow_embed/test_api.py

def test_dimensions_parameter_sent_when_supported():
    """If model supports native dimension reduction, send 'dimensions' param."""
    import httpx
    from uxflow_embed import ApiEmbedder
    
    received_body = {}
    def handler(request):
        import json
        received_body.update(json.loads(request.content))
        # Return smaller dimension as requested
        dim = received_body.get("dimensions", 8)
        return httpx.Response(200, json={
            "data": [{"embedding": [0.707] * dim, "index": 0}]
        })
    
    emb = ApiEmbedder(
        api_key="test",
        model="text-embedding-3-large",
        dimension=1024,
        base_url="https://example.com/v1",
        transport=httpx.MockTransport(handler),
    )
    result = emb.embed("hello")
    
    # Should request dimension=1024 from API
    assert received_body.get("dimensions") == 1024
    assert len(result) == 1024
```

- [ ] **Step 6: Run test to verify it fails**

Run: `uv run pytest tests/uxflow_embed/test_api.py::test_dimensions_parameter_sent_when_supported -v`

Expected: FAIL (dimensions not sent yet)

- [ ] **Step 7: Add dimensions parameter to request**

```python
# In src/uxflow_embed/api.py, update embed_batch request

def embed_batch(self, texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    
    import time
    last_exc = None
    for attempt in range(1, self._max_tries + 1):
        try:
            payload = {
                "model": self._model,
                "input": texts,
                "encoding_format": "base64",
                "dimensions": self._dimension,
            }
            resp = self._client.post(self._url, json=payload)
            resp.raise_for_status()
            rows = sorted(resp.json()["data"], key=lambda d: d["index"])
            return [self._decode_embedding(r["embedding"]) for r in rows]
        except (httpx.TimeoutException, httpx.HTTPStatusError) as e:
            if isinstance(e, httpx.HTTPStatusError):
                if 400 <= e.response.status_code < 500 and e.response.status_code != 429:
                    raise
            last_exc = e
            if attempt < self._max_tries:
                time.sleep(self._retry_delay)
    
    raise last_exc
```

- [ ] **Step 8: Run dimensions test**

Run: `uv run pytest tests/uxflow_embed/test_api.py::test_dimensions_parameter_sent_when_supported -v`

Expected: PASS

- [ ] **Step 9: Run all api tests**

Run: `uv run pytest tests/uxflow_embed/test_api.py -v`

Expected: All tests PASS

- [ ] **Step 10: Commit**

```bash
git add src/uxflow_embed/api.py tests/uxflow_embed/test_api.py
git commit -m "feat(embed): add preferred_batch_size and dimensions param to ApiEmbedder

- Default batch 32 for network constraints
- Send 'dimensions' param for native model reduction (better than truncation)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 7: Runtime — Wire LITELLM Credentials and Env Vars

**Files:**
- Modify: `src/uxflow_runtime.py:36-76`
- Test: `tests/test_runtime_wiring.py`

- [ ] **Step 1: Write failing test for LITELLM credential priority**

```python
# Add to tests/test_runtime_wiring.py

def test_api_backend_prefers_litellm_credentials(monkeypatch):
    """api backend should read LITELLM_BASE/KEY before falling back to OPENAI_API_KEY."""
    from uxflow_runtime import make_embedder
    
    monkeypatch.setenv("UXFLOW_EMBED_BACKEND", "api")
    monkeypatch.setenv("LITELLM_BASE", "http://lit.local/v1")
    monkeypatch.setenv("LITELLM_KEY", "lit-key")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    
    emb = make_embedder()
    # Should use LITELLM credentials, not OPENAI
    assert "lit.local" in emb._url
    assert "lit-key" in emb._client.headers["Authorization"]


def test_api_backend_falls_back_to_openai_key(monkeypatch):
    """If LITELLM vars missing, fall back to OPENAI_API_KEY."""
    from uxflow_runtime import make_embedder
    
    monkeypatch.setenv("UXFLOW_EMBED_BACKEND", "api")
    monkeypatch.delenv("LITELLM_BASE", raising=False)
    monkeypatch.delenv("LITELLM_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "openai-fallback")
    
    emb = make_embedder()
    assert "openai-fallback" in emb._client.headers["Authorization"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_runtime_wiring.py::test_api_backend_prefers_litellm_credentials -v`

Expected: FAIL (LITELLM not wired yet)

- [ ] **Step 3: Update make_embedder api branch to read LITELLM first**

```python
# In src/uxflow_runtime.py, update the api branch of make_embedder

if choice == "api":
    from uxflow_embed import ApiEmbedder
    
    # Prefer LITELLM credentials, fall back to OPENAI
    api_key = os.environ.get("LITELLM_KEY") or os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("LITELLM_BASE") or os.environ.get("UXFLOW_EMBED_API_BASE", "https://api.openai.com/v1")
    
    model = os.environ.get("UXFLOW_EMBED_API_MODEL", "text-embedding-3-large")
    dimension = int(os.environ.get("UXFLOW_EMBED_API_DIM", "3072"))
    
    # Tunable params for network constraints
    timeout = float(os.environ.get("UXFLOW_EMBED_TIMEOUT", "25"))
    max_tries = int(os.environ.get("UXFLOW_EMBED_MAX_TRIES", "5"))
    batch_size = int(os.environ.get("UXFLOW_EMBED_BATCH", "32"))
    
    try:
        return ApiEmbedder(
            model=model,
            dimension=dimension,
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_tries=max_tries,
            preferred_batch_size=batch_size,
        )
    except ValueError as e:
        raise SystemExit(
            f"{_EMBED_ENV}=api needs LITELLM_KEY or OPENAI_API_KEY set (see .env.example).\n"
            f"  underlying error: {e}"
        ) from e
```

- [ ] **Step 4: Run LITELLM test**

Run: `uv run pytest tests/test_runtime_wiring.py::test_api_backend_prefers_litellm_credentials -v`

Expected: PASS

- [ ] **Step 5: Run fallback test**

Run: `uv run pytest tests/test_runtime_wiring.py::test_api_backend_falls_back_to_openai_key -v`

Expected: PASS

- [ ] **Step 6: Write test for env var tunables**

```python
# Add to tests/test_runtime_wiring.py

def test_api_backend_respects_tunable_env_vars(monkeypatch):
    """Batch/timeout/retries should be configurable via env."""
    from uxflow_runtime import make_embedder
    
    monkeypatch.setenv("UXFLOW_EMBED_BACKEND", "api")
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    monkeypatch.setenv("UXFLOW_EMBED_BATCH", "64")
    monkeypatch.setenv("UXFLOW_EMBED_TIMEOUT", "10")
    monkeypatch.setenv("UXFLOW_EMBED_MAX_TRIES", "3")
    
    emb = make_embedder()
    assert emb.preferred_batch_size == 64
    assert emb._timeout == 10.0
    assert emb._max_tries == 3
```

- [ ] **Step 7: Run tunables test**

Run: `uv run pytest tests/test_runtime_wiring.py::test_api_backend_respects_tunable_env_vars -v`

Expected: PASS

- [ ] **Step 8: Run all runtime wiring tests**

Run: `uv run pytest tests/test_runtime_wiring.py -v`

Expected: All tests PASS

- [ ] **Step 9: Commit**

```bash
git add src/uxflow_runtime.py tests/test_runtime_wiring.py
git commit -m "feat(runtime): wire LITELLM credentials and tunable env vars for api backend

Priority: LITELLM_KEY/BASE > OPENAI_API_KEY/UXFLOW_EMBED_API_BASE
Defaults: text-embedding-3-large, 3072 dim, batch 32, 25s timeout, 5 tries
Tunable: UXFLOW_EMBED_BATCH/TIMEOUT/MAX_TRIES

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 8: Pipeline — Use preferred_batch_size

**Files:**
- Modify: `src/module1/pipeline.py:261-262`
- No new tests (covered by existing pipeline tests)

- [ ] **Step 1: Replace the hardcoded chunk constant with the embedder's preference**

Find this block in `ingest_trajectories` (currently at line 261):

```python
            _report("embedding", 0, len(texts))
            for start in range(0, len(texts), _EMBED_CHUNK):
                chunk = texts[start:start + _EMBED_CHUNK]
```

Replace with:

```python
            _report("embedding", 0, len(texts))
            # 后端自报最优批次：Api 受网关响应体上限约束（32≈0.5MB），Local 与其
            # 内部 encode batch_size 对齐（32）。getattr 带默认值 → 第三方实现
            # 不实现该属性也能跑，退回 _EMBED_CHUNK。
            chunk_size = getattr(emb_model, "preferred_batch_size", _EMBED_CHUNK)
            for start in range(0, len(texts), chunk_size):
                chunk = texts[start:start + chunk_size]
```

- [ ] **Step 2: Update the stale comment above the loop**

The comment at line 258-260 references `_EMBED_CHUNK` by name. Replace:

```python
            # 先报一条 0/N：向量化是整条链路最慢的一段，而进度只在 chunk 边界更新。
            # 切片数 <= _EMBED_CHUNK 时只有一个 chunk，不先发这条，界面会一直停在
            # "切片 N/N" 直到整批算完 —— 用户看到的是"卡在切片"，实际在跑向量化。
```

With:

```python
            # 先报一条 0/N：向量化是整条链路最慢的一段，而进度只在 chunk 边界更新。
            # 切片数 <= chunk_size 时只有一个 chunk，不先发这条，界面会一直停在
            # "切片 N/N" 直到整批算完 —— 用户看到的是"卡在切片"，实际在跑向量化。
```

- [ ] **Step 3: Update the docstring reference**

At line 213 the docstring says `_EMBED_CHUNK 分块 embed_batch 回填。模型调用数 N → ceil(N/64)`. The `64` was already stale. Replace that line with:

```python
        embedding 走批量：先把全部切片的签名算出来（纯 CPU，不含向量），再按
        emb_model.preferred_batch_size 分块 embed_batch 回填。模型调用数
        N → ceil(N/batch)，这是本路径最贵的一环（每次入库都付，非一次性成本）。
```

- [ ] **Step 4: Run module1 tests**

Run: `uv run pytest tests/module1/ -v`

Expected: All tests PASS. FakeEmbedder now reports 128, so chunking changes from 32→128; the count-mismatch guard and progress reporting must still hold.

- [ ] **Step 5: Run the full suite for regressions**

Run: `uv run pytest tests/ -q`

Expected: All tests PASS

- [ ] **Step 6: Commit**

```bash
git add src/module1/pipeline.py
git commit -m "feat(pipeline): respect embedder preferred_batch_size

Chunk size now comes from the embedder rather than a module constant:
ApiEmbedder needs 32 to stay under a 1MB gateway limit, LocalEmbedder
wants 32 to match its internal encode batch. getattr with _EMBED_CHUNK
as default keeps third-party embedders working unchanged.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 9: Documentation — Update .env.example

**Files:**
- Modify: `.env.example`

- [ ] **Step 1: Update embedding backend section**

```bash
# In .env.example, replace the UXFLOW_EMBED_* section with:

# ─────────────────────────────────────────────────────────────────────────────
# Embedding backend
# ─────────────────────────────────────────────────────────────────────────────
# fake   — deterministic, no ML deps (default)
# local  — Qwen3-Embedding-0.6B via sentence-transformers (needs local-embed extra)
# api    — OpenAI-compatible endpoint (needs LITELLM_KEY or OPENAI_API_KEY)
UXFLOW_EMBED_BACKEND=fake

# api backend config (only used when UXFLOW_EMBED_BACKEND=api)
# Credentials: LITELLM_KEY/BASE preferred, falls back to OPENAI_API_KEY
# UXFLOW_EMBED_API_MODEL=text-embedding-3-large
# UXFLOW_EMBED_API_DIM=3072

# Network tuning for restricted environments:
# UXFLOW_EMBED_BATCH=32           # batch size (default 32 ≈ 0.5MB response)
# UXFLOW_EMBED_TIMEOUT=25         # timeout per request in seconds
# UXFLOW_EMBED_MAX_TRIES=5        # max retry attempts (timeout/5xx/429 only)

# Response size reference (3072 dim + base64):
#   batch 16 → 0.26 MB    batch 64 → 1.03 MB
#   batch 32 → 0.51 MB    batch 128 → 2.06 MB
# If your gateway limit is 1 MB, keep batch ≤ 32.
```

- [ ] **Step 2: Verify .env.example parses**

Run: `grep "UXFLOW_EMBED" .env.example | grep -v "^#" | head -3`

Expected: Shows `UXFLOW_EMBED_BACKEND=fake` and commented tunables

- [ ] **Step 3: Run env example test**

Run: `uv run pytest tests/test_runtime_wiring.py::test_env_example_documents_embed_backend -v`

Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add .env.example
git commit -m "docs: update .env.example with api backend and network tuning

Document LITELLM credential priority, batch/timeout/retries tunables,
and response size reference for gateway-constrained networks.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 10: Documentation — Update README.md

**Files:**
- Modify: `README.md:240-260`

- [ ] **Step 1: Update embedding backends section**

```markdown
# In README.md, replace the "### Embedding backends" section:

### Embedding backends

Embedding is pluggable behind a single `Embedder` protocol in `uxflow_embed`:

- **`fake`** (default) — deterministic, no ML dependencies. Vectors carry no semantics.
- **`local`** — Qwen3-Embedding-0.6B via sentence-transformers (1024 dim). Needs the `local-embed` extra; downloads the model on first use.
- **`api`** — any OpenAI-compatible embedding endpoint. Reads credentials from `LITELLM_KEY`/`LITELLM_BASE` (preferred) or `OPENAI_API_KEY` (fallback). Defaults to `text-embedding-3-large` (3072 dim).

Configure with `UXFLOW_EMBED_BACKEND=fake|local|api`.

**Network tuning:** If behind a restrictive gateway (response size limits, high latency), tune `UXFLOW_EMBED_BATCH` (default 32, ~0.5MB response for 3072 dim), `UXFLOW_EMBED_TIMEOUT` (default 25s), and `UXFLOW_EMBED_MAX_TRIES` (default 5). See `.env.example` for response size reference table.

**Quality vs speed:** `text-embedding-3-large` at its native 3072 dimensions is the default. Measured against this proxy on 512-text batches, it was not slower than `-small` (median 57.5 vs 52.5 texts/s over 6 interleaved rounds) — batch latency is dominated by proxy queuing, not model size, so the larger model's quality costs nothing here. Truncating to fewer dimensions does cost recall: on 400 real trajectory slices, top-20 overlap against full 3072 was 95.5% at 1536, 91.9% at 1024, and 87.0% at 512.
```

Do not add a claim comparing `-small`'s recall quality — only its throughput
was measured, never its cosine distribution or overlap.

- [ ] **Step 2: Verify markdown renders correctly**

Run: `grep -A 8 "### Embedding backends" README.md`

Expected: Shows updated content with api backend details

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs(readme): update embedding backends section for api backend

Document LITELLM credential priority, network tuning vars, and
quality comparison (large vs small).

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 11: Documentation — Update README.zh-CN.md

**Files:**
- Modify: `README.zh-CN.md:240-260`

- [ ] **Step 1: Update Chinese embedding backends section**

```markdown
# In README.zh-CN.md, replace the "### 嵌入后端" section:

### 嵌入后端

嵌入能力通过 `uxflow_embed` 中统一的 `Embedder` 协议实现可插拔:

- **`fake`**（默认）—— 确定性输出，无 ML 依赖。向量不带语义。
- **`local`** —— 经 sentence-transformers 跑 Qwen3-Embedding-0.6B（1024 维）。需要 `local-embed` extra，首次使用时下载模型。
- **`api`** —— 任何 OpenAI 兼容的嵌入端点。从 `LITELLM_KEY`/`LITELLM_BASE`（优先）或 `OPENAI_API_KEY`（回退）读凭证。默认 `text-embedding-3-large`（3072 维）。

用 `UXFLOW_EMBED_BACKEND=fake|local|api` 配置。

**网络调优:** 若身处受限网关环境（响应体大小限制、高延迟），可调 `UXFLOW_EMBED_BATCH`（默认 32，3072 维下响应约 0.5MB）、`UXFLOW_EMBED_TIMEOUT`（默认 25 秒）、`UXFLOW_EMBED_MAX_TRIES`（默认 5 次）。响应体大小参照表见 `.env.example`。

**质量 vs 速度:** 默认用 `text-embedding-3-large` 的原生 3072 维。在本代理上以 512 条批次实测，它并不比 `-small` 慢（6 轮交替测试中位吞吐 57.5 vs 52.5 texts/s）—— 批量延迟主要由代理排队决定，而非模型大小，所以更大的模型在这里不付速度代价。但降维确实损失召回：400 条真实切片上，相对完整 3072 维的 top-20 重叠率，1536 维为 95.5%，1024 维为 91.9%，512 维为 87.0%。
```

不要写 `-small` 召回质量更差之类的结论 —— 只测过它的吞吐，从未测过它的余弦分布或重叠率。

- [ ] **Step 2: Verify Chinese markdown**

Run: `grep -A 8 "### 嵌入后端" README.zh-CN.md`

Expected: Shows updated Chinese content

- [ ] **Step 3: Commit**

```bash
git add README.zh-CN.md
git commit -m "docs(readme): update Chinese embedding backends section for api backend

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 12: Integration Test — End-to-End with Real API

**Files:**
- Create: `tests/integration/test_api_embedder_e2e.py`

- [ ] **Step 1: Write end-to-end test (skipped in CI)**

```python
# Create tests/integration/test_api_embedder_e2e.py

# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 dengxuanliang
"""End-to-end test for API embedder with real credentials.

Skipped in CI (needs LITELLM_KEY). Run manually to verify:
  pytest tests/integration/test_api_embedder_e2e.py -v -s
"""
import os
import pytest


@pytest.mark.skipif(
    not os.environ.get("LITELLM_KEY"),
    reason="Needs LITELLM_KEY for real API call"
)
def test_api_embedder_with_real_endpoint():
    """Smoke test: embed 3 texts via real API, verify dimension and norm."""
    from uxflow_runtime import make_embedder
    import os
    
    # Force api backend
    prev = os.environ.get("UXFLOW_EMBED_BACKEND")
    os.environ["UXFLOW_EMBED_BACKEND"] = "api"
    
    try:
        emb = make_embedder()
        
        texts = [
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning models process embeddings.",
            "SQLite stores vectors as BLOB columns.",
        ]
        
        # Test single embed
        vec1 = emb.embed(texts[0])
        assert len(vec1) == emb.dimension
        norm1 = sum(x*x for x in vec1) ** 0.5
        assert 0.99 < norm1 < 1.01, f"not L2-normalized: {norm1}"
        
        # Test batch embed
        vecs = emb.embed_batch(texts)
        assert len(vecs) == 3
        for vec in vecs:
            assert len(vec) == emb.dimension
            norm = sum(x*x for x in vec) ** 0.5
            assert 0.99 < norm < 1.01
        
        # Verify batch and single produce same vector for same text
        vec1_batch = vecs[0]
        for a, b in zip(vec1, vec1_batch):
            assert abs(a - b) < 1e-5, "single and batch diverge"
        
        print(f"\n✓ API embedder working: {emb.dimension} dim, batch {emb.preferred_batch_size}")
    
    finally:
        if prev is None:
            os.environ.pop("UXFLOW_EMBED_BACKEND", None)
        else:
            os.environ["UXFLOW_EMBED_BACKEND"] = prev
```

- [ ] **Step 2: Confirm the integration package already exists**

Run: `ls tests/integration/__init__.py`

Expected: File exists (the package is already present — do NOT create it)

- [ ] **Step 3: Run test with real credentials (manual)**

Run: `uv run pytest tests/integration/test_api_embedder_e2e.py -v -s`

Expected: SKIP in CI (no LITELLM_KEY), or PASS if run manually with credentials

- [ ] **Step 4: Commit**

```bash
git add tests/integration/test_api_embedder_e2e.py
git commit -m "test: add end-to-end API embedder integration test

Skipped without LITELLM_KEY so CI stays hermetic; run manually to
verify real API calls, dimension, and L2 normalization.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 13: Final Verification — Run Full Test Suite

**Files:**
- All modified files

- [ ] **Step 1: Run full test suite**

Run: `uv run pytest tests/ -q`

Expected: All tests PASS (integration test skipped without LITELLM_KEY)

- [ ] **Step 2: Run the linter**

Run: `uv run ruff check src/uxflow_embed/ src/uxflow_runtime.py src/module1/pipeline.py`

Expected: No new violations

Note: this project has no mypy and no pytest-cov configured (`dev` extra is
`pytest`, `pytest-asyncio`, `ruff` only). Do not invent those steps.

- [ ] **Step 3: Verify the local backend did not regress**

Run: `uv run pytest tests/uxflow_embed/test_local.py tests/module1/ -q`

Expected: All PASS — `local` keeps batch 32, identical to the previous `_EMBED_CHUNK` behaviour

- [ ] **Step 4: Commit any formatting fixes**

```bash
git add -u
git commit -m "chore: ruff pass"
```

---

### Task 14: Manual Verification — Real Ingest and Throughput

**Files:**
- None (verification only)

**Note:** `scripts/inspector_serve.py` takes NO CLI arguments — it is a uvicorn
web server, and ingest happens through file upload in the browser UI. Do not
invent an `--ingest` flag. The script below drives `ingest_trajectories()`
directly so throughput can be timed without the UI.

- [ ] **Step 1: Point at a scratch database and the api backend**

```bash
cd /Users/deng/开发/UXFlow
export UXFLOW_EMBED_BACKEND=api
export UXFLOW_DB=/tmp/uxflow_api_test.db
rm -f /tmp/uxflow_api_test.db
```

`LITELLM_BASE` / `LITELLM_KEY` are already in `.env`; the script below loads it.
The scratch DB keeps the real `~/.local/share/uxflow/uxflow.db` (1132 Qwen
vectors, 1024-dim) untouched, per spec §4.7.

- [ ] **Step 2: Run a timed ingest against real trajectories**

```bash
uv run python - <<'PY'
import pathlib, time
from dotenv import load_dotenv
load_dotenv(pathlib.Path(".env"))

from uxflow_runtime import make_embedder, backend_banner
from module1.pipeline import TrajectoryPipeline, PipelineConfig

print(backend_banner())
emb = make_embedder()
print(f"backend={type(emb).__name__} dim={emb.dimension} "
      f"batch={getattr(emb, 'preferred_batch_size', 'n/a')}")

paths = [pathlib.Path("dataset/swe-chat-reasoning-sample-50.jsonl")]
assert all(p.exists() for p in paths), f"missing: {[str(p) for p in paths if not p.exists()]}"
print(f"ingesting {len(paths)} file(s)")

pipe = TrajectoryPipeline(config=PipelineConfig(embedding_model=emb), gateway=None)

seen = {}
def on_progress(phase, done, total):
    if phase not in seen:
        seen[phase] = time.time()
        print(f"  [{phase}] start, total={total}")
    if phase == "embedding" and done == total:
        print(f"  [embedding] {total} texts in {time.time()-seen[phase]:.1f}s "
              f"= {total/max(time.time()-seen[phase], 1e-9):.1f} texts/s")

t0 = time.time()
pipe.ingest_trajectories([str(p) for p in paths], on_progress=on_progress)
print(f"total ingest wall time: {time.time()-t0:.1f}s")
PY
```

Expected:
- Banner prints `✓ LLM: real   ✓ Embedding: api`
- `backend=ApiEmbedder dim=3072 batch=32`
- Embedding phase advances in steps of 32
- No unhandled timeout — retries absorb the ~30% proxy queuing failures
- A `texts/s` number is printed

- [ ] **Step 3: Record the measured throughput**

Write the printed `texts/s` down. The spec's 21.5 texts/s was measured at
batch 512 / 1024-dim / JSON encoding, so **expect a different (likely lower)
number** at batch 32 / 3072-dim / base64. This step exists to replace that
stale figure with a real one, not to confirm it.

- [ ] **Step 4: Verify the vectors landed at the right dimension**

```bash
sqlite3 /tmp/uxflow_api_test.db \
  "SELECT length(embedding)/4 AS dim, COUNT(*) FROM signatures
   WHERE embedding IS NOT NULL GROUP BY 1;"
```

Expected: a single row `3072|<n>` — one dimension only, 3072, no mixing

- [ ] **Step 5: Confirm response bodies stayed under the 1MB gateway limit**

```bash
uv run python - <<'PY'
import os, pathlib, time, json, httpx
from dotenv import load_dotenv
load_dotenv(pathlib.Path(".env"))
base = os.environ["LITELLM_BASE"].rstrip("/"); key = os.environ["LITELLM_KEY"]
texts = [f"gateway size probe {i}" for i in range(32)]
r = httpx.post(base + "/embeddings",
               headers={"Authorization": f"Bearer {key}"},
               json={"model": "text-embedding-3-large", "input": texts,
                     "encoding_format": "base64", "dimensions": 3072},
               timeout=60.0)
mb = len(r.content) / 1024 / 1024
print(f"batch 32 response = {mb:.2f} MB  ({mb/1.0*100:.0f}% of the 1MB limit)")
assert mb < 1.0, "response exceeds the gateway limit — lower UXFLOW_EMBED_BATCH"
PY
```

Expected: `~0.51 MB (51% of the 1MB limit)`, assertion passes

- [ ] **Step 6: Sanity-check recall quality**

```bash
uv run python - <<'PY'
import pathlib
from dotenv import load_dotenv
load_dotenv(pathlib.Path(".env"))
from uxflow_runtime import make_embedder
from module1.sqlite_store import SqliteSliceStore

emb = make_embedder()
store = SqliteSliceStore("/tmp/uxflow_api_test.db")
q = emb.embed_batch(["fix a syntax error in a python file"])
hits = store.recall(structured_filters={}, keywords=["python", "error"],
                    query_embeddings=q, top_n=5)
for h in hits:
    print(f"  {h.signature.trajectory_id} slice={h.signature.slice_index}")
assert hits, "recall returned nothing — vectors or filters are broken"
PY
```

Expected: 5 hits printed, and they should look topically related to the query
rather than arbitrary. Semantically meaningless results mean the vectors are
wrong even though the dimension check passed.

- [ ] **Step 7: Clean up the scratch database**

```bash
rm -f /tmp/uxflow_api_test.db
unset UXFLOW_DB UXFLOW_EMBED_BACKEND
```

---

### Task 15: Threshold Calibration Data (spec §4.8)

The spec commits to *measuring* the cosine distribution under API vectors and
recording it, while explicitly **not** changing `mount_threshold=0.50` or
`UXFLOW_QUESTION_DEDUP_THRESHOLD=0.90` this round. This task produces that data.

**Files:**
- Create: `scripts/calibrate_thresholds_api.py`

- [ ] **Step 1: Write the calibration script**

```python
# Create scripts/calibrate_thresholds_api.py

"""Measure cosine distribution under the ACTIVE embedding backend.

The existing calibrate_mount_threshold.py hardcodes the local Qwen model
(module0.embedding.EmbeddingModel). This one goes through make_embedder(),
so it reports whatever backend UXFLOW_EMBED_BACKEND selects.

Run:  UXFLOW_EMBED_BACKEND=api uv run python scripts/calibrate_thresholds_api.py
"""
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 dengxuanliang
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))

import numpy as np
from dotenv import load_dotenv

load_dotenv(pathlib.Path(__file__).parent.parent / ".env")

from uxflow_runtime import make_embedder  # noqa: E402

# Child label -> its intended parent. Same pairs as calibrate_mount_threshold.py
# so the two backends' numbers are directly comparable.
PAIRS = [
    ("修复运行时抛出的异常，如 TypeError/KeyError/ValueError", "从错误中恢复并修复问题"),
    ("修复导入模块失败的问题", "从错误中恢复并修复问题"),
    ("为函数补充单元测试", "验证代码正确性"),
]

# Pairs that must NOT mount — establishes the noise floor.
UNRELATED = [
    ("为函数补充单元测试", "从错误中恢复并修复问题"),
    ("修复导入模块失败的问题", "验证代码正确性"),
]


def main():
    emb = make_embedder()
    print(f"backend={type(emb).__name__} dim={emb.dimension}\n")

    def cos(a, b):
        va, vb = np.asarray(a), np.asarray(b)
        return float(va @ vb / (np.linalg.norm(va) * np.linalg.norm(vb)))

    true_sims = []
    print("TRUE child->parent pairs (these must mount):")
    for child, parent in PAIRS:
        s = cos(emb.embed(child), emb.embed(parent))
        true_sims.append(s)
        print(f"  cos={s:.3f}  {child[:24]!r} -> {parent[:20]!r}")

    noise_sims = []
    print("\nUNRELATED pairs (these must NOT mount):")
    for child, parent in UNRELATED:
        s = cos(emb.embed(child), emb.embed(parent))
        noise_sims.append(s)
        print(f"  cos={s:.3f}  {child[:24]!r} -> {parent[:20]!r}")

    lo, hi = min(true_sims), max(noise_sims)
    print(f"\ntrue  min={lo:.3f}  max={max(true_sims):.3f}")
    print(f"noise min={min(noise_sims):.3f}  max={hi:.3f}")
    print(f"separation gap = {lo - hi:+.3f}")
    if lo > hi:
        print(f"→ any mount_threshold in ({hi:.3f}, {lo:.3f}) separates them; "
              f"midpoint {(lo + hi) / 2:.3f}")
    else:
        print("→ NO clean separation — true pairs overlap noise. "
              "mount_threshold cannot be set from these pairs alone.")
    print(f"\ncurrent default mount_threshold=0.50 "
          f"({'OK' if hi < 0.50 < lo else 'MISCALIBRATED for this backend'})")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it against the fake backend as a smoke test**

Run: `UXFLOW_EMBED_BACKEND=fake uv run python scripts/calibrate_thresholds_api.py`

Expected: Runs without error. Numbers will be meaningless (hash vectors ≈
orthogonal, all cosines near 0) — this only proves the script executes.

- [ ] **Step 3: Run it against the API backend to collect the real data**

Run: `UXFLOW_EMBED_BACKEND=api uv run python scripts/calibrate_thresholds_api.py`

Expected: Prints true-pair and noise-pair cosines under `text-embedding-3-large`
at 3072 dims, plus whether 0.50 still separates them.

- [ ] **Step 4: Record the output in the spec**

Append the printed numbers to spec §4.8 as a new subsection titled
"API 向量下的实测分布（本轮采集，未据此改阈值）". Do **not** change
`mount_threshold` in `src/module0_5/evolution.py` — the spec defers that.

- [ ] **Step 5: Commit**

```bash
git add scripts/calibrate_thresholds_api.py docs/superpowers/specs/2026-08-12-api-embedder-switch-design.md
git commit -m "feat(scripts): calibrate thresholds against the active embed backend

The existing calibrate_mount_threshold.py hardcodes local Qwen, so it
cannot say anything about API vectors. This one goes through
make_embedder() and also measures unrelated pairs, so the separation
gap is visible rather than inferred from true pairs alone.

Records the numbers in the spec; does not change mount_threshold —
retuning is deliberately a separate round.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 16: Final Commit and Branch Push

**Files:**
- All

- [ ] **Step 1: Review all changes**

Run: `git diff origin/main...HEAD --stat`

Expected: Shows modified files matching the File Structure section, plus
`scripts/calibrate_thresholds_api.py` and
`tests/integration/test_api_embedder_e2e.py`

- [ ] **Step 2: Check commit history**

Run: `git log origin/main..HEAD --oneline`

Expected: One clean commit per task, descriptive messages

- [ ] **Step 3: Confirm no stray files**

Run: `git status --short`

Expected: Clean. In particular `uv.lock` must NOT appear — if a Tsinghua-mirror
sync rewrote its URLs, revert it with `git checkout uv.lock` before pushing.

- [ ] **Step 4: Push branch**

```bash
git push -u origin feat/api-embedder-switch
```

- [ ] **Step 5: Open the PR with measured results**

The PR body must state the throughput measured in Task 14 Step 3, the response
size from Task 14 Step 5, and the calibration output from Task 15 Step 3.
Do not reuse the spec's 21.5 texts/s — that was measured under a different
configuration (batch 512 / 1024-dim / JSON).

---

## Verification Checklist

After completing all tasks:

- [ ] `UXFLOW_EMBED_BACKEND=api` with only `LITELLM_BASE`/`LITELLM_KEY` starts successfully
- [ ] `ApiEmbedder` returns 3072-dim L2-normalized vectors
- [ ] base64 and list response formats both decode correctly
- [ ] Timeout/5xx/429 trigger retry; 4xx fails fast
- [ ] Batch size defaults to 32 and respects `UXFLOW_EMBED_BATCH`
- [ ] `UXFLOW_EMBED_TIMEOUT` / `UXFLOW_EMBED_MAX_TRIES` take effect
- [ ] `local` and `fake` backends unchanged (zero regression)
- [ ] Full suite passes via `uv run pytest tests/ -q`
- [ ] Measured batch-32 response body is under 1 MB (Task 14 Step 5 asserts this)
- [ ] Real ingest completes and a throughput number is recorded
- [ ] Calibration data collected and written into spec §4.8

---

## Notes

- **No SQLite migration needed** — the embedding column is a dimension-agnostic BLOB
- **Threshold values unchanged this round** — Task 15 only *measures*; retuning
  `mount_threshold` / `UXFLOW_QUESTION_DEDUP_THRESHOLD` is deliberately deferred
- **Fresh database required** — API vectors are 3072-dim and semantically
  incompatible with the existing 1024-dim Qwen vectors. The `_check_dim` guard
  catches the dimension mismatch, but two same-dimension vectors from different
  models would pass it silently, so a separate DB is the real protection.
- **Local Qwen baseline was never measured** — torch installation timed out
  twice. No claim about "N× faster than local" is supported; do not add one.
- **The spec's 21.5 texts/s does not apply** — it was measured at batch 512 /
  1024-dim / JSON encoding. Task 14 replaces it with a figure from the shipped
  configuration.
- **Gateway limit is user-reported (~1 MB), not independently probed** — batch 32
  leaves roughly 2× headroom, which absorbs a reasonable error in that figure.
