"""ApiEmbedder — OpenAI-compatible /v1/embeddings backend (zero ML deps).

Works with any provider exposing the OpenAI embeddings format (OpenAI,
Qwen/DashScope compatible endpoint, vLLM, Ollama, ...). API key is read
from the api_key arg or the OPENAI_API_KEY env var.
"""
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 dengxuanliang

from __future__ import annotations

import base64
import binascii
import os
import struct
import time

import httpx
import numpy as np

__all__ = ["ApiEmbedder"]


class ApiEmbedder:
    """Call an OpenAI-compatible embeddings endpoint."""

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
        rate_limit_base: float = 2.0,
        rate_limit_cap: float = 32.0,
        preferred_batch_size: int = 32,
        transport: httpx.BaseTransport | None = None,
    ):
        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise ValueError(
                "ApiEmbedder needs an api_key or the OPENAI_API_KEY env var."
            )
        if max_tries < 1:
            raise ValueError("ApiEmbedder needs max_tries >= 1.")
        self._model = model
        self._dimension = dimension
        self._url = base_url.rstrip("/") + "/embeddings"
        self._timeout = timeout
        self._max_tries = max_tries
        self._retry_delay = retry_delay
        self._rate_limit_base = rate_limit_base
        self._rate_limit_cap = rate_limit_cap
        self._preferred_batch_size = preferred_batch_size
        self._client = httpx.Client(
            headers={"Authorization": f"Bearer {key}"},
            timeout=self._timeout,
            transport=transport,
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "ApiEmbedder":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def preferred_batch_size(self) -> int:
        """Batch size that keeps the response under the gateway's size limit."""
        return self._preferred_batch_size

    def embed(self, text: str) -> list[float]:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        # The API rejects empty input ("input[15]: input cannot be an empty
        # string") with a 400, which is deliberately not retried — so one blank
        # slice would abort a whole ingest. Real data does produce them: 19 of
        # 1099 slices from the sample dataset embed to "". Drop them from the
        # request and substitute zero vectors, which recall_core.vector_score
        # already skips via `any(v != 0.0 ...)`. The local backend accepted
        # empty strings, which is why this only surfaced on the API backend.
        keep = [i for i, t in enumerate(texts) if t and t.strip()]
        if not keep:
            return [self._zero_vector() for _ in texts]

        vectors = self._embed_nonempty([texts[i] for i in keep])

        # zip() stops at the shorter iterable, so a short response would leave
        # trailing slots at their pre-filled zero vector — indistinguishable
        # from a legitimately empty text, and permanently unrecallable. The
        # pipeline guards batch length for exactly this reason (「宁可炸掉本次
        # 入库，也不要悄悄写坏索引」), but that guard cannot fire here because
        # embed_batch always returns len(texts). So the check belongs here.
        if len(vectors) != len(keep):
            raise ValueError(
                f"model {self._model!r} at {self._url} returned "
                f"{len(vectors)} embeddings for {len(keep)} non-empty inputs; "
                f"refusing to guess which text each vector belongs to"
            )

        # Reassemble positionally: every input gets exactly one slot back, in
        # the original order. Getting this wrong would silently write the wrong
        # vector to the wrong slice, which is worse than crashing.
        out = [self._zero_vector() for _ in texts]
        for slot, vector in zip(keep, vectors):
            out[slot] = vector
        return out

    def _zero_vector(self) -> list[float]:
        """Sentinel for text that cannot be embedded.

        Not normalized: a zero vector has norm 0, and dividing by it would make
        every component NaN and poison downstream cosine comparisons.
        """
        return [0.0] * self._dimension

    def _embed_nonempty(self, texts: list[str]) -> list[list[float]]:
        """POST one batch, with retries. Every text here is known non-empty."""
        last_exc: Exception | None = None
        for attempt in range(1, self._max_tries + 1):
            try:
                resp = self._client.post(
                    self._url,
                    json={
                        "model": self._model,
                        "input": texts,
                        "encoding_format": "base64",
                        "dimensions": self._dimension,
                    },
                )
                resp.raise_for_status()

                # Parse the envelope, failing explicitly if it's malformed. A 200
                # carrying {"error": ...} is plausible proxy behavior, so the msg
                # should preserve the server's own words for whoever debugs this.
                # Consistent with _decode_embedding, which wraps base64 failures.
                try:
                    data = resp.json()["data"]
                    rows = sorted(data, key=lambda d: d["index"])
                    return [self._decode_embedding(r["embedding"]) for r in rows]
                except (KeyError, TypeError, IndexError) as e:
                    body = resp.text[:200]  # truncate if huge
                    raise ValueError(
                        f"model {self._model!r} at {self._url} returned "
                        f"malformed response: {e!r}. Body preview: {body}"
                    ) from None
            except (httpx.TimeoutException, httpx.HTTPStatusError) as e:
                status = (
                    e.response.status_code
                    if isinstance(e, httpx.HTTPStatusError)
                    else None
                )
                # 4xx means the request itself is wrong (bad key, malformed
                # body) — retrying just re-sends the identical broken request.
                # 429 is the exception: a transient rate limit, retryable.
                if status is not None and 400 <= status < 500 and status != 429:
                    raise
                last_exc = e
                if attempt < self._max_tries:
                    time.sleep(self._retry_wait(e, status, attempt))

        # max_tries >= 1 is enforced in __init__, so the loop always runs at
        # least once and always assigns last_exc before falling through here.
        raise last_exc

    def _retry_wait(
        self, exc: Exception, status: int | None, attempt: int
    ) -> float:
        """How long to wait before retry `attempt`+1.

        Timeouts/5xx and 429 are opposite failure modes and get opposite
        strategies — merging them breaks one or the other.
        """
        if status == 429:
            # A rate limit means the quota window is closed, so the only thing
            # that helps is waiting for it to reopen. Observed against an Azure
            # OpenAI S0 tier: a flat 1s retry knocked on the locked door 5 times
            # in 5 seconds and exhausted every attempt, because those windows
            # run ~60s. Hence exponential, capped so one batch can't hang
            # forever. Prefer a server-stated Retry-After when offered — this
            # endpoint sends none, but others do, and it beats guessing.
            retry_after = self._retry_after_seconds(exc)
            if retry_after is not None:
                return retry_after
            backoff = self._rate_limit_base * 2 ** (attempt - 1)
            return min(backoff, self._rate_limit_cap)

        # Flat delay, deliberately not exponential. This failure mode is
        # proxy-side queuing, not overload: a request already stuck in a slow
        # queue stays slow, while a fresh one usually lands on an idle worker.
        # Measured at n=64, 8 trials each: a 12s timeout with immediate retry
        # got 8/8 in 8.51s mean, while a patient 90s single attempt also got
        # 8/8 but averaged 29.39s. Backing off would only wait out a queue that
        # never speeds up. See design spec §2.
        return self._retry_delay

    @staticmethod
    def _retry_after_seconds(exc: Exception) -> float | None:
        """Parse a Retry-After header, if the response carried a usable one."""
        if not isinstance(exc, httpx.HTTPStatusError):
            return None
        raw = exc.response.headers.get("retry-after")
        if not raw:
            return None
        try:
            # Retry-After may also be an HTTP-date, which float() rejects; fall
            # back to our own backoff rather than guessing at a date format.
            return float(raw.strip())
        except ValueError:
            return None

    def _decode_embedding(self, raw: str | list[float]) -> list[float]:
        """Decode base64 or list embedding, then truncate/pad + L2-normalize.

        Providers that honour encoding_format return a base64 float32 blob; ones
        that ignore it return a plain list. Handle both so the switch is safe.
        """
        if isinstance(raw, str):
            try:
                blob = base64.b64decode(raw)
                count = len(blob) // 4
                # Explicit little-endian: the wire format is fixed, so it must
                # not follow whatever byte order the client host happens to use.
                vec = struct.unpack(f"<{count}f", blob)
            except (binascii.Error, struct.error) as e:
                # Not an httpx error, so it would otherwise skip the retry loop
                # and surface as a bare stdlib traceback naming neither the
                # model nor the endpoint it came from.
                raise ValueError(
                    f"could not decode a base64 embedding from model "
                    f"{self._model!r} at {self._url} ({len(raw)} chars): {e}"
                ) from e
            return self._ensure_dimension(list(vec))
        return self._ensure_dimension(raw)

    def _ensure_dimension(self, raw: list[float]) -> list[float]:
        vec = np.asarray(raw, dtype=np.float32)
        if len(vec) >= self._dimension:
            vec = vec[: self._dimension]
        else:
            vec = np.pad(vec, (0, self._dimension - len(vec)))
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.tolist()
