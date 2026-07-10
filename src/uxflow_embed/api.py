"""ApiEmbedder — OpenAI-compatible /v1/embeddings backend (zero ML deps).

Works with any provider exposing the OpenAI embeddings format (OpenAI,
Qwen/DashScope compatible endpoint, vLLM, Ollama, ...). API key is read
from the api_key arg or the OPENAI_API_KEY env var.
"""

from __future__ import annotations

import os

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
        timeout: float = 30.0,
        transport: httpx.BaseTransport | None = None,
    ):
        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise ValueError(
                "ApiEmbedder needs an api_key or the OPENAI_API_KEY env var."
            )
        self._model = model
        self._dimension = dimension
        self._url = base_url.rstrip("/") + "/embeddings"
        self._client = httpx.Client(
            headers={"Authorization": f"Bearer {key}"},
            timeout=timeout,
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

    def embed(self, text: str) -> list[float]:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        resp = self._client.post(self._url, json={"model": self._model, "input": texts})
        resp.raise_for_status()
        rows = sorted(resp.json()["data"], key=lambda d: d["index"])
        return [self._ensure_dimension(r["embedding"]) for r in rows]

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
