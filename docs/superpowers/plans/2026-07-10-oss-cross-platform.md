# UXFlow 开源 + 跨平台 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 UXFlow 改造为可在 GitHub 上被 clone、跨 Linux/macOS × Py3.11-3.13 跑起来、接受外部 PR 的开源项目（不上 PyPI）。

**Architecture:** 抽出 `Embedder` 协议到独立包 `src/uxflow_embed/`，把 torch/sentence-transformers 降为可选依赖 `[local-embed]`；纯逻辑测试注入 `FakeEmbedder` 从而脱离 ML；补齐 uv 锁定、CI（macOS+Linux）、LICENSE/README/CONTRIBUTING。分三批增量落地，每批可独立验证。

**Tech Stack:** Python 3.11-3.13, hatchling, uv, pytest, ruff, httpx, GitHub Actions, Apache-2.0。

**上游 spec:** `docs/superpowers/specs/2026-07-10-oss-cross-platform-design.md`

---

## 文件结构（决策锁定）

**新增包 `src/uxflow_embed/`**（横切依赖，独立成包避免模块间错误依赖方向）：
- `src/uxflow_embed/__init__.py` — 导出 `Embedder`, `FakeEmbedder`, `LocalEmbedder`, `ApiEmbedder`
- `src/uxflow_embed/protocol.py` — `Embedder` 运行时协议
- `src/uxflow_embed/fake.py` — `FakeEmbedder`（hash-based 确定性，单测用）
- `src/uxflow_embed/local.py` — `LocalEmbedder`（迁移自 `module0/embedding.py`，去 `local_files_only`）
- `src/uxflow_embed/api.py` — `ApiEmbedder`（OpenAI 兼容）

**向后兼容 shim**：`src/module0/embedding.py` 改为 `from uxflow_embed import LocalEmbedder as EmbeddingModel`，保住所有现有 `import`。

**工程文件（仓库根）**：`LICENSE`、`README.md`、`README.zh-CN.md`、`CONTRIBUTING.md`、`CONTRIBUTING.zh-CN.md`、`.github/workflows/ci.yml`。

**改动**：`pyproject.toml`（命名+依赖分层+packages+pytest markers）、`.gitignore`（移除 uv.lock）、若干测试文件（注入 FakeEmbedder + marker）。

---

## 前置说明（实现者必读）

- **现状核对**：全库只用 embedding 的三个成员：`embed(text)->list[float]`、`embed_batch(texts)->list[list[float]]`、`dimension` 属性（见 `src/module0/embedding.py`）。下游（`module1/pipeline.py:31` 已是 `object|None`、`signature.py`、`compiler.py`）全按鸭子类型调用，不 import 具体类型。
- **测试现状精化**：`tests/module0/test_embedding.py` 已有 `pytest.mark.skipif(not HAS_DEPS)`（缺 torch 自动跳过）——所以 CI 无 torch 时它已能跳过。本计划把这类"缺依赖跳过"**统一为 `requires_model` marker**，口径一致、可 `-m "not requires_model"` 精确控制。`tests/module0/test_compiler.py` 用 `embedding_model=None`，不需改。
- **命令前缀**：项目用 `.venv/bin/python -m pytest ...`（本机 `python` 是 3.9，会 import 失败）。开发环境即将迁移到 uv；Batch 1/2 阶段仍可用 `.venv/bin/python -m pytest`，Batch 2 引入 uv 后改 `uv run pytest`。

---

## Batch 1 — Embedder 抽象 + 测试解耦

### Task 1: 创建 `Embedder` 协议 + 包骨架

**Files:**
- Create: `src/uxflow_embed/__init__.py`
- Create: `src/uxflow_embed/protocol.py`
- Test: `tests/uxflow_embed/__init__.py`, `tests/uxflow_embed/test_protocol.py`

- [ ] **Step 1: 写失败测试**

Create `tests/uxflow_embed/__init__.py` (empty), then `tests/uxflow_embed/test_protocol.py`:

```python
from uxflow_embed import Embedder


class _Impl:
    @property
    def dimension(self) -> int:
        return 3

    def embed(self, text: str) -> list[float]:
        return [0.0, 0.0, 0.0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [[0.0, 0.0, 0.0] for _ in texts]


def test_embedder_is_runtime_checkable_protocol():
    assert isinstance(_Impl(), Embedder)


def test_non_impl_fails_isinstance():
    class Missing:
        def embed(self, text): ...
    assert not isinstance(Missing(), Embedder)
```

- [ ] **Step 2: 运行测试确认失败**

Run: `.venv/bin/python -m pytest tests/uxflow_embed/test_protocol.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'uxflow_embed'`

- [ ] **Step 3: 写协议实现**

Create `src/uxflow_embed/protocol.py`:

```python
"""Embedder protocol — the single embedding seam for all UXFlow modules.

Contract §6: all embedding call sites MUST use the same model + dimension.
Injecting one Embedder instance makes that a structural guarantee.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

__all__ = ["Embedder"]


@runtime_checkable
class Embedder(Protocol):
    """Produces L2-normalized vectors. Implementations: Fake / Local / Api."""

    @property
    def dimension(self) -> int: ...

    def embed(self, text: str) -> list[float]: ...

    def embed_batch(self, texts: list[str]) -> list[list[float]]: ...
```

Create `src/uxflow_embed/__init__.py`:

```python
"""UXFlow embedding backends behind a single Embedder protocol."""

from uxflow_embed.protocol import Embedder

__all__ = ["Embedder"]
```

- [ ] **Step 4: 运行测试确认通过**

Run: `.venv/bin/python -m pytest tests/uxflow_embed/test_protocol.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: 提交**

```bash
git add src/uxflow_embed/protocol.py src/uxflow_embed/__init__.py tests/uxflow_embed/
git commit -m "feat(embed): add Embedder protocol + uxflow_embed package skeleton"
```

---

### Task 2: `FakeEmbedder`（hash-based 确定性）

**Files:**
- Create: `src/uxflow_embed/fake.py`
- Modify: `src/uxflow_embed/__init__.py`
- Test: `tests/uxflow_embed/test_fake.py`

> **说明**：spec §3.2 提到复用现有 `DeterministicEmbeddingModel`。核对后该类是 smoke 专用（按文件路径分桶，制造 006/007 近重复），不适合做通用单测 embedder。本任务建**通用 hash-based `FakeEmbedder`**（任意文本→稳定归一化向量）供跨模块单测；smoke runner 的专用类保留不动（它产出的受控近重复是 hash embedder 无法复现的）。

- [ ] **Step 1: 写失败测试**

Create `tests/uxflow_embed/test_fake.py`:

```python
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
```

- [ ] **Step 2: 运行测试确认失败**

Run: `.venv/bin/python -m pytest tests/uxflow_embed/test_fake.py -v`
Expected: FAIL — `ImportError: cannot import name 'FakeEmbedder'`

- [ ] **Step 3: 写实现**

Create `src/uxflow_embed/fake.py`:

```python
"""FakeEmbedder — deterministic hash-based vectors, zero ML deps.

For unit tests across modules 1/2/3/0.5: they care about how vectors are
ranked/deduped/recalled, not vector semantics. Lets tests run on CI with
no torch and no GPU.
"""

from __future__ import annotations

import hashlib

import numpy as np

__all__ = ["FakeEmbedder"]


class FakeEmbedder:
    """Stable L2-normalized vector per text, derived from its hash."""

    def __init__(self, dimension: int = 1024):
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, text: str) -> list[float]:
        # Seed a deterministic RNG from the text hash → stable pseudo-vector.
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        seed = int.from_bytes(digest[:8], "big")
        rng = np.random.default_rng(seed)
        vec = rng.standard_normal(self._dimension)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]
```

Update `src/uxflow_embed/__init__.py`:

```python
"""UXFlow embedding backends behind a single Embedder protocol."""

from uxflow_embed.protocol import Embedder
from uxflow_embed.fake import FakeEmbedder

__all__ = ["Embedder", "FakeEmbedder"]
```

- [ ] **Step 4: 运行测试确认通过**

Run: `.venv/bin/python -m pytest tests/uxflow_embed/test_fake.py -v`
Expected: PASS (7 passed)

- [ ] **Step 5: 验证 FakeEmbedder 满足协议**

Run: `.venv/bin/python -c "from uxflow_embed import Embedder, FakeEmbedder; assert isinstance(FakeEmbedder(), Embedder); print('ok')"`
Expected: `ok`

- [ ] **Step 6: 提交**

```bash
git add src/uxflow_embed/fake.py src/uxflow_embed/__init__.py tests/uxflow_embed/test_fake.py
git commit -m "feat(embed): add hash-based FakeEmbedder for ML-free tests"
```

---

### Task 3: 迁移 `EmbeddingModel` → `LocalEmbedder`（去 local_files_only + 兼容 shim）

**Files:**
- Create: `src/uxflow_embed/local.py`
- Modify: `src/uxflow_embed/__init__.py`
- Modify: `src/module0/embedding.py`（改为 shim）
- Test: `tests/uxflow_embed/test_local.py`

- [ ] **Step 1: 写失败测试（不需真实模型，只验证结构 + 懒加载 + 兼容别名）**

Create `tests/uxflow_embed/test_local.py`:

```python
import pytest


def test_local_embedder_importable_without_torch():
    # Import must not require torch at module load (lazy import in __init__).
    from uxflow_embed import LocalEmbedder
    assert LocalEmbedder is not None


def test_backcompat_alias_exists():
    # Existing code imports EmbeddingModel from module0.embedding.
    from module0.embedding import EmbeddingModel
    from uxflow_embed import LocalEmbedder
    assert EmbeddingModel is LocalEmbedder


def test_instantiation_without_torch_raises_clear_error():
    from uxflow_embed import LocalEmbedder
    try:
        import sentence_transformers  # noqa: F401
        pytest.skip("sentence-transformers installed; cannot test missing-dep path")
    except ImportError:
        pass
    with pytest.raises(ImportError) as exc:
        LocalEmbedder()
    assert "local-embed" in str(exc.value)
```

- [ ] **Step 2: 运行测试确认失败**

Run: `.venv/bin/python -m pytest tests/uxflow_embed/test_local.py -v`
Expected: FAIL — `ImportError: cannot import name 'LocalEmbedder'`

- [ ] **Step 3: 写实现（迁移现有逻辑 + 去 local_files_only + 缺依赖清晰报错）**

Create `src/uxflow_embed/local.py`:

```python
"""LocalEmbedder — Qwen3-Embedding via sentence-transformers (optional dep).

Produces 1024-d L2-normalized vectors on MPS/CUDA/CPU. Model auto-downloads
from HuggingFace on first use (no local_files_only). Requires the
[local-embed] extra; heavy imports are lazy so `import uxflow_embed` works
without torch installed.

Contract §6: same model + dimension across all embedding call sites.
"""

from __future__ import annotations

import numpy as np

__all__ = ["LocalEmbedder"]

_MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
_DIMENSION = 1024


class LocalEmbedder:
    """Wrapper for local Qwen3-Embedding inference."""

    def __init__(self, model_name: str = _MODEL_NAME, device: str | None = None):
        try:
            from sentence_transformers import SentenceTransformer
            import torch
        except ImportError as e:
            raise ImportError(
                "LocalEmbedder requires the optional 'local-embed' dependencies. "
                'Install with: pip install -e ".[local-embed]"'
            ) from e

        if device is None:
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"

        # No local_files_only: auto-download so cloners can run out of the box.
        self._model = SentenceTransformer(model_name, device=device)
        self._dimension = _DIMENSION

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, text: str) -> list[float]:
        vec = self._model.encode(
            text, normalize_embeddings=True, output_value="sentence_embedding"
        )
        return self._ensure_dimension(vec).tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vecs = self._model.encode(
            texts,
            normalize_embeddings=True,
            output_value="sentence_embedding",
            batch_size=32,
        )
        return [self._ensure_dimension(v).tolist() for v in vecs]

    def _ensure_dimension(self, vec: np.ndarray) -> np.ndarray:
        """Truncate or zero-pad to dimension, then re-normalize (MRL)."""
        if len(vec) >= self._dimension:
            vec = vec[: self._dimension]
        else:
            vec = np.pad(vec, (0, self._dimension - len(vec)))
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec
```

Update `src/uxflow_embed/__init__.py`:

```python
"""UXFlow embedding backends behind a single Embedder protocol."""

from uxflow_embed.protocol import Embedder
from uxflow_embed.fake import FakeEmbedder
from uxflow_embed.local import LocalEmbedder

__all__ = ["Embedder", "FakeEmbedder", "LocalEmbedder"]
```

Replace `src/module0/embedding.py` entirely with a back-compat shim:

```python
"""Back-compat shim. Canonical location is uxflow_embed.LocalEmbedder.

Kept so existing imports (`from module0.embedding import EmbeddingModel`)
keep working after the Embedder extraction.
"""

from __future__ import annotations

from uxflow_embed import LocalEmbedder as EmbeddingModel

__all__ = ["EmbeddingModel"]
```

- [ ] **Step 4: 运行测试确认通过**

Run: `.venv/bin/python -m pytest tests/uxflow_embed/test_local.py -v`
Expected: PASS (若本机装了 sentence-transformers，第 3 个测试 skip；否则 3 passed)

- [ ] **Step 5: 回归——现有 module0 测试不因 shim 破坏**

Run: `.venv/bin/python -m pytest tests/module0/ -v`
Expected: PASS（含 test_embedding.py 的 skipif 行为不变）

- [ ] **Step 6: 提交**

```bash
git add src/uxflow_embed/local.py src/uxflow_embed/__init__.py src/module0/embedding.py tests/uxflow_embed/test_local.py
git commit -m "refactor(embed): migrate EmbeddingModel to uxflow_embed.LocalEmbedder

Drop local_files_only (auto-download), lazy heavy imports with a clear
missing-dependency error, keep module0.embedding as a back-compat shim."
```

---

### Task 4: `ApiEmbedder`（OpenAI 兼容参考实现）

**Files:**
- Create: `src/uxflow_embed/api.py`
- Modify: `src/uxflow_embed/__init__.py`
- Test: `tests/uxflow_embed/test_api.py`

- [ ] **Step 1: 写失败测试（注入假 httpx transport，不发真实网络）**

Create `tests/uxflow_embed/test_api.py`:

```python
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
```

- [ ] **Step 2: 运行测试确认失败**

Run: `.venv/bin/python -m pytest tests/uxflow_embed/test_api.py -v`
Expected: FAIL — `ImportError: cannot import name 'ApiEmbedder'`

- [ ] **Step 3: 写实现**

Create `src/uxflow_embed/api.py`:

```python
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
        return [self._normalize(r["embedding"]) for r in rows]

    def _normalize(self, raw: list[float]) -> list[float]:
        vec = np.asarray(raw, dtype=np.float32)
        if len(vec) >= self._dimension:
            vec = vec[: self._dimension]
        else:
            vec = np.pad(vec, (0, self._dimension - len(vec)))
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.tolist()
```

Update `src/uxflow_embed/__init__.py`:

```python
"""UXFlow embedding backends behind a single Embedder protocol."""

from uxflow_embed.protocol import Embedder
from uxflow_embed.fake import FakeEmbedder
from uxflow_embed.local import LocalEmbedder
from uxflow_embed.api import ApiEmbedder

__all__ = ["Embedder", "FakeEmbedder", "LocalEmbedder", "ApiEmbedder"]
```

- [ ] **Step 4: 运行测试确认通过**

Run: `.venv/bin/python -m pytest tests/uxflow_embed/test_api.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: 提交**

```bash
git add src/uxflow_embed/api.py src/uxflow_embed/__init__.py tests/uxflow_embed/test_api.py
git commit -m "feat(embed): add OpenAI-compatible ApiEmbedder"
```

---

### Task 5: `requires_model` marker + 测试注入 FakeEmbedder

**Files:**
- Modify: `pyproject.toml`（`[tool.pytest.ini_options]` 加 markers）
- Modify: `tests/module0/test_embedding.py`（skipif → marker）
- Modify: `tests/module0/test_integration.py`（真实 embedding 测试打 marker）
- Test: 用现有测试回归

- [ ] **Step 1: 注册 marker**

Modify `pyproject.toml` 的 `[tool.pytest.ini_options]` 段，加入：

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"
markers = [
    "requires_model: needs local Qwen model + torch (skipped in CI)",
]
```

- [ ] **Step 2: `test_embedding.py` 改用 marker**

将 `tests/module0/test_embedding.py` 顶部的 `pytestmark = pytest.mark.skipif(...)` 改为叠加 marker（保留 skipif 兜底，追加 requires_model 便于 `-m` 过滤）：

```python
pytestmark = [
    pytest.mark.requires_model,
    pytest.mark.skipif(not HAS_DEPS, reason="sentence-transformers/torch not installed"),
]
```

- [ ] **Step 3: `test_integration.py` 的真实 embedding 测试打 marker**

在 `tests/module0/test_integration.py` 的 `async def test_embedding_integration(taxonomy):` 上方加一行：

```python
@pytest.mark.requires_model
async def test_embedding_integration(taxonomy):
```

确认该文件顶部已 `import pytest`（若无则加）。

- [ ] **Step 4: 验证 marker 过滤生效**

Run: `.venv/bin/python -m pytest -m "not requires_model" -q`
Expected: PASS，且输出的 deselected 数量 > 0（requires_model 测试被排除）

Run: `.venv/bin/python -m pytest -m "requires_model" --co -q`
Expected: 至少列出 test_embedding.py / test_embedding_integration 的用例

- [ ] **Step 5: 全量回归（本机若有模型）**

Run: `.venv/bin/python -m pytest -q`
Expected: 与改造前一致（209 passed 或含 requires_model 的 skip）

- [ ] **Step 6: 提交**

```bash
git add pyproject.toml tests/module0/test_embedding.py tests/module0/test_integration.py
git commit -m "test: standardize ML-dependent tests on requires_model marker"
```

---

## Batch 2 — 依赖分层 + 命名 + uv

### Task 6: `pyproject.toml` 重构（命名 + 依赖分层 + packages）

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: 改写 `[project]` + 依赖 + packages**

将 `pyproject.toml` 改为（保留已加的 `[tool.pytest.ini_options]` markers）：

```toml
[project]
name = "uxflow"
version = "0.1.0"
description = "SWE trajectory selector for SFT dataset curation"
requires-python = ">=3.11"
license = "Apache-2.0"
readme = "README.md"
authors = [{name = "dengxuanliang"}]

dependencies = [
    "httpx>=0.27",
    "datasketch>=1.6",
    "numpy",
]

[project.optional-dependencies]
local-embed = ["sentence-transformers>=3.0", "torch>=2.0"]
onnx-embed = ["onnxruntime>=1.17"]
dev = ["pytest>=7.0", "pytest-asyncio>=0.23", "python-dotenv", "ruff"]

[project.urls]
Repository = "https://github.com/dengxuanliang/uxflow"
Issues = "https://github.com/dengxuanliang/uxflow/issues"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/llm_gateway", "src/module0", "src/module1", "src/module2", "src/module3", "src/uxflow_embed"]

[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"
markers = [
    "requires_model: needs local Qwen model + torch (skipped in CI)",
]
```

> 注意：`packages` 新增了 `src/uxflow_embed`；`src/module0_5` 待模块0.5 计划落地时再加。

- [ ] **Step 2: 验证包元数据可解析 + import 正常**

Run: `.venv/bin/python -c "import uxflow_embed, module0, module1, module2, module3, llm_gateway; print('imports ok')"`
Expected: `imports ok`

- [ ] **Step 3: 验证轻装可跑纯逻辑测试**

Run: `.venv/bin/python -m pytest -m "not requires_model" -q`
Expected: PASS（全绿）

- [ ] **Step 4: 提交**

```bash
git add pyproject.toml
git commit -m "build: rename package to uxflow and split ML deps into [local-embed]"
```

---

### Task 7: 引入 uv + 提交 `uv.lock`

**Files:**
- Modify: `.gitignore`（移除 `uv.lock` 行）
- Create: `uv.lock`

- [ ] **Step 1: 从 `.gitignore` 移除 `uv.lock`**

编辑 `.gitignore`，删除单独的 `uv.lock` 行。

- [ ] **Step 2: 生成 lock（需已装 uv；未装则 `pip install uv` 或参考 astral.sh/uv）**

Run: `uv lock`
Expected: 生成 `uv.lock`，无解析错误

- [ ] **Step 3: 用 uv 同步 dev 环境并跑测试（验证 lock 可用）**

Run: `uv sync --extra dev && uv run pytest -m "not requires_model" -q`
Expected: 依赖安装成功，测试全绿

- [ ] **Step 4: 提交**

```bash
git add .gitignore uv.lock
git commit -m "build: add uv.lock for reproducible envs (unignore it)"
```

---

## Batch 3 — CI + 工程文件 + 仓库卫生

### Task 8: `LICENSE` + 源文件许可证头

**Files:**
- Create: `LICENSE`
- Modify: 各 `src/**/*.py` 顶部（批量加头）

- [ ] **Step 1: 写入 Apache-2.0 全文**

获取标准全文（`https://www.apache.org/licenses/LICENSE-2.0.txt`）写入 `LICENSE`。附录 "APPENDIX" 的版权占位替换为 `Copyright 2026 dengxuanliang`。

- [ ] **Step 2: 给每个源文件加简短许可证头**

在每个 `src/**/*.py` 文件的模块 docstring **之后**插入一行注释块（不破坏现有 docstring）：

```python
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 dengxuanliang
```

（SPDX 单行标识是 Apache 推荐的轻量写法，避免每文件贴长头。）

- [ ] **Step 3: 验证无语法破坏**

Run: `.venv/bin/python -m pytest -m "not requires_model" -q`
Expected: PASS

- [ ] **Step 4: 提交**

```bash
git add LICENSE src/
git commit -m "docs: add Apache-2.0 LICENSE and SPDX headers"
```

---

### Task 9: `README` + `CONTRIBUTING`（中英双语）

**Files:**
- Create: `README.md`, `README.zh-CN.md`, `CONTRIBUTING.md`, `CONTRIBUTING.zh-CN.md`

- [ ] **Step 1: 写 `README.md`（英文）**

内容需含：项目定位（SWE trajectory selector for SFT dataset curation）；跨平台矩阵表（Linux/macOS × Python 3.11-3.13；Windows untested）；源码安装：

````markdown
## Install (from source)

```bash
git clone https://github.com/dengxuanliang/uxflow
cd uxflow
uv sync --extra dev            # or: pip install -e ".[dev]"
```

Optional local embedding backend (Qwen via torch):

```bash
uv sync --extra local-embed    # or: pip install -e ".[local-embed]"
```

## Test

```bash
uv run pytest -m "not requires_model"   # pure-logic, no ML deps
uv run pytest                           # full suite (needs local-embed)
```
````

顶部加语言切换链接：`[English](README.md) | [中文](README.zh-CN.md)`。

- [ ] **Step 2: 写 `README.zh-CN.md`（中文对照）**

同等内容中文版；顶部同样加语言切换链接。

- [ ] **Step 3: 写 `CONTRIBUTING.md`（英文）**

含：开发环境（`uv sync --extra dev`）；测试约定（纯逻辑测试注入 `FakeEmbedder`；ML 相关测试打 `@pytest.mark.requires_model`，CI 用 `-m "not requires_model"`）；代码风格 `ruff check`；PR 流程（fork → branch → PR，CI 须绿）。

- [ ] **Step 4: 写 `CONTRIBUTING.zh-CN.md`（中文对照）**

- [ ] **Step 5: 提交**

```bash
git add README.md README.zh-CN.md CONTRIBUTING.md CONTRIBUTING.zh-CN.md
git commit -m "docs: add bilingual README and CONTRIBUTING"
```

---

### Task 10: CI workflow（macOS + Linux × 3.11-3.13）

**Files:**
- Create: `.github/workflows/ci.yml`

- [ ] **Step 1: 写 CI 配置**

Create `.github/workflows/ci.yml`:

```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:

jobs:
  test:
    strategy:
      fail-fast: false
      matrix:
        os: [ubuntu-latest, macos-latest]
        python: ["3.11", "3.12", "3.13"]
    runs-on: ${{ matrix.os }}
    steps:
      - uses: actions/checkout@v4
      - name: Install uv
        uses: astral-sh/setup-uv@v5
        with:
          python-version: ${{ matrix.python }}
      - name: Sync dev deps (no torch)
        run: uv sync --extra dev
      - name: Ruff
        run: uv run ruff check
      - name: Pytest (pure-logic)
        run: uv run pytest -m "not requires_model" -q
```

- [ ] **Step 2: 本地预演 CI 命令（验证在无 local-embed 下全绿）**

Run: `uv sync --extra dev && uv run ruff check && uv run pytest -m "not requires_model" -q`
Expected: ruff 通过、测试全绿

> 若 `ruff check` 报大量既有问题，本步骤仅需保证**新增文件**通过；既有告警的整改若超范围，记为后续任务，不阻塞。CI 可先用 `uv run ruff check src/uxflow_embed` 缩小范围（实现者按实际情况定，并在 PR 说明）。

- [ ] **Step 3: 提交**

```bash
git add .github/workflows/ci.yml
git commit -m "ci: add macOS+Linux x Py3.11-3.13 matrix (pure-logic tests)"
```

---

### Task 11: 公开前泄密体检 + clone 冒烟

**Files:** 无（验证任务）

- [ ] **Step 1: 泄密体检**

Run: `git ls-files | grep -iE "\.env$|secret|credential|\.pem$|id_rsa" || echo "clean"`
Expected: `clean`（无敏感文件被跟踪）

Run: `git log --all --oneline -- .env || echo "no .env history"`
Expected: `no .env history`

- [ ] **Step 2: 确认 .gitignore 覆盖敏感项**

确认 `.gitignore` 含 `.env`、`.claude/`、`.DS_Store`（现已有）。

- [ ] **Step 3: clone 冒烟（模拟他人，干净环境）**

```bash
tmp=$(mktemp -d) && git clone "$(git rev-parse --show-toplevel)" "$tmp/uxflow-clone"
cd "$tmp/uxflow-clone" && uv sync --extra dev && uv run pytest -m "not requires_model" -q
```
Expected: clone → sync → 纯逻辑测试全绿（验证"clone 即可跑"）。完成后 `cd` 回原仓库。

- [ ] **Step 4: 无代码改动，无需提交（如体检发现问题，另起修复任务）**

---

## Self-Review（已执行）

**1. Spec coverage** — 逐条对照 spec：
- §3 Embedder 协议+四后端 → Task 1-4（ONNX 仅协议预留，spec 明确不实现 ✓）
- §4 依赖分层+命名+numpy 提核心+懒加载守卫 → Task 3(懒加载)/Task 6(分层) ✓
- §5 测试脱离 ML → Task 2(FakeEmbedder)/Task 5(marker) ✓
- §6 uv.lock+CI 矩阵 → Task 7/Task 10 ✓
- §7 LICENSE/README/CONTRIBUTING/许可证头/泄密体检/文档公开 → Task 8/9/11 ✓
- §8 落地顺序三批 → Batch 1/2/3 对应 ✓
- §9 端到端验证 → Task 11 clone 冒烟 + 各 Task 内回归 ✓

**2. Placeholder scan** — 无 TBD/TODO；README/CONTRIBUTING 正文要点已列（Task 9 给了英文骨架，中文对照为翻译，非占位）；LICENSE 全文引用官方源（标准文本，不逐字贴入计划合理）。

**3. Type consistency** — `Embedder` 三成员（`dimension`/`embed`/`embed_batch`）在 Fake/Local/Api 三实现中签名一致；`module0.embedding.EmbeddingModel` = `LocalEmbedder` 别名贯穿；`packages` 列表新增 `uxflow_embed` 与新包路径一致。

**一处对 spec 的有意偏差（已在 Task 2 标注）**：spec §3.2 说"复用 DeterministicEmbeddingModel"，实现改为"新建通用 hash-based FakeEmbedder + 保留 smoke 专用类"——因前者是 smoke 领域专用（路径分桶造近重复），不适合通用单测。功能意图（包内可注入的假 embedder）完全满足。

---

## 落地顺序与验证门

- **Batch 1 完成门**：`pytest -m "not requires_model"` 全绿，且 uxflow_embed 四文件就位、module0 shim 不破坏现有测试。
- **Batch 2 完成门**：`uv sync --extra dev && uv run pytest -m "not requires_model"` 全绿，包名 uxflow，uv.lock 提交。
- **Batch 3 完成门**：CI 配置就位、双语文档+LICENSE 齐全、clone 冒烟通过、泄密体检 clean。
- 三批全部完成后，即可执行"连接 remote + 首推"（此前用户已选择：改造+体检完成后再连接推送）。
