# 模块 2+3 主链路 Implementation Plan（第一批）

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 打通 "Problem Spec → 最终 SFT 数据集(N 条)" 可交付链路：抽出 `SliceStore` 存储接口、模块 2 软加分精排、模块 3 去重/submodular 覆盖优选/配比。

**Architecture:** 复用模块 1 现有 recall+judge，把召回改为带回 RRF 分（`RecallHit`），judge 硬过滤改软加分（未命中 ×0.3），产出切片级 `ScoredCandidate`；模块 3 在切片级做两层去重 → facility-location submodular 贪心选 N 条 → 配比 + 通用数据拼接钩子。存储走 `SliceStore` 抽象接口，V1 用现有 `MemoryIndex` 实现，真实持久化后续替换实现类。

**Tech Stack:** Python 3.11+, numpy, datasketch(MinHash), pytest + pytest-asyncio(`asyncio_mode=auto`)。测试命令一律用 `.venv/bin/python -m pytest`。

**Spec:** `docs/superpowers/specs/2026-07-09-modules-2-3-0.5-design.md`（§2/§3/§4/§6.5/§7）。本批为 §6.5 第一批；模块 0.5 为第二批，另出计划。

**Git 工作流:** 本项目是标准 git 仓库（`main` 分支）。实现前先从 `main` 切功能分支 `feat/modules-2-3-main-chain`，不直接在 main 上提交。每个 Task 的 "Checkpoint" 步骤 = 跑全量 `.venv/bin/python -m pytest -q` 全绿后 `git add <具体文件>` + `git commit`（提交信息见各步）。只 add 该 Task 涉及的文件，不用 `git add .`。

---

## File Structure（决策锁定）

**新增**
- `src/module1/store.py` — `SliceStore` 协议 + `RecallHit` dataclass（存储抽象层，模块 2/3/0.5 共同依赖）。
- `src/module2/__init__.py`、`src/module2/models.py`（`ScoredCandidate`）、`src/module2/rerank.py`（软加分排序）。
- `src/module3/__init__.py`、`src/module3/dedup.py`、`src/module3/selection.py`、`src/module3/compose.py`。
- 测试：`tests/module2/`、`tests/module3/` 及各 `__init__.py`/`conftest.py`。

**改动**
- `src/module1/models.py` — `TrajectorySignature` 加 `capability_labels`。
- `src/module1/index.py` — `recall()` 改返回 `list[RecallHit]`；补 `get_slice`/`update_labels`；`MemoryIndex` 声明实现 `SliceStore`。
- `src/module1/pipeline.py` — 拆分 `_process_sub_problem`，硬过滤改软加分，接入模块 2；slice_map 下沉进 store。
- `tests/module1/test_index.py`、`tests/module1/test_pipeline.py` — 适配新返回类型。
- `pyproject.toml` — 加 `datasketch` 依赖；wheel packages 加 `src/module2`、`src/module3`。

---

## Task 1: 加依赖与打包配置

**Files:**
- Modify: `pyproject.toml:6-9`（dependencies）、`pyproject.toml:24`（wheel packages）

- [ ] **Step 1: 安装 datasketch 并加入依赖**

修改 `pyproject.toml` 的 `dependencies`：

```toml
dependencies = [
    "httpx>=0.27",
    "sentence-transformers>=3.0",
    "datasketch>=1.6",
]
```

- [ ] **Step 2: wheel packages 加入模块 2/3**

修改 `pyproject.toml:23-24`：

```toml
[tool.hatch.build.targets.wheel]
packages = ["src/llm_gateway", "src/module0", "src/module1", "src/module2", "src/module3"]
```

- [ ] **Step 3: 安装并验证导入**

Run: `.venv/bin/pip install "datasketch>=1.6" && .venv/bin/python -c "from datasketch import MinHash; print('ok')"`
Expected: 输出 `ok`

> **导入机制说明（已核实）**：本项目 editable 安装把 `src/` 目录直接放到 `sys.path`（`.venv/.../_editable_impl_llm_gateway.pth` 内容就是 `.../UXFlow/src`），**不是**按包名映射。因此 `src/module2`、`src/module3` 作为新的顶层包**建好即可导入，无需重装**。Step 2 改 `pyproject.toml` 的 wheel `packages` 只影响构建可分发 wheel，不影响本地开发/测试。

- [ ] **Step 4: Checkpoint**

Run: `.venv/bin/python -m pytest -q`
Expected: 现有 173 passed（未改代码，仅加依赖）。

---

## Task 2: TrajectorySignature 增加 capability_labels 字段

**Files:**
- Modify: `src/module1/models.py:43-56`
- Test: `tests/module1/test_models.py`

- [ ] **Step 1: Write the failing test**

追加到 `tests/module1/test_models.py`：

```python
def test_signature_capability_labels_defaults_none():
    from module1.models import TrajectorySignature
    sig = TrajectorySignature(
        trajectory_id="t1", slice_index=0, step_range=(0, 5), step_count=6,
        turn_count=1, languages=["python"], tools_used=["Bash"],
        has_error_pattern=False, has_success_pattern=True,
        has_verification_step=False, bm25_tokens=["python"],
    )
    assert sig.capability_labels is None
    sig.capability_labels = ["valid_syntax_in_toolcall"]
    assert sig.capability_labels == ["valid_syntax_in_toolcall"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/module1/test_models.py::test_signature_capability_labels_defaults_none -v`
Expected: FAIL（`TypeError` 或 `AttributeError`：无 `capability_labels`）

- [ ] **Step 3: Implement**

修改 `src/module1/models.py` 的 `TrajectorySignature`，在 `embedding` 字段后追加：

```python
    embedding: list[float] = field(default_factory=list)
    capability_labels: list[str] | None = None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/module1/test_models.py -v`
Expected: PASS

- [ ] **Step 5: Checkpoint**

Run: `.venv/bin/python -m pytest -q`
Expected: 174 passed（新增 1）。

---

## Task 3: SliceStore 接口 + RecallHit

**Files:**
- Create: `src/module1/store.py`
- Test: `tests/module1/test_store.py`

- [ ] **Step 1: Write the failing test**

创建 `tests/module1/test_store.py`：

```python
from module1.store import RecallHit, SliceStore
from module1.models import TrajectorySignature


def _sig(traj_id="t1"):
    return TrajectorySignature(
        trajectory_id=traj_id, slice_index=0, step_range=(0, 5), step_count=6,
        turn_count=1, languages=["python"], tools_used=["Bash"],
        has_error_pattern=False, has_success_pattern=True,
        has_verification_step=False, bm25_tokens=["python"],
    )


def test_recall_hit_carries_signature_and_score():
    hit = RecallHit(signature=_sig(), rrf_score=0.42)
    assert hit.signature.trajectory_id == "t1"
    assert hit.rrf_score == 0.42


def test_slicestore_is_runtime_checkable_protocol():
    # MemoryIndex (Task 4) will satisfy this; here just assert the protocol exists
    # and declares the required methods.
    for name in ("add_batch", "recall", "get_slice", "update_labels"):
        assert hasattr(SliceStore, name)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/module1/test_store.py -v`
Expected: FAIL（`ModuleNotFoundError: module1.store`）

- [ ] **Step 3: Implement**

创建 `src/module1/store.py`：

```python
"""Storage abstraction shared by modules 2/3/0.5.

V1 implementation is MemoryIndex (src/module1/index.py). A real persistent
backend (ES/Qdrant or SQLite+faiss) is a later task that swaps the impl class;
callers depend only on this protocol.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from module1.models import Slice, TrajectorySignature

__all__ = ["RecallHit", "SliceStore"]


@dataclass
class RecallHit:
    """A recalled slice signature plus its RRF fusion score.

    Module 2 needs the fusion score as the base for relevance scoring, which
    the old recall() (returning bare signatures) discarded.
    """
    signature: TrajectorySignature
    rrf_score: float


@runtime_checkable
class SliceStore(Protocol):
    """Read + mutable-writeback interface over slice signatures."""

    def add_batch(self, sigs: list[TrajectorySignature]) -> None: ...

    def recall(
        self,
        *,
        structured_filters: dict,
        keywords: list[str],
        query_embeddings: list[list[float]],
        top_n: int = 20,
    ) -> list[RecallHit]: ...

    def get_slice(self, trajectory_id: str, slice_index: int) -> Slice | None: ...

    def update_labels(
        self, trajectory_id: str, slice_index: int, labels: list[str]
    ) -> None: ...
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/module1/test_store.py -v`
Expected: PASS（2 passed）

- [ ] **Step 5: Checkpoint**

Run: `.venv/bin/python -m pytest -q`
Expected: 176 passed。

---

## Task 4: MemoryIndex 改造 —— recall 带回 RRF 分 + get_slice/update_labels

**Files:**
- Modify: `src/module1/index.py`（`recall` 尾部 `src/module1/index.py:86-88`；类体新增方法）
- Test: `tests/module1/test_index.py`（适配返回类型）、新增回写测试

模块 1 现有 `recall()` 返回 `list[TrajectorySignature]`；改为 `list[RecallHit]`。`MemoryIndex` 需要能按 `(trajectory_id, slice_index)` 取回 slice 原文并回写标签——为此让 index 持有 slice 引用（由 `add` 时传入，见 Task 6 pipeline 下沉）。本任务先在 index 内部维护 `slice_index` 侧的签名映射，slice 原文映射在 Task 6 补齐。

- [ ] **Step 1: Write the failing test**

修改 `tests/module1/test_index.py`——把所有 `r.trajectory_id`（10 处 recall 结果访问）改为 `r.signature.trajectory_id`。示例（`test_filter_languages`）：

```python
    results = idx.recall(
        structured_filters={"languages": ["python"]},
        keywords=[], query_embeddings=[], top_n=10,
    )
    traj_ids = [r.signature.trajectory_id for r in results]
    assert "t1" in traj_ids
    assert "t3" in traj_ids
    assert "t2" not in traj_ids
```

对 `test_bm25_recall` / `test_vector_recall`：`results[0].trajectory_id` → `results[0].signature.trajectory_id`。
对 `test_rrf_fusion` / `test_bm25_mixed_case_idf`：列表推导同样改 `r.signature.trajectory_id`。
`test_empty_index_returns_empty` / `test_top_n_limits_output` / `test_null_filters_skip_filtering` 只断言 `len`/`== []`，无需改。

追加带分 + 回写测试到 `tests/module1/test_index.py`：

```python
def test_recall_returns_recall_hits_with_scores():
    from module1.store import RecallHit
    idx = MemoryIndex()
    idx.add(_make_sig("t1", bm25_tokens=["syntaxerror", "python"]))
    results = idx.recall(
        structured_filters={}, keywords=["syntaxerror"],
        query_embeddings=[], top_n=5,
    )
    assert results
    assert all(isinstance(r, RecallHit) for r in results)
    assert all(r.rrf_score >= 0.0 for r in results)


def test_update_labels_mutates_signature():
    idx = MemoryIndex()
    idx.add(_make_sig("t1", slice_idx=2))
    idx.update_labels("t1", 2, ["valid_syntax_in_toolcall"])
    hits = idx.recall(structured_filters={}, keywords=["python"],
                      query_embeddings=[], top_n=5)
    match = [h for h in hits if h.signature.trajectory_id == "t1"][0]
    assert match.signature.capability_labels == ["valid_syntax_in_toolcall"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/module1/test_index.py -v`
Expected: FAIL（`AttributeError: 'TrajectorySignature' object has no attribute 'signature'`，以及 `update_labels` 不存在）

- [ ] **Step 3: Implement**

修改 `src/module1/index.py`：

顶部 import 增加：

```python
from module1.store import RecallHit
```

`recall()` 结尾（当前 `src/module1/index.py:86-88`）由：

```python
        fused.sort(key=lambda x: x[1], reverse=True)
        return [sig for sig, _score in fused[:top_n]]
```

改为：

```python
        fused.sort(key=lambda x: x[1], reverse=True)
        return [RecallHit(signature=sig, rrf_score=score) for sig, score in fused[:top_n]]
```

在 `MemoryIndex` 类体追加两个方法（放在 `recall` 之后）：

```python
    def get_slice(self, trajectory_id: str, slice_index: int):
        """Return the slice signature for (trajectory_id, slice_index), or None.

        Slice *原文* (Slice objects) is attached by the pipeline via
        set_slice_source (Task 6); here we expose the signature lookup.
        """
        for sig in self._signatures:
            if sig.trajectory_id == trajectory_id and sig.slice_index == slice_index:
                return sig
        return None

    def update_labels(self, trajectory_id: str, slice_index: int, labels: list[str]) -> None:
        """Attach/overwrite capability_labels on the matching signature (writeback seam)."""
        sig = self.get_slice(trajectory_id, slice_index)
        if sig is not None:
            sig.capability_labels = list(labels)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/module1/test_index.py -v`
Expected: PASS（原有 + 2 新增）

- [ ] **Step 5: Checkpoint**

Run: `.venv/bin/python -m pytest -q`
Expected: 全绿（module1 其余测试仍依赖 pipeline，见 Task 6；此步 test_index 全过，test_pipeline 可能因 recall 返回类型变化而红——若红，属预期，Task 6 修复。若要保持每步全绿，可将 Task 6 与本任务合并执行）。
> **执行提示**：本任务与 Task 6 对 `recall()` 消费方有耦合。建议 Task 4→6 连续做，中间只跑 `test_index`/`test_store`，最后在 Task 6 Step 收尾时跑全量。

---

## Task 5: 模块 2 —— ScoredCandidate 数据模型

**Files:**
- Create: `src/module2/__init__.py`、`src/module2/models.py`
- Test: `tests/module2/__init__.py`、`tests/module2/test_models.py`

- [ ] **Step 1: Write the failing test**

创建 `tests/module2/__init__.py`（空文件）和 `tests/module2/test_models.py`：

```python
from module2.models import ScoredCandidate


def test_scored_candidate_fields():
    sc = ScoredCandidate(
        trajectory_id="t1", slice_index=0, trajectory_path="/x.jsonl",
        sub_problem_id="p1", capability=["valid_syntax_in_toolcall"],
        relevance_score=0.9, judge_confidence=0.88,
        loss_mask_spans=[{"start_step": 0, "end_step": 3}],
        embedding=[0.1] * 1024,
    )
    assert sc.relevance_score == 0.9
    assert sc.sub_problem_id == "p1"
    assert sc.loss_mask_spans[0]["end_step"] == 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/module2/test_models.py -v`
Expected: FAIL（`ModuleNotFoundError: module2`）

- [ ] **Step 3: Implement**

创建 `src/module2/__init__.py`：

```python
from module2.models import ScoredCandidate
from module2.rerank import rerank

__all__ = ["ScoredCandidate", "rerank"]
```

创建 `src/module2/models.py`：

```python
"""Module 2 output model: slice-level scored candidate."""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ["ScoredCandidate"]


@dataclass
class ScoredCandidate:
    """One (slice, sub_problem) scored record. Module 3 consumes relevance_score."""
    trajectory_id: str
    slice_index: int
    trajectory_path: str
    sub_problem_id: str
    capability: list[str]
    relevance_score: float          # sort key for module 3 submodular
    judge_confidence: float         # quality reference, not used for ranking
    loss_mask_spans: list[dict]
    embedding: list[float] = field(default_factory=list)
```

> **注**：`src/module2/__init__.py` 导入 `rerank`，而 `rerank.py` 在 Task 7 才创建。**先把 `__init__.py` 写成只导出 `ScoredCandidate`**，Task 7 再补 `rerank` 导出，避免本任务 import 失败。即本步 `__init__.py` 实际内容为：
> ```python
> from module2.models import ScoredCandidate
> __all__ = ["ScoredCandidate"]
> ```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/module2/test_models.py -v`
Expected: PASS

- [ ] **Step 5: Checkpoint**

Run: `.venv/bin/python -m pytest -q tests/module2 tests/module1/test_store.py`
Expected: 全绿。

---

## Task 6: pipeline 改造 —— slice 原文下沉 store + recall 消费 RecallHit

**Files:**
- Modify: `src/module1/index.py`（补 slice 原文映射 + `get_slice_obj`）
- Modify: `src/module1/pipeline.py`（`_build_index`、`_process_sub_problem`）
- Test: `tests/module1/test_pipeline.py`（适配）

模块 3 的 submodular 在切片级工作，需要 slice 原文回溯。让 `MemoryIndex` 同时持有 slice 原文（pipeline `add` 时传入），`pipeline._slice_map` 下沉。

- [ ] **Step 1: Write the failing test**

修改 `tests/module1/test_pipeline.py`：现有 `test_pipeline_rerun_no_accumulation`（约 line 116）断言 `pipeline._index.size == len(pipeline._slice_map)`。改为直接查 store：

```python
    assert len(second) == len(first)
    assert pipeline._store.size == pipeline._store.slice_source_count
```

其余 pipeline 测试（`test_pipeline_end_to_end` 等）断言的是 `run()` 返回的 `SFTCandidate`——本任务保持 `run()` 对外契约不变（仍返回 `list[SFTCandidate]`），故这些测试无需改。

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/module1/test_pipeline.py::test_pipeline_rerun_no_accumulation -v`
Expected: FAIL（`_store` 不存在）

- [ ] **Step 3: Implement — index 持有 slice 原文**

在 `src/module1/index.py` 的 `MemoryIndex.__init__` 增加 slice 原文映射：

```python
    def __init__(self):
        self._signatures: list[TrajectorySignature] = []
        self._slice_source: dict[tuple[str, int], object] = {}  # (traj_id, idx) -> Slice
```

追加方法：

```python
    def set_slice_source(self, trajectory_id: str, slice_index: int, slice_obj) -> None:
        self._slice_source[(trajectory_id, slice_index)] = slice_obj

    def get_slice_obj(self, trajectory_id: str, slice_index: int):
        return self._slice_source.get((trajectory_id, slice_index))

    @property
    def slice_source_count(self) -> int:
        return len(self._slice_source)
```

- [ ] **Step 4: Implement — pipeline 用 store，recall 消费 RecallHit**

修改 `src/module1/pipeline.py`：

`__init__` 里 `self._index = MemoryIndex()` 改名为 `self._store`（保持类型 `MemoryIndex`），删除 `self._slice_map`；`self._traj_paths` 保留。

```python
    def __init__(self, config: PipelineConfig, gateway):
        self._config = config
        self._gateway = gateway
        self._store = MemoryIndex()
        self._judge = Judge(
            gateway=gateway, model=config.judge_model,
            batch_size=config.judge_batch_size,
        )
        self._traj_paths: dict[str, str] = {}
```

`run()` 开头重置改为：

```python
        # Reset per-run state so reusing a pipeline instance doesn't accumulate.
        self._store = MemoryIndex()
        self._traj_paths.clear()
        self._build_index(trajectory_paths)
        if self._store.size == 0:
            return []
```

`_build_index` 内 `self._index.add(sig)` + slice_map 赋值改为：

```python
                for sl in slices:
                    sig = extract_signature(sl, embedding_model=self._config.embedding_model)
                    self._store.add(sig)
                    self._store.set_slice_source(sl.trajectory_id, sl.slice_index, sl)
```

`_process_sub_problem` 内 recall 消费改为（`recalled_sigs` 现为 `list[RecallHit]`）：

```python
        recalled_hits = self._store.recall(
            structured_filters=structured_filters,
            keywords=keywords,
            query_embeddings=query_embeddings,
            top_n=self._config.recall_top_n,
        )
        if not recalled_hits:
            return []

        recalled_slices = []
        for hit in recalled_hits:
            sl = self._store.get_slice_obj(hit.signature.trajectory_id, hit.signature.slice_index)
            if sl is not None:
                recalled_slices.append(sl)
        if not recalled_slices:
            return []
```

> **注**：本任务保持 `_process_sub_problem` 的 judge + 硬过滤逻辑不动（仍返回 `SFTCandidate`），只切换到 `_store` + `RecallHit`。软加分改造在 Task 7 做（届时 `_process_sub_problem` 拆分）。这样本任务是纯粹的存储接口迁移，可独立验证。

- [ ] **Step 5: Run tests to verify pass**

Run: `.venv/bin/python -m pytest tests/module1 -v`
Expected: PASS（全部 module1 测试，含 test_index/test_pipeline）

- [ ] **Step 6: Checkpoint**

Run: `.venv/bin/python -m pytest -q`
Expected: 全绿（此时存储接口迁移完成，行为等价）。

---

## Task 7: 模块 2 —— 软加分 rerank + pipeline 接入

**Files:**
- Create: `src/module2/rerank.py`
- Modify: `src/module2/__init__.py`（补 rerank 导出）
- Modify: `src/module1/pipeline.py`（`_process_sub_problem` 拆出评分，硬过滤改软加分）
- Test: `tests/module2/test_rerank.py`

relevance 公式（spec §3.2 决策 3）：`relevance_score = rrf_score ×(judge.match ? 1.0 : 0.3)`。未命中不砍，以 ×0.3 保留。

- [ ] **Step 1: Write the failing test**

创建 `tests/module2/test_rerank.py`：

```python
from module1.store import RecallHit
from module1.models import TrajectorySignature, JudgeResult
from module2.rerank import rerank


def _hit(traj_id, idx, score, emb=None):
    sig = TrajectorySignature(
        trajectory_id=traj_id, slice_index=idx, step_range=(0, 5), step_count=6,
        turn_count=1, languages=["python"], tools_used=["Bash"],
        has_error_pattern=False, has_success_pattern=True,
        has_verification_step=False, bm25_tokens=["python"],
        embedding=emb or [0.1] * 1024,
    )
    return RecallHit(signature=sig, rrf_score=score)


_SUB = {
    "id": "p1", "target_capability": ["valid_syntax_in_toolcall"],
}


def test_matched_keeps_full_score():
    hits = [_hit("t1", 0, 0.5)]
    judged = [JudgeResult(match=True, confidence=0.9, spans=[{"start_step": 0, "end_step": 2}])]
    out = rerank(hits, judged, _SUB, trajectory_path="/x.jsonl")
    assert len(out) == 1
    assert out[0].relevance_score == 0.5          # 0.5 × 1.0
    assert out[0].judge_confidence == 0.9
    assert out[0].sub_problem_id == "p1"


def test_unmatched_decayed_not_dropped():
    hits = [_hit("t1", 0, 0.5)]
    judged = [JudgeResult(match=False, confidence=0.1, spans=[])]
    out = rerank(hits, judged, _SUB, trajectory_path="/x.jsonl")
    assert len(out) == 1                           # NOT dropped
    assert abs(out[0].relevance_score - 0.15) < 1e-9  # 0.5 × 0.3


def test_sorted_by_relevance_desc():
    hits = [_hit("t1", 0, 0.2), _hit("t2", 1, 0.9)]
    judged = [
        JudgeResult(match=True, confidence=0.8, spans=[]),
        JudgeResult(match=False, confidence=0.1, spans=[]),
    ]
    out = rerank(hits, judged, _SUB, trajectory_path="/x.jsonl")
    # t1: 0.2×1=0.2 ; t2: 0.9×0.3=0.27 → t2 first
    assert out[0].trajectory_id == "t2"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/module2/test_rerank.py -v`
Expected: FAIL（`ModuleNotFoundError: module2.rerank`）

- [ ] **Step 3: Implement rerank**

创建 `src/module2/rerank.py`：

```python
"""Module 2: relevance re-ranking with soft label scoring.

Design §3.2: relevance_score = rrf_score × (judge.match ? 1.0 : 0.3).
Unmatched slices are decayed, NOT filtered — recall already guarantees
semantic relevance; the judge only refines. This avoids the noise-driven
false kills the old hard filter (pipeline.py) caused.
"""

from __future__ import annotations

from module1.store import RecallHit
from module1.models import JudgeResult
from module2.models import ScoredCandidate

__all__ = ["rerank"]

_MISS_DECAY = 0.3


def rerank(
    hits: list[RecallHit],
    judge_results: list[JudgeResult],
    sub_problem: dict,
    *,
    trajectory_path: str,
) -> list[ScoredCandidate]:
    """Score recalled hits against judge verdicts, return sorted candidates.

    hits and judge_results are index-aligned (one judge verdict per hit).
    """
    sub_id = sub_problem.get("id", "unknown")
    capability = sub_problem.get("target_capability", [])

    scored: list[ScoredCandidate] = []
    for hit, jr in zip(hits, judge_results):
        multiplier = 1.0 if jr.match else _MISS_DECAY
        scored.append(ScoredCandidate(
            trajectory_id=hit.signature.trajectory_id,
            slice_index=hit.signature.slice_index,
            trajectory_path=trajectory_path,
            sub_problem_id=sub_id,
            capability=capability,
            relevance_score=hit.rrf_score * multiplier,
            judge_confidence=jr.confidence,
            loss_mask_spans=jr.spans,
            embedding=hit.signature.embedding,
        ))
    scored.sort(key=lambda c: c.relevance_score, reverse=True)
    return scored
```

补 `src/module2/__init__.py` 导出：

```python
from module2.models import ScoredCandidate
from module2.rerank import rerank

__all__ = ["ScoredCandidate", "rerank"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/module2 -v`
Expected: PASS

- [ ] **Step 5: pipeline 接入软加分 —— 新增 run_scored 方法**

为不破坏现有 `run()`→`SFTCandidate` 契约（模块 1 测试依赖），**新增** `run_scored()` 返回 `list[ScoredCandidate]`，作为模块 2 的对外入口。在 `src/module1/pipeline.py`：

顶部 import：

```python
from module2.rerank import rerank
from module2.models import ScoredCandidate
```

抽出评分子过程并新增方法：

```python
    async def run_scored(
        self,
        *,
        trajectory_paths: list[str | pathlib.Path],
        problem_specs: list[dict],
    ) -> list[ScoredCandidate]:
        """Module 1→2 entry: recall + judge + soft-score, slice-level output."""
        self._store = MemoryIndex()
        self._traj_paths.clear()
        self._build_index(trajectory_paths)
        if self._store.size == 0:
            return []

        all_scored: list[ScoredCandidate] = []
        for spec in problem_specs:
            for sp in spec.get("sub_problems", []):
                all_scored.extend(await self._score_sub_problem(sp))
        return all_scored

    async def _score_sub_problem(self, sub_problem: dict) -> list[ScoredCandidate]:
        structured_filters = sub_problem.get("structured_filters", {})
        keywords = sub_problem.get("keywords", [])
        hyde_positive = sub_problem.get("hyde_positive", [])
        target_capability = sub_problem.get("target_capability", [])
        trajectory_signal = sub_problem.get("trajectory_signal", "")

        query_embeddings = []
        if self._config.embedding_model and hyde_positive:
            query_embeddings = self._config.embedding_model.embed_batch(hyde_positive)

        hits = self._store.recall(
            structured_filters=structured_filters, keywords=keywords,
            query_embeddings=query_embeddings, top_n=self._config.recall_top_n,
        )
        if not hits:
            return []

        slices = []
        kept_hits = []
        for hit in hits:
            sl = self._store.get_slice_obj(hit.signature.trajectory_id, hit.signature.slice_index)
            if sl is not None:
                slices.append(sl)
                kept_hits.append(hit)
        if not slices:
            return []

        judge_results = await self._judge.judge_batch(
            slices=slices, target_capability=target_capability,
            trajectory_signal=trajectory_signal,
        )

        # cache writeback: matched slices get their capability labels persisted
        for hit, jr in zip(kept_hits, judge_results):
            if jr.match:
                self._store.update_labels(
                    hit.signature.trajectory_id, hit.signature.slice_index, target_capability,
                )

        traj_path = self._traj_paths.get(
            kept_hits[0].signature.trajectory_id, "",
        ) if kept_hits else ""
        # per-hit path (hits may span trajectories) resolved inside rerank via map:
        scored = rerank(kept_hits, judge_results, sub_problem, trajectory_path="")
        for sc in scored:
            sc.trajectory_path = self._traj_paths.get(sc.trajectory_id, "")
        return scored
```

> **注**：`rerank` 的 `trajectory_path` 传空串占位，随后按每条 candidate 的 `trajectory_id` 回填正确路径（因一次召回可能跨多条轨迹）。

- [ ] **Step 6: Write pipeline scored test**

追加到 `tests/module1/test_pipeline.py`：

```python
async def test_run_scored_soft_scoring(trajectories_path, problem_spec_dict):
    """run_scored keeps unmatched slices (decayed), returns slice-level candidates."""
    from module2.models import ScoredCandidate
    judge_resp = '[{"match": true, "confidence": 0.9, "spans": [{"start_step": 0, "end_step": 3}], "reasoning": "ok"}]'
    gw = FakeGateway([judge_resp] * 50)
    cfg = PipelineConfig(judge_model="test-model", recall_top_n=5)
    pipeline = TrajectoryPipeline(config=cfg, gateway=gw)

    scored = await pipeline.run_scored(
        trajectory_paths=[trajectories_path], problem_specs=[problem_spec_dict],
    )
    assert all(isinstance(s, ScoredCandidate) for s in scored)
    assert scored  # recall matched fixture → non-empty
    assert all(s.relevance_score >= 0.0 for s in scored)
    # sorted descending
    assert scored == sorted(scored, key=lambda s: s.relevance_score, reverse=True)
```

- [ ] **Step 7: Run tests to verify pass**

Run: `.venv/bin/python -m pytest tests/module1 tests/module2 -v`
Expected: PASS

- [ ] **Step 8: Checkpoint**

Run: `.venv/bin/python -m pytest -q`
Expected: 全绿。`run()`（旧 SFTCandidate 契约）与 `run_scored()`（新软加分）并存。

---

## Task 8: 模块 3 —— 语义去重（两层）

**Files:**
- Create: `src/module3/__init__.py`、`src/module3/dedup.py`
- Test: `tests/module3/__init__.py`、`tests/module3/conftest.py`、`tests/module3/test_dedup.py`

两层：MinHash 抓逐字近重复（token 级）；embedding cosine > 0.95 抓语义重复。一组重复保留 `relevance_score` 最高者。

- [ ] **Step 1: Write the failing test**

创建 `tests/module3/__init__.py`（空）、`tests/module3/conftest.py`：

```python
import pytest
from module2.models import ScoredCandidate


@pytest.fixture
def mk_candidate():
    def _mk(traj_id, idx, score, emb, sub="p1", tokens=None):
        c = ScoredCandidate(
            trajectory_id=traj_id, slice_index=idx, trajectory_path="/x.jsonl",
            sub_problem_id=sub, capability=["valid_syntax_in_toolcall"],
            relevance_score=score, judge_confidence=0.8,
            loss_mask_spans=[], embedding=emb,
        )
        c._tokens = tokens or ["import", "os", "syntaxerror"]  # test-only helper
        return c
    return _mk
```

创建 `tests/module3/test_dedup.py`：

```python
from module3.dedup import deduplicate


def _emb(v):
    return [v] + [0.0] * 1023


def test_embedding_near_duplicates_collapsed(mk_candidate):
    # two near-identical embeddings → keep higher relevance
    a = mk_candidate("t1", 0, 0.9, _emb(1.0))
    b = mk_candidate("t2", 0, 0.5, _emb(0.999))  # cosine ~1.0 with a
    out = deduplicate([a, b], cosine_threshold=0.95)
    assert len(out) == 1
    assert out[0].trajectory_id == "t1"  # higher relevance kept


def test_distinct_embeddings_all_kept(mk_candidate):
    a = mk_candidate("t1", 0, 0.9, _emb(1.0))
    b = mk_candidate("t2", 0, 0.5, [0.0, 1.0] + [0.0] * 1022)  # orthogonal
    out = deduplicate([a, b], cosine_threshold=0.95)
    assert len(out) == 2


def test_empty_input(mk_candidate):
    assert deduplicate([], cosine_threshold=0.95) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/module3/test_dedup.py -v`
Expected: FAIL（`ModuleNotFoundError: module3.dedup`）

- [ ] **Step 3: Implement**

创建 `src/module3/__init__.py`：

```python
from module3.dedup import deduplicate

__all__ = ["deduplicate"]
```

创建 `src/module3/dedup.py`：

```python
"""Module 3 step 1: semantic dedup.

Two layers (design §4.1):
- embedding cosine > threshold → semantic near-duplicate.
- MinHash Jaccard on bm25-like token sets → near-verbatim duplicate.
Within a duplicate group, keep the highest relevance_score.

For fixture scale the embedding layer dominates; MinHash guards verbatim
repeats when the corpus grows.
"""

from __future__ import annotations

import numpy as np

from module2.models import ScoredCandidate

__all__ = ["deduplicate"]


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(a @ b / (na * nb))


def deduplicate(
    candidates: list[ScoredCandidate],
    *,
    cosine_threshold: float = 0.95,
) -> list[ScoredCandidate]:
    """Collapse near-duplicate candidates, keeping highest relevance per group."""
    if not candidates:
        return []

    # Sort by relevance desc so the first survivor in each group is the best.
    ordered = sorted(candidates, key=lambda c: c.relevance_score, reverse=True)
    kept: list[ScoredCandidate] = []
    kept_vecs: list[np.ndarray] = []

    for c in ordered:
        vec = np.asarray(c.embedding, dtype=np.float32)
        is_dup = any(_cosine(vec, kv) > cosine_threshold for kv in kept_vecs)
        if not is_dup:
            kept.append(c)
            kept_vecs.append(vec)
    return kept
```

> **注**：MinHash 层放在此文件的后续增强——本步先落 embedding 层（fixture 规模足够，且测试只验证 embedding 去重）。MinHash 逐字层作为 Task 8b 补充（见下）。这样保证每步可测。

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/module3/test_dedup.py -v`
Expected: PASS

- [ ] **Step 5: Checkpoint**

Run: `.venv/bin/python -m pytest -q`
Expected: 全绿。

---

## Task 8b: 模块 3 —— MinHash 逐字去重层

**Files:**
- Modify: `src/module3/dedup.py`
- Test: `tests/module3/test_dedup.py`

- [ ] **Step 1: Write the failing test**

追加到 `tests/module3/test_dedup.py`：

```python
def test_minhash_verbatim_duplicates_collapsed(mk_candidate):
    # identical token sets but ORTHOGONAL embeddings → only MinHash catches it
    a = mk_candidate("t1", 0, 0.9, [1.0] + [0.0] * 1023, tokens=["import", "os", "sys", "re"])
    b = mk_candidate("t2", 0, 0.4, [0.0, 1.0] + [0.0] * 1022, tokens=["import", "os", "sys", "re"])
    out = deduplicate([a, b], cosine_threshold=0.95, minhash_threshold=0.9)
    assert len(out) == 1
    assert out[0].trajectory_id == "t1"


def test_minhash_disjoint_tokens_kept(mk_candidate):
    a = mk_candidate("t1", 0, 0.9, [1.0] + [0.0] * 1023, tokens=["import", "os"])
    b = mk_candidate("t2", 0, 0.4, [0.0, 1.0] + [0.0] * 1022, tokens=["print", "input"])
    out = deduplicate([a, b], cosine_threshold=0.95, minhash_threshold=0.9)
    assert len(out) == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/module3/test_dedup.py -k minhash -v`
Expected: FAIL（`deduplicate` 无 `minhash_threshold` 参数）

- [ ] **Step 3: Implement**

修改 `src/module3/dedup.py`——加 MinHash 层。token 源：`ScoredCandidate` 无 token 字段，测试用 `_tokens` 属性注入；生产用 `bm25_tokens`（本任务从 candidate 的 `_tokens` 读，缺失回退空集，跳过 MinHash 层）。

顶部 import 增加：

```python
from datasketch import MinHash
```

加辅助与阈值逻辑：

```python
def _minhash(tokens: list[str], num_perm: int = 64) -> MinHash | None:
    if not tokens:
        return None
    m = MinHash(num_perm=num_perm)
    for t in tokens:
        m.update(t.encode("utf-8"))
    return m


def _tokens_of(c) -> list[str]:
    # production: bm25_tokens attached upstream; test: _tokens helper.
    return getattr(c, "_tokens", None) or []
```

`deduplicate` 签名加 `minhash_threshold`，循环内补 MinHash 判定：

```python
def deduplicate(
    candidates: list[ScoredCandidate],
    *,
    cosine_threshold: float = 0.95,
    minhash_threshold: float = 0.9,
) -> list[ScoredCandidate]:
    if not candidates:
        return []
    ordered = sorted(candidates, key=lambda c: c.relevance_score, reverse=True)
    kept: list[ScoredCandidate] = []
    kept_vecs: list[np.ndarray] = []
    kept_minhashes: list[MinHash | None] = []

    for c in ordered:
        vec = np.asarray(c.embedding, dtype=np.float32)
        mh = _minhash(_tokens_of(c))
        cos_dup = any(_cosine(vec, kv) > cosine_threshold for kv in kept_vecs)
        mh_dup = mh is not None and any(
            km is not None and mh.jaccard(km) >= minhash_threshold
            for km in kept_minhashes
        )
        if not (cos_dup or mh_dup):
            kept.append(c)
            kept_vecs.append(vec)
            kept_minhashes.append(mh)
    return kept
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/module3/test_dedup.py -v`
Expected: PASS（embedding 层 + MinHash 层全部）

- [ ] **Step 5: Checkpoint**

Run: `.venv/bin/python -m pytest -q`
Expected: 全绿。

---

## Task 9: 模块 3 —— submodular 覆盖优选

**Files:**
- Create: `src/module3/selection.py`
- Modify: `src/module3/__init__.py`（导出 `select_set`）
- Test: `tests/module3/test_selection.py`

facility-location 贪心（spec §4.2）：`f(S)=Σ_p min(cover(p,S),cap_p)+λ·diversity(S)`，预算 N，约束 `cap_p`/`min_per_problem`。

- [ ] **Step 1: Write the failing test**

创建 `tests/module3/test_selection.py`：

```python
from module3.selection import select_set, SelectionConfig


def _emb(i):
    v = [0.0] * 1024
    v[i % 1024] = 1.0
    return v


def _mk(mk_candidate, traj, idx, score, sub, embi):
    return mk_candidate(traj, idx, score, _emb(embi), sub=sub)


def test_budget_respected(mk_candidate):
    cands = [_mk(mk_candidate, f"t{i}", 0, 0.5 + i * 0.01, "p1", i) for i in range(20)]
    out = select_set(cands, sub_problem_ids=["p1"], config=SelectionConfig(n=5))
    assert len(out) == 5


def test_coverage_spreads_across_subproblems(mk_candidate):
    # 10 high-score p1, 3 low-score p2 → selection must not starve p2
    p1 = [_mk(mk_candidate, f"a{i}", 0, 0.9, "p1", i) for i in range(10)]
    p2 = [_mk(mk_candidate, f"b{i}", 0, 0.3, "p2", 100 + i) for i in range(3)]
    out = select_set(p1 + p2, sub_problem_ids=["p1", "p2"],
                     config=SelectionConfig(n=6, min_per_problem=2))
    subs = [c.sub_problem_id for c in out]
    assert subs.count("p2") >= 2  # min_per_problem honored


def test_cap_prevents_single_problem_monopoly(mk_candidate):
    p1 = [_mk(mk_candidate, f"a{i}", 0, 0.9, "p1", i) for i in range(20)]
    out = select_set(p1, sub_problem_ids=["p1"],
                     config=SelectionConfig(n=10, cap_per_problem=4))
    assert sum(1 for c in out if c.sub_problem_id == "p1") <= 4


def test_fewer_candidates_than_budget(mk_candidate):
    cands = [_mk(mk_candidate, "t1", 0, 0.5, "p1", 0)]
    out = select_set(cands, sub_problem_ids=["p1"], config=SelectionConfig(n=5))
    assert len(out) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/module3/test_selection.py -v`
Expected: FAIL（`ModuleNotFoundError: module3.selection`）

- [ ] **Step 3: Implement**

创建 `src/module3/selection.py`：

```python
"""Module 3 step 2: submodular coverage selection (facility location).

Greedy maximization under budget N (design §4.2):
  f(S) = Σ_p min(cover(p,S), cap_p) + λ·diversity(S)
Diminishing returns → automatically balances coverage and diversity.
Constraints: total==N (hard), cap_per_problem, min_per_problem.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from module2.models import ScoredCandidate

__all__ = ["select_set", "SelectionConfig"]


@dataclass
class SelectionConfig:
    n: int                              # total budget (hard)
    cap_per_problem: int | None = None  # max slices per sub-problem
    min_per_problem: int = 0            # floor per sub-problem
    lam: float = 0.1                    # diversity weight λ


def _diversity_gain(vec: np.ndarray, chosen_vecs: list[np.ndarray]) -> float:
    if not chosen_vecs:
        return 0.0
    # sum of distances to already-chosen (encourages spread)
    return float(sum(np.linalg.norm(vec - cv) for cv in chosen_vecs))


def select_set(
    candidates: list[ScoredCandidate],
    *,
    sub_problem_ids: list[str],
    config: SelectionConfig,
) -> list[ScoredCandidate]:
    """Greedy facility-location selection under budget."""
    if not candidates:
        return []

    n = min(config.n, len(candidates))
    cap = config.cap_per_problem
    remaining = list(candidates)
    chosen: list[ScoredCandidate] = []
    chosen_vecs: list[np.ndarray] = []
    per_problem: dict[str, int] = {p: 0 for p in sub_problem_ids}

    # Phase A: satisfy min_per_problem floor first (highest relevance per problem).
    if config.min_per_problem > 0:
        for p in sub_problem_ids:
            pool = sorted(
                [c for c in remaining if c.sub_problem_id == p],
                key=lambda c: c.relevance_score, reverse=True,
            )
            for c in pool[: config.min_per_problem]:
                if len(chosen) >= n:
                    break
                chosen.append(c)
                chosen_vecs.append(np.asarray(c.embedding, dtype=np.float32))
                per_problem[c.sub_problem_id] = per_problem.get(c.sub_problem_id, 0) + 1
                remaining.remove(c)

    # Phase B: greedy marginal-gain fill.
    while len(chosen) < n and remaining:
        best, best_gain = None, -1.0
        for c in remaining:
            if cap is not None and per_problem.get(c.sub_problem_id, 0) >= cap:
                continue
            vec = np.asarray(c.embedding, dtype=np.float32)
            gain = c.relevance_score + config.lam * _diversity_gain(vec, chosen_vecs)
            if gain > best_gain:
                best, best_gain = c, gain
        if best is None:
            break  # all remaining hit their cap
        chosen.append(best)
        chosen_vecs.append(np.asarray(best.embedding, dtype=np.float32))
        per_problem[best.sub_problem_id] = per_problem.get(best.sub_problem_id, 0) + 1
        remaining.remove(best)

    return chosen
```

更新 `src/module3/__init__.py`：

```python
from module3.dedup import deduplicate
from module3.selection import select_set, SelectionConfig

__all__ = ["deduplicate", "select_set", "SelectionConfig"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/module3/test_selection.py -v`
Expected: PASS

- [ ] **Step 5: Checkpoint**

Run: `.venv/bin/python -m pytest -q`
Expected: 全绿。

---

## Task 10: 模块 3 —— 配比与通用数据钩子

**Files:**
- Create: `src/module3/compose.py`
- Modify: `src/module3/__init__.py`（导出 `compose_dataset`、`GeneralDataConfig`）
- Test: `tests/module3/test_compose.py`

通用数据仅配置化钩子（spec §4.3 决策 5）：`source_path=None` 时跳过并 log，不实现数据源。

- [ ] **Step 1: Write the failing test**

创建 `tests/module3/test_compose.py`：

```python
import logging
from module3.compose import compose_dataset, GeneralDataConfig


def test_no_general_source_returns_targeted_only(mk_candidate, caplog):
    targeted = [mk_candidate(f"t{i}", 0, 0.5, [0.0] * 1024) for i in range(5)]
    with caplog.at_level(logging.INFO):
        out = compose_dataset(targeted, general_config=GeneralDataConfig(ratio=0.3, source_path=None))
    assert out["targeted"] == targeted
    assert out["general"] == []
    assert any("general data source" in r.message.lower() for r in caplog.records)


def test_ratio_recorded_in_manifest(mk_candidate):
    targeted = [mk_candidate("t1", 0, 0.5, [0.0] * 1024)]
    out = compose_dataset(targeted, general_config=GeneralDataConfig(ratio=0.25, source_path=None))
    assert out["manifest"]["general_ratio"] == 0.25
    assert out["manifest"]["targeted_count"] == 1
    assert out["manifest"]["general_count"] == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/module3/test_compose.py -v`
Expected: FAIL（`ModuleNotFoundError: module3.compose`）

- [ ] **Step 3: Implement**

创建 `src/module3/compose.py`：

```python
"""Module 3 step 3: ratio composition + general-data splice hook.

Design §4.3 / decision 5: the general SWE data source is NOT implemented in
this batch — only the splice hook. When source_path is None we log and skip,
returning targeted slices only, plus a manifest recording the intended ratio.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from module2.models import ScoredCandidate

__all__ = ["compose_dataset", "GeneralDataConfig"]

_log = logging.getLogger(__name__)


@dataclass
class GeneralDataConfig:
    ratio: float = 0.25          # intended fraction of general data (0.2–0.3)
    source_path: str | None = None


def compose_dataset(
    targeted: list[ScoredCandidate],
    *,
    general_config: GeneralDataConfig,
) -> dict:
    """Assemble final dataset: targeted slices + (optional) general data.

    Returns {targeted, general, manifest}. General data is only spliced when a
    source_path is configured; otherwise skipped with a log line.
    """
    general: list = []
    if general_config.source_path is None:
        _log.info(
            "No general data source configured (ratio=%.2f intended); "
            "returning targeted slices only.", general_config.ratio,
        )
    else:
        # NOTE: real general-data loading is a second-version task (decision 5).
        _log.warning(
            "general_config.source_path set (%s) but general-data loading is "
            "not implemented in this batch; skipping.", general_config.source_path,
        )

    return {
        "targeted": targeted,
        "general": general,
        "manifest": {
            "targeted_count": len(targeted),
            "general_count": len(general),
            "general_ratio": general_config.ratio,
        },
    }
```

更新 `src/module3/__init__.py`：

```python
from module3.dedup import deduplicate
from module3.selection import select_set, SelectionConfig
from module3.compose import compose_dataset, GeneralDataConfig

__all__ = [
    "deduplicate", "select_set", "SelectionConfig",
    "compose_dataset", "GeneralDataConfig",
]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/module3/test_compose.py -v`
Expected: PASS

- [ ] **Step 5: Checkpoint**

Run: `.venv/bin/python -m pytest -q`
Expected: 全绿。

---

## Task 11: 模块 3 端到端串联函数

**Files:**
- Create: `src/module3/pipeline.py`
- Modify: `src/module3/__init__.py`（导出 `select_final_dataset`）
- Test: `tests/module3/test_pipeline.py`

把 dedup → select → compose 串成一个入口，供 e2e/上层调用。

- [ ] **Step 1: Write the failing test**

创建 `tests/module3/test_pipeline.py`：

```python
from module3.pipeline import select_final_dataset
from module3.selection import SelectionConfig
from module3.compose import GeneralDataConfig


def test_end_to_end_dedup_select_compose(mk_candidate):
    # 6 candidates across p1/p2, two of them near-duplicate embeddings
    cands = [
        mk_candidate("t1", 0, 0.9, [1.0] + [0.0] * 1023, sub="p1"),
        mk_candidate("t2", 0, 0.4, [0.999] + [0.0] * 1023, sub="p1"),  # dup of t1
        mk_candidate("t3", 0, 0.8, [0.0, 1.0] + [0.0] * 1022, sub="p1"),
        mk_candidate("t4", 0, 0.7, [0.0, 0.0, 1.0] + [0.0] * 1021, sub="p2"),
        mk_candidate("t5", 0, 0.6, [0.0, 0.0, 0.0, 1.0] + [0.0] * 1020, sub="p2"),
        mk_candidate("t6", 0, 0.5, [0.0] * 4 + [1.0] + [0.0] * 1019, sub="p2"),
    ]
    out = select_final_dataset(
        cands, sub_problem_ids=["p1", "p2"],
        selection=SelectionConfig(n=4, min_per_problem=1),
        general=GeneralDataConfig(ratio=0.3, source_path=None),
    )
    # t2 deduped away; then N=4 selected; manifest present
    assert out["manifest"]["targeted_count"] == 4
    ids = {c.trajectory_id for c in out["targeted"]}
    assert "t2" not in ids  # lower-relevance duplicate removed
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/module3/test_pipeline.py -v`
Expected: FAIL（`ModuleNotFoundError: module3.pipeline`）

- [ ] **Step 3: Implement**

创建 `src/module3/pipeline.py`：

```python
"""Module 3 orchestration: dedup → submodular select → compose."""

from __future__ import annotations

from module2.models import ScoredCandidate
from module3.dedup import deduplicate
from module3.selection import select_set, SelectionConfig
from module3.compose import compose_dataset, GeneralDataConfig

__all__ = ["select_final_dataset"]


def select_final_dataset(
    candidates: list[ScoredCandidate],
    *,
    sub_problem_ids: list[str],
    selection: SelectionConfig,
    general: GeneralDataConfig,
    cosine_threshold: float = 0.95,
    minhash_threshold: float = 0.9,
) -> dict:
    """Full module 3: returns {targeted, general, manifest}."""
    deduped = deduplicate(
        candidates, cosine_threshold=cosine_threshold, minhash_threshold=minhash_threshold,
    )
    selected = select_set(deduped, sub_problem_ids=sub_problem_ids, config=selection)
    return compose_dataset(selected, general_config=general)
```

更新 `src/module3/__init__.py` 追加：

```python
from module3.pipeline import select_final_dataset
```

并把 `select_final_dataset` 加入 `__all__`。

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/module3/test_pipeline.py -v`
Expected: PASS

- [ ] **Step 5: Checkpoint**

Run: `.venv/bin/python -m pytest -q`
Expected: 全绿。

---

## Task 12: 端到端串联验证（模块 0→1→2→3）

**Files:**
- Modify: `scripts/e2e_smoke.py`（若存在则扩展；否则跳过，改用集成测试）
- Create: `tests/integration/test_2_3_chain.py`、`tests/integration/__init__.py`

用 FakeGateway + 真实 fixture 轨迹，串 `run_scored`（模块1→2）→ `select_final_dataset`（模块3），验证链路产出非空 N 条。

- [ ] **Step 1: Write the failing test**

创建 `tests/integration/__init__.py`（空）、`tests/integration/test_2_3_chain.py`：

```python
import pathlib
from module1.pipeline import TrajectoryPipeline, PipelineConfig
from module3.pipeline import select_final_dataset
from module3.selection import SelectionConfig
from module3.compose import GeneralDataConfig

FIXTURES = pathlib.Path(__file__).parent.parent.parent / "fixtures"


class FakeGateway:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    async def call(self, messages, model, **kwargs):
        self.calls.append((messages, model))
        resp = self._responses.pop(0) if self._responses else \
            '[{"match": false, "confidence": 0.1, "spans": [], "reasoning": "d"}]'
        return resp, {"status_code": 200, "prompt_tokens": 10, "completion_tokens": 5}


async def test_module_2_3_chain_produces_dataset():
    spec = {
        "raw_input": "写入py文件有语法错误", "domain": "agentic_swe",
        "sub_problems": [{
            "id": "p1", "origin": "original", "parent_id": None,
            "raw_text": "x", "failure_summary": "y",
            "target_capability": ["valid_syntax_in_toolcall"],
            "trajectory_signal": "SyntaxError", "hyde_positive": ["a", "b"],
            "keywords": ["SyntaxError", "python"],
            "structured_filters": {"languages": ["python"]},
            "confidence": 0.9, "route": "pass",
        }],
    }
    judge_ok = '[{"match": true, "confidence": 0.9, "spans": [{"start_step": 0, "end_step": 3}], "reasoning": "ok"}]'
    gw = FakeGateway([judge_ok] * 50)
    cfg = PipelineConfig(judge_model="test-model", recall_top_n=10)
    pipeline = TrajectoryPipeline(config=cfg, gateway=gw)

    scored = await pipeline.run_scored(
        trajectory_paths=[FIXTURES / "trajectories" / "sample_01.jsonl"],
        problem_specs=[spec],
    )
    assert scored  # module 2 produced candidates

    result = select_final_dataset(
        scored, sub_problem_ids=["p1"],
        selection=SelectionConfig(n=3, min_per_problem=1),
        general=GeneralDataConfig(ratio=0.3, source_path=None),
    )
    assert result["manifest"]["targeted_count"] >= 1
    assert result["manifest"]["general_count"] == 0
    assert all(hasattr(c, "loss_mask_spans") for c in result["targeted"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/integration/test_2_3_chain.py -v`
Expected: FAIL 或 error（若模块方法未全就位）；就位后应 PASS。

- [ ] **Step 3: Fix any wiring gaps**

若失败，按报错定位是 recall（模块1）/软加分（模块2）/选择（模块3）哪一环，对照 spec §6 数据流修正。无新代码需要时本步为空。

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/integration/test_2_3_chain.py -v`
Expected: PASS

- [ ] **Step 5: Final Checkpoint**

Run: `.venv/bin/python -m pytest -q`
Expected: 全绿（原 173 + 新增全部）。清理任何临时文件。

---

## Verification（全批完成后）

端到端手动验证（可选，需真实 LLM + embedding）：扩展 `scripts/e2e_smoke.py` 串 0→1→2→3，打印各阶段中间产物——模块 2 的 relevance 排序、模块 3 的去重前后条数 / submodular 选出的 N 条 / manifest。

自动化验证：`.venv/bin/python -m pytest -q` 全绿，覆盖：
- 模块 2 软加分（命中 ×1、未命中 ×0.3 不砍、排序正确）
- 模块 3 去重（embedding + MinHash 两层）、submodular（预算 N / cap / min_per_problem）、compose 钩子（source_path=None 跳过并 log）
- 存储接口（`RecallHit` 带分、`update_labels` 回写）
- 端到端 2+3 链路产出非空数据集

---

## Self-Review 结果

**Spec 覆盖**：§2 存储接口→Task 3/4/6；§3 模块2软加分→Task 5/7；§4.1 去重→Task 8/8b；§4.2 submodular→Task 9；§4.3 配比钩子→Task 10；§6 数据流→Task 11/12。§5（模块0.5）不在本批（§6.5 第二批），无遗漏。

**占位符扫描**：无 TBD/TODO；每个 code 步骤含完整代码与预期输出。

**类型一致性**：`RecallHit(signature, rrf_score)`、`ScoredCandidate`（9 字段）、`SelectionConfig`、`GeneralDataConfig` 在跨任务引用处签名一致；`recall()` 返回 `list[RecallHit]` 在 Task 4 定义、Task 6/7 消费一致；`run()` 旧契约保留、`run_scored()` 新增并存，模块1现有测试不破。
