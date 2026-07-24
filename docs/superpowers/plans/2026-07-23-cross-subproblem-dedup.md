# 跨子问题去重 + 合并掩码 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让"最终入选 SFT 样本"按 `(trajectory_id, slice_index)` 跨子问题去重——同一 slice 被多个子问题选中时算 1 条、掩码取并集，`manifest.targeted_count` 反映真实唯一样本数。

**Architecture:** 在 module3 引入 `MergedCandidate`（`module3/merge.py`），在 `select_final_dataset` 里于 dedup **之前**把共享 `(traj, slice)` 的 `ScoredCandidate` 塌缩为多归属候选（`sub_problem_ids` 列表 + 并集掩码 + 分子问题相关度）。`selection.py` 的覆盖度/配额记账、`dedup.py` 的近重塌缩全部改为多归属语义。`viewmodel.py` 的 `selected_keys` 从 3-tuple 降为 2-tuple。证据（`evidence_step`/`criteria_hit`）不经 `MergedCandidate`，仍由 viewmodel 从 `scored` 侧读取。

**Tech Stack:** Python 3.11+，numpy，datasketch(MinHash)，pytest。

**参考 spec:** `docs/superpowers/specs/2026-07-23-dedup-evidence-db-management-design.md` §4、§5。

---

## 关键前提（实现前已核验，见 spec §4）

- `ScoredCandidate`（`module2/models.py`）：同一 slice 跨子问题时 `embedding`/`bm25_tokens` 相同，`relevance_score`/`loss_mask_spans` 不同。
- viewmodel 的 hit 来自 `scored`（per-sub_problem），证据字段天然按子问题正确 → `MergedCandidate` **不带**证据字段。`targeted` 仅供 `selected` 布尔。
- `targeted` 下游消费者：`viewmodel.py:40`（本计划改）、`compose.py:46`（口径不变）、`scripts/e2e_smoke.py:253`（读 `sub_problem_id`，本计划改为 `sub_problem_ids`）、`scripts/complex_smoke_report.py`（只读 `trajectory_id`，不动）。
- `test_dedup.py:58` `test_same_slice_kept_when_it_covers_different_subproblems` **断言旧行为**（同 slice 两子问题保留 2 条）——本计划**故意推翻**，Task 4 重写它。

---

## File Structure

- **Create** `src/module3/merge.py` — `MergedCandidate` dataclass + `merge_by_slice()` + `_absorb()` 合并辅助（供 dedup 复用）。
- **Modify** `src/module3/pipeline.py` — `select_final_dataset` 插入 `merge_by_slice`。
- **Modify** `src/module3/selection.py` — 覆盖度/配额多归属记账。
- **Modify** `src/module3/dedup.py` — 近重塌缩改多归属守卫 + 吸收。
- **Modify** `src/service/viewmodel.py` — `selected_keys` 2-tuple。
- **Modify** `src/service/orchestrator.py` — 预算 `n` 口径。
- **Modify** 测试：`tests/module3/{test_dedup,test_selection,test_pipeline}.py`、`tests/service/test_viewmodel.py`。
- **Modify** `scripts/e2e_smoke.py` — `sub_problem_id` → `sub_problem_ids`。

---

## Task 1: MergedCandidate + merge_by_slice

**Files:**
- Create: `src/module3/merge.py`
- Test: `tests/module3/test_merge.py`

- [ ] **Step 1: 写失败测试**

创建 `tests/module3/test_merge.py`：

```python
from module3.merge import MergedCandidate, merge_by_slice


def _sc(traj, idx, sub, score, spans, emb=None):
    # 用 tests/module3/conftest.py 的 mk_candidate 也可，这里直接造一个轻对象
    class C:
        pass
    c = C()
    c.trajectory_id = traj
    c.slice_index = idx
    c.trajectory_path = "/x.jsonl"
    c.sub_problem_id = sub
    c.capability = ["cap_" + sub]
    c.relevance_score = score
    c.judge_confidence = 0.8
    c.judge_match = True
    c.loss_mask_spans = spans
    c.embedding = emb or ([1.0] + [0.0] * 1023)
    c.bm25_tokens = ["tok"]
    return c


def test_single_candidate_becomes_single_merged():
    out = merge_by_slice([_sc("t1", 0, "p1", 0.9, [{"start_step": 0, "end_step": 1}])])
    assert len(out) == 1
    m = out[0]
    assert isinstance(m, MergedCandidate)
    assert m.trajectory_id == "t1"
    assert m.slice_index == 0
    assert m.sub_problem_ids == ["p1"]
    assert m.relevance_by_problem == {"p1": 0.9}
    assert m.relevance_score == 0.9


def test_same_slice_two_subproblems_merges_to_one():
    a = _sc("t1", 0, "p1", 0.9, [{"start_step": 0, "end_step": 1}])
    b = _sc("t1", 0, "p2", 0.7, [{"start_step": 2, "end_step": 3}])
    out = merge_by_slice([a, b])
    assert len(out) == 1
    m = out[0]
    assert m.sub_problem_ids == ["p1", "p2"]           # 首次出现顺序
    assert m.relevance_by_problem == {"p1": 0.9, "p2": 0.7}
    assert m.relevance_score == 0.9                     # max
    # 掩码并集，按 (start,end) 去重后升序
    assert m.loss_mask_spans == [
        {"start_step": 0, "end_step": 1},
        {"start_step": 2, "end_step": 3},
    ]


def test_span_union_dedups_identical():
    a = _sc("t1", 0, "p1", 0.9, [{"start_step": 0, "end_step": 1}])
    b = _sc("t1", 0, "p2", 0.7, [{"start_step": 0, "end_step": 1}])
    out = merge_by_slice([a, b])
    assert out[0].loss_mask_spans == [{"start_step": 0, "end_step": 1}]


def test_distinct_slices_stay_separate():
    a = _sc("t1", 0, "p1", 0.9, [{"start_step": 0, "end_step": 1}])
    b = _sc("t2", 0, "p1", 0.7, [{"start_step": 0, "end_step": 1}])
    out = merge_by_slice([a, b])
    assert len(out) == 2
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/module3/test_merge.py -v`
Expected: FAIL（`ModuleNotFoundError: No module named 'module3.merge'`）

- [ ] **Step 3: 实现 `src/module3/merge.py`**

```python
"""Module 3 跨子问题合并：把共享 (trajectory_id, slice_index) 的候选塌缩为多归属候选。"""
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 dengxuanliang

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = ["MergedCandidate", "merge_by_slice", "absorb"]


@dataclass
class MergedCandidate:
    """多归属候选：一个 (trajectory_id, slice_index) 唯一样本，可覆盖多个子问题。

    字段名与 ScoredCandidate 尽量兼容，供 selection/dedup/compose duck-typing 读。
    """
    trajectory_id: str
    slice_index: int
    trajectory_path: str
    sub_problem_ids: list[str]
    relevance_by_problem: dict[str, float]
    relevance_score: float
    loss_mask_spans: list[dict]
    embedding: list[float]
    bm25_tokens: list[str]
    capability: list[str] = field(default_factory=list)
    judge_match: bool = True


def _union_spans(spans_lists: list[list[dict]]) -> list[dict]:
    """并集去重（按 start_step/end_step）后升序。"""
    seen: dict[tuple, dict] = {}
    for spans in spans_lists:
        for s in spans or []:
            seen[(s["start_step"], s["end_step"])] = s
    return [seen[k] for k in sorted(seen)]


def _new_from(c: Any) -> MergedCandidate:
    """从一个 ScoredCandidate-like 起一个 MergedCandidate（单归属）。"""
    sub = c.sub_problem_id
    return MergedCandidate(
        trajectory_id=c.trajectory_id,
        slice_index=c.slice_index,
        trajectory_path=getattr(c, "trajectory_path", ""),
        sub_problem_ids=[sub],
        relevance_by_problem={sub: c.relevance_score},
        relevance_score=c.relevance_score,
        loss_mask_spans=list(c.loss_mask_spans or []),
        embedding=list(getattr(c, "embedding", []) or []),
        bm25_tokens=list(getattr(c, "bm25_tokens", []) or []),
        capability=list(getattr(c, "capability", []) or []),
        judge_match=True,
    )


def _add_member(m: MergedCandidate, c: Any) -> None:
    """把一个 ScoredCandidate-like 并入已有 MergedCandidate。"""
    sub = c.sub_problem_id
    if sub not in m.relevance_by_problem:
        m.sub_problem_ids.append(sub)
        m.relevance_by_problem[sub] = c.relevance_score
    else:
        m.relevance_by_problem[sub] = max(m.relevance_by_problem[sub], c.relevance_score)
    m.relevance_score = max(m.relevance_by_problem.values())
    m.loss_mask_spans = _union_spans([m.loss_mask_spans, list(c.loss_mask_spans or [])])
    for cap in getattr(c, "capability", []) or []:
        if cap not in m.capability:
            m.capability.append(cap)


def merge_by_slice(candidates: list[Any]) -> list[MergedCandidate]:
    """按 (trajectory_id, slice_index) 塌缩为多归属候选。保持首次出现顺序。"""
    merged: dict[tuple, MergedCandidate] = {}
    order: list[tuple] = []
    for c in candidates:
        key = (c.trajectory_id, c.slice_index)
        if key not in merged:
            merged[key] = _new_from(c)
            order.append(key)
        else:
            _add_member(merged[key], c)
    return [merged[k] for k in order]


def absorb(survivor: MergedCandidate, dropped: MergedCandidate) -> None:
    """dedup 近重塌缩：把 dropped 的归属/相关度/掩码并入 survivor（保覆盖）。"""
    for sub in dropped.sub_problem_ids:
        r = dropped.relevance_by_problem[sub]
        if sub not in survivor.relevance_by_problem:
            survivor.sub_problem_ids.append(sub)
            survivor.relevance_by_problem[sub] = r
        else:
            survivor.relevance_by_problem[sub] = max(survivor.relevance_by_problem[sub], r)
    survivor.relevance_score = max(survivor.relevance_by_problem.values())
    survivor.loss_mask_spans = _union_spans([survivor.loss_mask_spans, dropped.loss_mask_spans])
    for cap in dropped.capability:
        if cap not in survivor.capability:
            survivor.capability.append(cap)
```

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/module3/test_merge.py -v`
Expected: PASS（4 passed）

- [ ] **Step 5: 提交**

```bash
git add src/module3/merge.py tests/module3/test_merge.py
git commit -m "feat(module3): MergedCandidate + merge_by_slice（跨子问题按 slice 塌缩）

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Task 2: selection.py 覆盖度/配额多归属记账

**Files:**
- Modify: `src/module3/selection.py`（`select_set` 全函数）
- Test: `tests/module3/test_selection.py`

现状 `select_set`（`selection.py:34-113`）以单个 `candidate.sub_problem_id` 记账。多归属候选（`MergedCandidate`）有 `sub_problem_ids` 列表 + `relevance_by_problem` 字典，没有 `sub_problem_id` 属性。

- [ ] **Step 1: 写失败测试（多归属记账语义）**

追加到 `tests/module3/test_selection.py` 末尾：

```python
from module3.merge import MergedCandidate


def _merged(traj, idx, rel_by_problem, embi):
    v = [0.0] * 1024
    v[embi % 1024] = 1.0
    return MergedCandidate(
        trajectory_id=traj, slice_index=idx, trajectory_path="/x.jsonl",
        sub_problem_ids=list(rel_by_problem.keys()),
        relevance_by_problem=dict(rel_by_problem),
        relevance_score=max(rel_by_problem.values()),
        loss_mask_spans=[{"start_step": 0, "end_step": 1}],
        embedding=v, bm25_tokens=[],
    )


def test_shared_slice_satisfies_both_min_per_problem():
    # 唯一共享 slice 同时属于 p1、p2；min_per_problem=1 时选它 1 条即满足两个保底
    shared = _merged("t1", 0, {"p1": 0.9, "p2": 0.8}, 0)
    out = select_set([shared], sub_problem_ids=["p1", "p2"],
                     config=SelectionConfig(n=5, min_per_problem=1))
    assert len(out) == 1
    assert set(out[0].sub_problem_ids) == {"p1", "p2"}


def test_multi_attribution_not_double_appended_in_min_phase():
    # 保底阶段遍历 sub_problem_ids 不能把同一 slice append 两次
    shared = _merged("t1", 0, {"p1": 0.9, "p2": 0.8}, 0)
    extra = _merged("t2", 0, {"p1": 0.5}, 1)
    out = select_set([shared, extra], sub_problem_ids=["p1", "p2"],
                     config=SelectionConfig(n=5, min_per_problem=1))
    keys = [(c.trajectory_id, c.slice_index) for c in out]
    assert len(keys) == len(set(keys))   # 无重复


def test_cap_blocks_when_any_subproblem_saturated():
    # cap_per_problem=1：p1 先被一条占满后，属于 p1 的其它候选被挡
    a = _merged("t1", 0, {"p1": 0.9}, 0)
    b = _merged("t2", 0, {"p1": 0.8}, 1)
    out = select_set([a, b], sub_problem_ids=["p1"],
                     config=SelectionConfig(n=5, cap_per_problem=1))
    assert len(out) == 1
```

同时**改造**该文件已有的老测试（它们造的是单归属 `mk_candidate`，有 `sub_problem_id` 无 `sub_problem_ids`）——见 Step 4。

- [ ] **Step 2: 跑新测试确认失败**

Run: `uv run pytest tests/module3/test_selection.py -k "shared_slice or multi_attribution or cap_blocks" -v`
Expected: FAIL（`AttributeError: 'MergedCandidate' object has no attribute 'sub_problem_id'` 之类）

- [ ] **Step 3: 改写 `select_set` 为多归属记账**

把 `src/module3/selection.py` 的 `select_set` 整体替换为：

```python
def select_set(
    candidates: list[Any],
    *,
    sub_problem_ids: list[str],
    config: SelectionConfig,
) -> list[Any]:
    """Select candidates under budget, minimum coverage, and optional caps.

    候选为多归属（MergedCandidate）：以 sub_problem_ids/relevance_by_problem 记账。
    一个候选被选中即同时计入它覆盖的所有子问题的 counts/cover。
    """
    if not candidates or config.n <= 0:
        return []

    budget = min(config.n, len(candidates))
    remaining = list(candidates)
    chosen: list[Any] = []
    chosen_vecs: list[np.ndarray] = []
    chosen_keys: set[tuple] = set()   # (traj, slice) 去重，防多归属重复 append
    counts = {sub_id: 0 for sub_id in sub_problem_ids}
    cover = {sub_id: 0.0 for sub_id in sub_problem_ids}

    def _rel(c, sub_id):
        return c.relevance_by_problem.get(sub_id, 0.0)

    def _capped(c):
        # 任一归属子问题达 cap → 挡下（保守）
        if config.cap_per_problem is None:
            return False
        return any(counts.get(sid, 0) >= config.cap_per_problem
                   for sid in c.sub_problem_ids)

    def _commit(c):
        chosen.append(c)
        chosen_vecs.append(_vector(c))
        chosen_keys.add((c.trajectory_id, c.slice_index))
        for sid in c.sub_problem_ids:
            counts[sid] = counts.get(sid, 0) + 1
            cover[sid] = cover.get(sid, 0.0) + _rel(c, sid)
        remaining.remove(c)

    if config.min_per_problem > 0:
        for sub_id in sub_problem_ids:
            pool = sorted(
                [c for c in remaining if sub_id in c.sub_problem_ids],
                key=lambda c: _rel(c, sub_id),
                reverse=True,
            )
            taken = 0
            for candidate in pool:
                if taken >= config.min_per_problem or len(chosen) >= budget:
                    break
                if (candidate.trajectory_id, candidate.slice_index) in chosen_keys:
                    taken += 1          # 已被选（可能因别的子问题），计入该子问题保底
                    continue
                if _capped(candidate):
                    continue
                _commit(candidate)
                taken += 1

    while len(chosen) < budget and remaining:
        best = None
        best_gain = float("-inf")
        for candidate in remaining:
            if _capped(candidate):
                continue
            vec = _vector(candidate)
            gain = 0.0
            for sid in candidate.sub_problem_ids:
                cur = cover.get(sid, 0.0)
                r = _rel(candidate, sid)
                if config.coverage_cap_per_problem is None:
                    gain += r
                else:
                    gain += max(0.0, min(cur + r, config.coverage_cap_per_problem) - cur)
            gain += config.lam * _diversity_gain(vec, chosen_vecs)
            if gain > best_gain:
                best = candidate
                best_gain = gain

        if best is None:
            break
        _commit(best)

    return chosen
```

- [ ] **Step 4: 迁移老测试到多归属**

`tests/module3/test_selection.py` 顶部的 `_mk` 走 `mk_candidate`（单归属，有 `sub_problem_id`）。把它改为产出 `MergedCandidate`：

```python
from module3.selection import SelectionConfig, select_set
from module3.merge import MergedCandidate


def _emb(i):
    v = [0.0] * 1024
    v[i % 1024] = 1.0
    return v


def _mk(mk_candidate, traj, idx, score, sub, embi):
    # 直接产 MergedCandidate（单归属），替代原 mk_candidate
    return MergedCandidate(
        trajectory_id=traj, slice_index=idx, trajectory_path="/x.jsonl",
        sub_problem_ids=[sub], relevance_by_problem={sub: score},
        relevance_score=score, loss_mask_spans=[{"start_step": 0, "end_step": 1}],
        embedding=_emb(embi), bm25_tokens=[],
    )
```

原有断言里读 `c.sub_problem_id` 的（如 `test_coverage_spreads_across_subproblems` 的 `subs = [c.sub_problem_id for c in out]`、`test_cap_prevents_single_problem_monopoly` 的 `c.sub_problem_id == "p1"`）改为读 `c.sub_problem_ids`：

```python
# test_coverage_spreads_across_subproblems:
subs = [sid for c in out for sid in c.sub_problem_ids]
assert subs.count("p2") >= 2

# test_cap_prevents_single_problem_monopoly:
assert sum(1 for c in out if "p1" in c.sub_problem_ids) <= 4
```

- [ ] **Step 5: 跑整个 selection 测试**

Run: `uv run pytest tests/module3/test_selection.py -v`
Expected: PASS（全部，含新旧用例）

- [ ] **Step 6: 提交**

```bash
git add src/module3/selection.py tests/module3/test_selection.py
git commit -m "feat(module3): select_set 覆盖度/配额改为多归属记账（MergedCandidate）

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Task 3: dedup.py 近重塌缩多归属 + 吸收

**Files:**
- Modify: `src/module3/dedup.py`（`deduplicate` 内塌缩逻辑）
- Test: `tests/module3/test_dedup.py`

现状 `deduplicate`（`dedup.py:37-74`）以 `ScoredCandidate` 为单位，`sub_problem_id ==` 守卫，命中即 `continue` 丢弃。改为以 `MergedCandidate` 为单位、归属集合有交集才算同域、丢弃时吸收进幸存者。

- [ ] **Step 1: 写/改失败测试**

`tests/module3/test_dedup.py` 里所有 `mk_candidate` 造的是 `ScoredCandidate`-like。dedup 现在吃 `MergedCandidate`。新增一个本地工厂并改造用例：

```python
from module3.dedup import deduplicate
from module3.merge import MergedCandidate


def _m(traj, idx, subs_rel, emb, tokens=None):
    return MergedCandidate(
        trajectory_id=traj, slice_index=idx, trajectory_path="/x.jsonl",
        sub_problem_ids=list(subs_rel.keys()), relevance_by_problem=dict(subs_rel),
        relevance_score=max(subs_rel.values()),
        loss_mask_spans=[{"start_step": 0, "end_step": 1}],
        embedding=emb, bm25_tokens=tokens or [],
    )


def test_embedding_near_duplicates_collapsed_and_absorbed():
    a = _m("t1", 0, {"p1": 0.9}, [1.0] + [0.0] * 1023)
    b = _m("t2", 0, {"p1": 0.5}, [0.999] + [0.0] * 1023)
    out = deduplicate([a, b], cosine_threshold=0.95)
    assert len(out) == 1
    assert out[0].trajectory_id == "t1"   # 高分幸存


def test_distinct_embeddings_all_kept():
    a = _m("t1", 0, {"p1": 0.9}, [1.0] + [0.0] * 1023)
    b = _m("t2", 0, {"p1": 0.5}, [0.0, 1.0] + [0.0] * 1022)
    out = deduplicate([a, b], cosine_threshold=0.95)
    assert len(out) == 2


def test_near_dup_across_subproblems_absorbs_attribution():
    # 两个近重 slice 分属 p1/p2 → 塌缩为 1 条，幸存者吸收 p2 归属（保覆盖）
    a = _m("t1", 0, {"p1": 0.9}, [1.0] + [0.0] * 1023)
    b = _m("t2", 0, {"p2": 0.5}, [0.999] + [0.0] * 1023)
    out = deduplicate([a, b], cosine_threshold=0.95)
    assert len(out) == 1
    assert set(out[0].sub_problem_ids) == {"p1", "p2"}


def test_empty_input():
    assert deduplicate([], cosine_threshold=0.95) == []
```

**删除**旧的 `test_same_slice_kept_when_it_covers_different_subproblems`（`test_dedup.py:58`）——它断言的"同 slice 两子问题保留 2 条"已被 `merge_by_slice` 在 dedup 前消解，语义作废。其正向意图（跨子问题不丢覆盖）由上面 `test_near_dup_across_subproblems_absorbs_attribution` 承接。

minhash 的两个用例（`test_minhash_*`）把 `mk_candidate(...)` 换成 `_m(...)`，token 通过 `tokens=[...]` 传入（`MergedCandidate.bm25_tokens`）。

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/module3/test_dedup.py -v`
Expected: FAIL（旧塌缩用 `sub_problem_id ==`，`MergedCandidate` 无该属性 → AttributeError；且无吸收逻辑）

- [ ] **Step 3: 改写 `deduplicate`**

**设计决策（重要）**：近重判定**不看归属**。跨子问题的近重轨迹（如 a 只属 p1、b 只属 p2 但内容近似）**应当**塌缩——同一份轨迹证据不该因挂在不同子问题下就重复进 SFT；被丢弃者的归属/掩码通过 `absorb` 并入幸存者以保覆盖。因此原 `sub_problem_id ==` 守卫**直接删除**，不用任何归属交集条件。

`src/module3/dedup.py` 的 `deduplicate` 替换为（`_cosine`/`_tokens_of`/`_minhash` 保持不变）：

```python
def deduplicate(
    candidates: list[Any],
    *,
    cosine_threshold: float = 0.95,
    minhash_threshold: float = 0.9,
) -> list[Any]:
    """Collapse near-duplicate MergedCandidates, absorbing attribution into survivor."""
    if not candidates:
        return []
    from module3.merge import absorb

    ordered = sorted(candidates, key=lambda c: c.relevance_score, reverse=True)
    kept: list[Any] = []
    kept_vectors: list[np.ndarray] = []
    kept_minhashes: list[MinHash | None] = []

    for candidate in ordered:
        vec = np.asarray(candidate.embedding, dtype=np.float32)
        minhash = _minhash(_tokens_of(candidate))
        dup_of = None
        for i, kv in enumerate(kept_vectors):
            if _cosine(vec, kv) > cosine_threshold:
                dup_of = i
                break
            existing = kept_minhashes[i]
            if (minhash is not None and existing is not None
                    and minhash.jaccard(existing) >= minhash_threshold):
                dup_of = i
                break
        if dup_of is not None:
            absorb(kept[dup_of], candidate)   # 吸收归属/掩码，保覆盖
            continue
        kept.append(candidate)
        kept_vectors.append(vec)
        kept_minhashes.append(minhash)

    return kept
```

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/module3/test_dedup.py -v`
Expected: PASS（含 `test_near_dup_across_subproblems_absorbs_attribution` → 1 条、归属 {p1,p2}）

- [ ] **Step 5: 提交**

```bash
git add src/module3/dedup.py tests/module3/test_dedup.py
git commit -m "feat(module3): dedup 近重塌缩改多归属，丢弃者吸收进幸存者（保覆盖）

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Task 4: pipeline.py 串入 merge_by_slice + viewmodel 2-tuple

**Files:**
- Modify: `src/module3/pipeline.py`（`select_final_dataset`）
- Modify: `src/service/viewmodel.py`（`selected_keys`）
- Test: `tests/module3/test_pipeline.py`、`tests/service/test_viewmodel.py`

- [ ] **Step 1: 写失败测试（pipeline 端到端计数）**

追加到 `tests/module3/test_pipeline.py`：

```python
def test_same_slice_two_subproblems_counts_once(mk_candidate):
    # 同一 (t1, slice0) 被 p1、p2 各选中 → 最终入选应为 1 条
    a = mk_candidate("t1", 0, 0.9, [1.0] + [0.0] * 1023, sub="p1")
    b = mk_candidate("t1", 0, 0.7, [1.0] + [0.0] * 1023, sub="p2")
    out = select_final_dataset(
        [a, b], sub_problem_ids=["p1", "p2"],
        selection=SelectionConfig(n=10, min_per_problem=1),
        general=GeneralDataConfig(ratio=0.3, source_path=None),
    )
    assert out["manifest"]["targeted_count"] == 1
    m = out["targeted"][0]
    assert set(m.sub_problem_ids) == {"p1", "p2"}
    # 掩码并集（mk_candidate 默认 spans 相同 → 去重后 1 段）
    assert m.loss_mask_spans == [{"start_step": 0, "end_step": 1}]
```

- [ ] **Step 2: 跑确认失败**

Run: `uv run pytest tests/module3/test_pipeline.py::test_same_slice_two_subproblems_counts_once -v`
Expected: FAIL（现在算 2 条 / `targeted` 是 ScoredCandidate 无 `sub_problem_ids`）

- [ ] **Step 3: 改 `select_final_dataset` 串入合并**

`src/module3/pipeline.py` 的 `select_final_dataset` 改为：

```python
from module3.compose import GeneralDataConfig, compose_dataset
from module3.dedup import deduplicate
from module3.merge import merge_by_slice
from module3.selection import SelectionConfig, select_set

__all__ = ["select_final_dataset"]


def select_final_dataset(
    candidates: list[Any],
    *,
    sub_problem_ids: list[str],
    selection: SelectionConfig,
    general: GeneralDataConfig,
    cosine_threshold: float = 0.95,
    minhash_threshold: float = 0.9,
) -> dict:
    """Run module3's full final dataset selection flow (跨子问题按 slice 去重)."""
    trainable = [
        c for c in candidates
        if getattr(c, "judge_match", True) and getattr(c, "loss_mask_spans", [])
    ]
    merged = merge_by_slice(trainable)          # 选择前合并：N=唯一 slice 数
    deduped = deduplicate(
        merged,
        cosine_threshold=cosine_threshold,
        minhash_threshold=minhash_threshold,
    )
    selected = select_set(deduped, sub_problem_ids=sub_problem_ids, config=selection)
    return compose_dataset(selected, general_config=general)
```

- [ ] **Step 4: 跑 pipeline 测试**

Run: `uv run pytest tests/module3/test_pipeline.py -v`
Expected: PASS。分析原 `test_end_to_end_dedup_select_compose`（`n=4`，断言 `targeted_count == 4` 且 `"t2" not in ids`）：t1-t6 各不同 traj、各单归属 → `merge_by_slice` 后仍 6 条 → `deduplicate` 把 t2（与 t1 近重）塌缩进 t1，剩 5 条 → `select_set` 预算 4 选出 4 条。t2 已被塌缩，必不在结果 → 两条断言仍成立。`ids` 用 `c.trajectory_id`（MergedCandidate 有该字段）仍可读。

- [ ] **Step 5: 改 viewmodel `selected_keys` 为 2-tuple**

`src/service/viewmodel.py` 第 39-42 行：

```python
    # (traj_id, slice_index) of module3-selected candidates（MergedCandidate 多归属，
    # 无单一 sub_problem_id；同一 slice 在其覆盖的所有子问题卡片下都算已入选）
    selected_keys = {
        (c.trajectory_id, c.slice_index)
        for c in select_result.get("targeted", [])
    }
```

第 76 行的 `selected` 判定：

```python
                    "selected": (c.trajectory_id, c.slice_index) in selected_keys,
```

- [ ] **Step 6: 改 viewmodel 测试**

`tests/service/test_viewmodel.py` 现有用 `FakeScored`（有 `trajectory_id`+`slice_index`）作 `targeted` 项，2-tuple 键仍能匹配 → 大部分自动通过。`test_selected_keyed_on_slice_index_not_just_trajectory` 的意图（按 slice 区分）在 2-tuple 下依然成立。新增一条多归属"已入选跨卡片一致"用例：

```python
def test_selected_reflected_across_subproblem_cards():
    from module3.merge import MergedCandidate
    # 同一 (t1, slice0) 覆盖 p1、p2，进 targeted → 两个子问题卡片下都 selected
    spec = {"raw_input": "x", "domain": "agentic_swe", "sub_problems": [
        {"id": "p1", "failure_summary": "a", "target_capability": ["c1"], "confidence": 0.9},
        {"id": "p2", "failure_summary": "b", "target_capability": ["c2"], "confidence": 0.9},
    ]}
    scored = [
        FakeScored("t1", 0, "p1", ["c1"], 0.9, 0.9, [{"start_step": 0, "end_step": 1}]),
        FakeScored("t1", 0, "p2", ["c2"], 0.7, 0.8, [{"start_step": 0, "end_step": 1}]),
    ]
    merged = MergedCandidate(
        trajectory_id="t1", slice_index=0, trajectory_path="/x.jsonl",
        sub_problem_ids=["p1", "p2"], relevance_by_problem={"p1": 0.9, "p2": 0.7},
        relevance_score=0.9, loss_mask_spans=[{"start_step": 0, "end_step": 1}],
        embedding=[1.0] + [0.0] * 1023, bm25_tokens=[])
    view = build_inspector_view(
        run_id="r", spec=spec, scored=scored,
        select_result={"targeted": [merged], "manifest": {"targeted_count": 1}})
    for p in view["problems"]:
        hit = p["capabilities"][0]["hit_trajectories"][0]
        assert hit["selected"] is True
```

- [ ] **Step 7: 跑 viewmodel 测试**

Run: `uv run pytest tests/service/test_viewmodel.py -v`
Expected: PASS（全部）

- [ ] **Step 8: 提交**

```bash
git add src/module3/pipeline.py src/service/viewmodel.py tests/module3/test_pipeline.py tests/service/test_viewmodel.py
git commit -m "feat(module3+viewmodel): select_final_dataset 串入 merge_by_slice；selected_keys 降为 2-tuple

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Task 5: orchestrator 预算口径 + e2e_smoke 脚本修正

**Files:**
- Modify: `src/service/orchestrator.py`（两处 `SelectionConfig(n=...)`）
- Modify: `scripts/e2e_smoke.py`（`c.sub_problem_id` → `c.sub_problem_ids`）

- [ ] **Step 1: orchestrator 预算 n 对齐唯一 slice 数**

`orchestrator.py` 有两处 `n=min(10, max(1, len(scored)))`（run_pipeline 约 245 行、run_search 约 316 行），两处都已传 `min_per_problem=1`。`len(scored)` 是合并前计数，可能虚高。改为按唯一 `(traj, slice)` 数。

在两处 `if selection_config is None or general_config is None:` 分支内，新增一行 `_n_unique` 并把 `len(scored)` 替换为它。run_pipeline 处（现约 244-246 行）：

```python
        _n_unique = len({(c.trajectory_id, c.slice_index) for c in scored})
        selection_config = selection_config or SelectionConfig(
            n=min(10, max(1, _n_unique)), min_per_problem=1
        )
```

run_search 处（现约 315-316 行）同样替换（保留其原有 `min_per_problem=1`）：

```python
        _n_unique = len({(c.trajectory_id, c.slice_index) for c in scored})
        selection_config = selection_config or SelectionConfig(
            n=min(10, max(1, _n_unique)), min_per_problem=1)
```

- [ ] **Step 2: 修 e2e_smoke.py 打印**

`scripts/e2e_smoke.py` 约 250-255 行遍历 `targeted` 打印 `sub={c.sub_problem_id}`。`targeted` 现为 `MergedCandidate`（有 `sub_problem_ids`）。改为：

```python
            for c in targeted:
                print(
                    f"  {c.trajectory_id}/slice{c.slice_index}: "
                    f"subs={c.sub_problem_ids}, relevance={c.relevance_score:.4f}, "
                    f"spans={c.loss_mask_spans}"
                )
```

- [ ] **Step 3: 全量 module3 + service 逻辑测试回归**

Run: `uv run pytest -m "not requires_model" tests/module3 tests/service -v`
Expected: PASS（全部）

- [ ] **Step 4: 提交**

```bash
git add src/service/orchestrator.py scripts/e2e_smoke.py
git commit -m "fix(orchestrator+smoke): 预算 n 对齐唯一 slice 数；e2e_smoke 改读 sub_problem_ids

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## 验收标准（对齐 spec §10）

- [ ] 两个子问题各选中同一 `(traj, slice)` → `manifest.targeted_count == 1`，`targeted[0].sub_problem_ids` 含两个子问题。
- [ ] 合并项 `loss_mask_spans` 是各子问题 spans 的去重并集且升序。
- [ ] viewmodel 下该 slice 在两个子问题卡片下 `selected` 均为 `True`。
- [ ] `min_per_problem=1`、`p1`/`p2` 仅共享一个 slice 时，选中该 1 条即同时满足两个保底。
- [ ] `scripts/e2e_smoke.py` 不再 `AttributeError`。
- [ ] `uv run pytest -m "not requires_model" tests/module3 tests/service` 全绿。
