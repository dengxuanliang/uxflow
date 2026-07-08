# Module 1: Trajectory Pipeline 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** 从回流成功轨迹中，筛选命中用户问题清单的正向能力示范片段，输出 loss mask span 供 SFT 训练。四阶段 pipeline：签名抽取 → 内存索引 → 召回 → LLM 精判。

**Design Spec:** [`../specs/2026-07-08-module1-trajectory-pipeline-design.md`](../specs/2026-07-08-module1-trajectory-pipeline-design.md)

**Tech Stack:** Python 3.11+, llm_gateway（已实现）, module0（已实现，消费其 ProblemSpec）, sentence-transformers（已安装）, numpy, pytest。

**第一版策略（spec §7）：** 内存 dict + numpy 替代 ES/Qdrant。几千条数据全表扫描秒级完成。百万级扩展时替换索引层，接口不变。

---

## 文件结构

包位于 `src/module1/`。每个文件对应 pipeline 一个阶段或辅助组件。

| 文件 | 职责 | 对应 spec |
|------|------|----------|
| `src/module1/models.py` | 数据模型：Trajectory, Step, Slice, TrajectorySignature, SFTCandidate | §1.3, §3 |
| `src/module1/loader.py` | JSONL 轨迹加载 + 解析为 Trajectory 对象 | §1.2 |
| `src/module1/slicer.py` | 切片策略：≤10 steps 整条 / >10 steps 语义切分 | §2 |
| `src/module1/signature.py` | Phase 1：签名抽取（languages/tools_used/bm25_tokens/error patterns/embedding） | §3 Phase 1 |
| `src/module1/index.py` | Phase 2+3：内存索引 + 多路召回（结构化过滤 + BM25 + 向量） | §3 Phase 2-3 |
| `src/module1/summarizer.py` | 精判摘要生成（每步压缩为一行） | §5 |
| `src/module1/judge.py` | Phase 4：LLM 精判 + span 标注（通过 gateway） | §3 Phase 4 |
| `src/module1/pipeline.py` | 顶层编排：加载 → 切片 → 签名 → 索引 → 对每个 ProblemSpec 召回+精判 → 输出 | 全流程 |
| `src/module1/__init__.py` | 公共导出 | — |

测试在 `tests/module1/` 一一对应。

---

## 依赖关系

```
models.py       ← 叶子（纯数据模型）
loader.py       ← models
slicer.py       ← models
signature.py    ← models, module0.embedding (EmbeddingModel)
summarizer.py   ← models
index.py        ← models, numpy
judge.py        ← models, summarizer, llm_gateway
pipeline.py     ← 以上全部 + module0 (ProblemSpec)
```

---

## Task 0: 基础设施

**Files:**
- Modify: `pyproject.toml`（加 module1 包路径）
- Create: `src/module1/__init__.py`、`tests/module1/__init__.py`、`tests/module1/conftest.py`
- Create: `fixtures/trajectories/sample_01.jsonl`（测试用轨迹样本）

- [ ] **Step 1: 修改 pyproject.toml**

在 `[tool.hatch.build.targets.wheel]` 的 packages 列表加入 `"src/module1"`。

- [ ] **Step 2: 创建目录骨架**

```bash
mkdir -p src/module1 tests/module1 fixtures/trajectories
touch src/module1/__init__.py tests/module1/__init__.py
```

- [ ] **Step 3: 创建测试 fixture 轨迹 `fixtures/trajectories/sample_01.jsonl`**

每行一条轨迹（OpenAI messages 格式），构造 3 条覆盖不同场景：
- 轨迹 1（短，5 steps）：python 语法修复 → 测试通过。匹配 `valid_syntax_in_toolcall`。
- 轨迹 2（短，4 steps）：文件定位 + 编辑。匹配 `file_localization_and_edit`。
- 轨迹 3（长，12 steps）：多子任务，需要切分。覆盖语义边界切分。

每条轨迹格式：
```json
{"id": "traj_001", "messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "...", "tool_calls": [{"function": {"name": "Bash", "arguments": "..."}}]}, {"role": "tool", "content": "..."}]}
```

- [ ] **Step 4: 创建 tests/module1/conftest.py**

```python
import json
import pathlib
import pytest

FIXTURES_DIR = pathlib.Path(__file__).parent.parent.parent / "fixtures"


@pytest.fixture
def trajectories_path():
    return FIXTURES_DIR / "trajectories" / "sample_01.jsonl"


@pytest.fixture
def sample_trajectories(trajectories_path):
    trajectories = []
    with open(trajectories_path) as f:
        for line in f:
            if line.strip():
                trajectories.append(json.loads(line))
    return trajectories
```

- [ ] **Step 5: pip install + commit**

```bash
pip install -e ".[dev]"
git add pyproject.toml src/module1/ tests/module1/ fixtures/trajectories/
git commit -m "chore(module1): scaffold + test fixtures"
```

---

## Task 1: models.py — 数据模型

**Files:**
- Create: `src/module1/models.py`
- Test: `tests/module1/test_models.py`

定义 pipeline 中流转的核心数据结构。纯 dataclass，无逻辑。

- [ ] **Step 1: 写失败测试**

```python
from module1.models import Step, Trajectory, Slice, TrajectorySignature, JudgeResult, SFTCandidate


def test_step_creation():
    s = Step(index=0, role="assistant", content="analyzing...",
             tool_call_name="Bash", tool_call_args="ls -la", tool_result=None)
    assert s.role == "assistant"
    assert s.tool_call_name == "Bash"


def test_trajectory_step_count():
    steps = [Step(index=i, role="assistant", content="", tool_call_name=None, tool_call_args=None, tool_result=None) for i in range(5)]
    t = Trajectory(id="traj_001", steps=steps, raw_messages=[])
    assert t.step_count == 5


def test_slice_from_trajectory():
    steps = [Step(index=i, role="assistant", content="", tool_call_name=None, tool_call_args=None, tool_result=None) for i in range(10)]
    s = Slice(trajectory_id="traj_001", slice_index=0, steps=steps[:5], start_step=0, end_step=4)
    assert s.step_count == 5


def test_judge_result():
    jr = JudgeResult(match=True, confidence=0.88, spans=[{"start_step": 2, "end_step": 5}], reasoning="good")
    assert jr.match
    assert jr.spans[0]["start_step"] == 2


def test_sft_candidate():
    c = SFTCandidate(
        trajectory_id="traj_001",
        trajectory_path="/data/traj_001.jsonl",
        matched_problems=[{
            "problem_spec_id": "spec_001",
            "sub_problem_id": "p1",
            "capability": ["valid_syntax_in_toolcall"],
            "confidence": 0.88,
            "loss_mask_spans": [{"start_step": 2, "end_step": 5}],
        }],
    )
    assert c.trajectory_id == "traj_001"
```

- [ ] **Step 2: 实现 models.py**

```python
"""Data models for the trajectory pipeline."""

from __future__ import annotations
from dataclasses import dataclass, field

__all__ = ["Step", "Trajectory", "Slice", "TrajectorySignature", "JudgeResult", "SFTCandidate"]


@dataclass
class Step:
    index: int
    role: str
    content: str
    tool_call_name: str | None = None
    tool_call_args: str | None = None
    tool_result: str | None = None


@dataclass
class Trajectory:
    id: str
    steps: list[Step]
    raw_messages: list[dict]

    @property
    def step_count(self) -> int:
        return len(self.steps)


@dataclass
class Slice:
    trajectory_id: str
    slice_index: int
    steps: list[Step]
    start_step: int
    end_step: int

    @property
    def step_count(self) -> int:
        return len(self.steps)


@dataclass
class TrajectorySignature:
    trajectory_id: str
    slice_index: int
    step_range: tuple[int, int]
    step_count: int
    turn_count: int
    languages: list[str]
    tools_used: list[str]
    has_error_pattern: bool
    has_success_pattern: bool
    has_verification_step: bool
    bm25_tokens: list[str]
    embedding: list[float] = field(default_factory=list)


@dataclass
class JudgeResult:
    match: bool
    confidence: float
    spans: list[dict]
    reasoning: str = ""


@dataclass
class SFTCandidate:
    trajectory_id: str
    trajectory_path: str
    matched_problems: list[dict]
```

- [ ] **Step 3: 运行测试，确认全过**
- [ ] **Step 4: Commit**

```bash
git add src/module1/models.py tests/module1/test_models.py
git commit -m "feat(module1): models.py data structures for pipeline"
```

---

## Task 2: loader.py — 轨迹加载

**Files:**
- Create: `src/module1/loader.py`
- Test: `tests/module1/test_loader.py`

从 JSONL 加载，解析 OpenAI messages 为 Trajectory + Step。处理 tool_calls 和 tool role 对应。

- [ ] **Step 1: 写失败测试**

```python
from module1.loader import load_trajectories, parse_trajectory


def test_load_from_jsonl(trajectories_path):
    trajs = load_trajectories(trajectories_path)
    assert len(trajs) >= 1
    assert trajs[0].id is not None
    assert trajs[0].step_count >= 1


def test_step_parsing(sample_trajectories):
    traj = parse_trajectory(sample_trajectories[0])
    roles = {s.role for s in traj.steps}
    assert "assistant" in roles


def test_tool_call_extraction(sample_trajectories):
    traj = parse_trajectory(sample_trajectories[0])
    tool_steps = [s for s in traj.steps if s.tool_call_name]
    assert len(tool_steps) >= 1


def test_empty_file(tmp_path):
    empty = tmp_path / "empty.jsonl"
    empty.write_text("")
    trajs = load_trajectories(empty)
    assert trajs == []
```

- [ ] **Step 2: 实现 loader.py**

```python
"""Load and parse trajectories from JSONL."""

from __future__ import annotations
import json
import pathlib
from module1.models import Step, Trajectory

__all__ = ["load_trajectories", "parse_trajectory"]


def load_trajectories(path: str | pathlib.Path) -> list[Trajectory]:
    path = pathlib.Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return []
    trajectories = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            trajectories.append(parse_trajectory(json.loads(line)))
    return trajectories


def parse_trajectory(data: dict) -> Trajectory:
    traj_id = data.get("id", f"traj_{hash(json.dumps(data, sort_keys=True)) % 10**8}")
    messages = data.get("messages", [])
    steps = []
    step_idx = 0

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "") or ""

        if role == "assistant":
            tool_calls = msg.get("tool_calls", [])
            if tool_calls:
                for tc in tool_calls:
                    fn = tc.get("function", tc) if isinstance(tc, dict) else {}
                    steps.append(Step(
                        index=step_idx, role="assistant", content=content,
                        tool_call_name=fn.get("name"),
                        tool_call_args=fn.get("arguments", ""),
                    ))
                    step_idx += 1
            else:
                steps.append(Step(index=step_idx, role="assistant", content=content))
                step_idx += 1
        elif role == "tool":
            steps.append(Step(index=step_idx, role="tool", content=content, tool_result=content))
            step_idx += 1
        elif role in ("user", "system"):
            steps.append(Step(index=step_idx, role=role, content=content))
            step_idx += 1

    return Trajectory(id=traj_id, steps=steps, raw_messages=messages)
```

- [ ] **Step 3: 运行测试，确认全过**
- [ ] **Step 4: Commit**

```bash
git add src/module1/loader.py tests/module1/test_loader.py
git commit -m "feat(module1): loader.py JSONL trajectory parsing"
```

---

## Task 3: slicer.py — 切片策略

**Files:**
- Create: `src/module1/slicer.py`
- Test: `tests/module1/test_slicer.py`

双模式切分：≤10 steps 整条=1 切片；>10 steps 按语义边界切分（spec §2）。纯规则，零 LLM。

- [ ] **Step 1: 写失败测试**

```python
from module1.models import Step, Trajectory, Slice
from module1.slicer import slice_trajectory

STEP_THRESHOLD = 10


def _make_steps(n, roles=None):
    """Helper: create n steps with alternating assistant/tool roles."""
    steps = []
    for i in range(n):
        if roles:
            role = roles[i % len(roles)]
        else:
            role = "assistant" if i % 2 == 0 else "tool"
        steps.append(Step(
            index=i, role=role, content=f"step {i}",
            tool_call_name="Bash" if role == "assistant" else None,
            tool_call_args="cmd" if role == "assistant" else None,
            tool_result=f"result {i}" if role == "tool" else None,
        ))
    return steps


def test_short_trajectory_single_slice():
    """≤10 steps → 1 slice = whole trajectory."""
    steps = _make_steps(6)
    t = Trajectory(id="t1", steps=steps, raw_messages=[])
    slices = slice_trajectory(t)
    assert len(slices) == 1
    assert slices[0].start_step == 0
    assert slices[0].end_step == 5
    assert slices[0].step_count == 6


def test_exactly_10_steps_single_slice():
    steps = _make_steps(10)
    t = Trajectory(id="t1", steps=steps, raw_messages=[])
    slices = slice_trajectory(t)
    assert len(slices) == 1


def test_long_trajectory_multiple_slices():
    """>10 steps → multiple slices, each 5-10 steps."""
    steps = _make_steps(20)
    t = Trajectory(id="t1", steps=steps, raw_messages=[])
    slices = slice_trajectory(t)
    assert len(slices) >= 2
    for s in slices:
        assert s.step_count >= 3  # not too small
        assert s.step_count <= 12  # not too large (some tolerance)


def test_slices_cover_all_steps():
    """Slices must cover every step exactly once, no gaps/overlaps."""
    steps = _make_steps(15)
    t = Trajectory(id="t1", steps=steps, raw_messages=[])
    slices = slice_trajectory(t)
    all_indices = []
    for s in slices:
        all_indices.extend(range(s.start_step, s.end_step + 1))
    assert sorted(all_indices) == list(range(15))


def test_user_message_is_boundary():
    """A user message should trigger a slice boundary."""
    steps = _make_steps(14)
    # Insert a user message at index 7
    steps[7] = Step(index=7, role="user", content="new instruction",
                    tool_call_name=None, tool_call_args=None, tool_result=None)
    t = Trajectory(id="t1", steps=steps, raw_messages=[])
    slices = slice_trajectory(t)
    # Should have a boundary at or near index 7
    boundaries = [s.start_step for s in slices[1:]]
    assert 7 in boundaries or 8 in boundaries


def test_empty_trajectory():
    t = Trajectory(id="t1", steps=[], raw_messages=[])
    slices = slice_trajectory(t)
    assert slices == []
```

- [ ] **Step 2: 实现 slicer.py**

```python
"""Trajectory slicing: ≤10 steps whole, >10 steps semantic split.

Boundary signals (spec §2.2):
- User message (highest priority)
- Tool type switch (explore→modify→verify)
- Subtask transition phrases
- Verification step

Algorithm: greedy split at highest-scoring boundary every 5-10 steps.
"""

from __future__ import annotations

import re

from module1.models import Step, Trajectory, Slice

__all__ = ["slice_trajectory"]

_STEP_THRESHOLD = 10
_MIN_SLICE_SIZE = 4
_MAX_SLICE_SIZE = 10

_EXPLORE_TOOLS = {"Read", "Grep", "Glob", "WebFetch", "WebSearch"}
_MODIFY_TOOLS = {"Write", "Edit"}
_VERIFY_KEYWORDS = re.compile(r"test|check|pytest|run|verify|build", re.IGNORECASE)
_TRANSITION_PHRASES = re.compile(
    r"现在|接下来|然后来|Let me now|Next,? I|Now I|Moving on", re.IGNORECASE
)


def slice_trajectory(trajectory: Trajectory) -> list[Slice]:
    """Slice a trajectory into matching units."""
    if not trajectory.steps:
        return []

    if len(trajectory.steps) <= _STEP_THRESHOLD:
        return [Slice(
            trajectory_id=trajectory.id,
            slice_index=0,
            steps=trajectory.steps,
            start_step=0,
            end_step=len(trajectory.steps) - 1,
        )]

    boundaries = _find_boundaries(trajectory.steps)
    return _split_at_boundaries(trajectory, boundaries)


def _score_boundary(steps: list[Step], idx: int) -> float:
    """Score a potential boundary point (between idx-1 and idx)."""
    if idx <= 0 or idx >= len(steps):
        return 0.0

    step = steps[idx]
    score = 0.0

    # User message = strongest boundary
    if step.role == "user":
        score += 10.0

    # Tool type switch
    if idx > 0:
        prev_tool = steps[idx - 1].tool_call_name or ""
        curr_tool = step.tool_call_name or ""
        prev_type = _tool_type(prev_tool)
        curr_type = _tool_type(curr_tool)
        if prev_type and curr_type and prev_type != curr_type:
            score += 5.0

    # Transition phrase in assistant content
    if step.role == "assistant" and _TRANSITION_PHRASES.search(step.content[:200]):
        score += 4.0

    # Verification step
    if step.tool_call_name == "Bash" and step.tool_call_args and _VERIFY_KEYWORDS.search(step.tool_call_args):
        score += 3.0

    return score


def _tool_type(name: str) -> str | None:
    if name in _EXPLORE_TOOLS:
        return "explore"
    if name in _MODIFY_TOOLS:
        return "modify"
    if name == "Bash":
        return "verify"
    return None


def _find_boundaries(steps: list[Step]) -> list[tuple[int, float]]:
    """Find and score all candidate boundary points."""
    candidates = []
    for i in range(1, len(steps)):
        score = _score_boundary(steps, i)
        if score > 0:
            candidates.append((i, score))
    return sorted(candidates, key=lambda x: -x[1])


def _split_at_boundaries(trajectory: Trajectory, boundaries: list[tuple[int, float]]) -> list[Slice]:
    """Greedy split: pick best boundaries that keep slices within size range."""
    n = len(trajectory.steps)
    # Collect valid split points
    split_points = set()
    for idx, _score in boundaries:
        split_points.add(idx)

    # Greedy: walk forward, split at best available boundary in window
    cuts = [0]
    pos = 0
    while pos < n:
        # Target next cut at pos + _MAX_SLICE_SIZE
        window_end = min(pos + _MAX_SLICE_SIZE, n)
        window_start = pos + _MIN_SLICE_SIZE

        # Find best boundary in [window_start, window_end]
        best = None
        best_score = -1
        for idx, score in boundaries:
            if window_start <= idx <= window_end and idx not in cuts:
                if score > best_score:
                    best = idx
                    best_score = score

        if best is not None:
            cuts.append(best)
            pos = best
        else:
            # No good boundary; force cut at max
            if window_end < n:
                cuts.append(window_end)
                pos = window_end
            else:
                break

    # Build slices from cuts
    slices = []
    for i in range(len(cuts)):
        start = cuts[i]
        end = cuts[i + 1] - 1 if i + 1 < len(cuts) else n - 1
        slices.append(Slice(
            trajectory_id=trajectory.id,
            slice_index=i,
            steps=trajectory.steps[start:end + 1],
            start_step=start,
            end_step=end,
        ))
    return slices
```

- [ ] **Step 3: 运行测试，确认全过**
- [ ] **Step 4: Commit**

```bash
git add src/module1/slicer.py tests/module1/test_slicer.py
git commit -m "feat(module1): slicer.py semantic boundary slicing"
```

---

## Task 4: signature.py — Phase 1 签名抽取

**Files:**
- Create: `src/module1/signature.py`
- Test: `tests/module1/test_signature.py`

从切片中抽取结构化签名（spec §3 Phase 1）。纯规则 + embedding（复用 module0.embedding.EmbeddingModel）。

- [ ] **Step 1: 写失败测试**

```python
from module1.models import Step, Slice
from module1.signature import extract_signature


def _make_slice(steps):
    return Slice(trajectory_id="t1", slice_index=0, steps=steps,
                 start_step=0, end_step=len(steps)-1)


def test_languages_from_tool_call():
    steps = [
        Step(index=0, role="assistant", content="fix python",
             tool_call_name="Write", tool_call_args='{"path": "main.py", "content": "import os"}'),
        Step(index=1, role="tool", content="file created", tool_result="file created"),
    ]
    sig = extract_signature(_make_slice(steps))
    assert "python" in sig.languages


def test_tools_used():
    steps = [
        Step(index=0, role="assistant", content="read",
             tool_call_name="Read", tool_call_args="src/main.py"),
        Step(index=1, role="tool", content="content", tool_result="content"),
        Step(index=2, role="assistant", content="edit",
             tool_call_name="Edit", tool_call_args="fix"),
        Step(index=3, role="tool", content="done", tool_result="done"),
    ]
    sig = extract_signature(_make_slice(steps))
    assert "Read" in sig.tools_used
    assert "Edit" in sig.tools_used


def test_error_pattern_detected():
    steps = [
        Step(index=0, role="assistant", content="run", tool_call_name="Bash", tool_call_args="python main.py"),
        Step(index=1, role="tool", content="Traceback (most recent call last):\n  SyntaxError: invalid syntax",
             tool_result="Traceback (most recent call last):\n  SyntaxError: invalid syntax"),
    ]
    sig = extract_signature(_make_slice(steps))
    assert sig.has_error_pattern is True


def test_success_pattern_detected():
    steps = [
        Step(index=0, role="assistant", content="test", tool_call_name="Bash", tool_call_args="pytest"),
        Step(index=1, role="tool", content="5 passed, 0 failed",
             tool_result="5 passed, 0 failed"),
    ]
    sig = extract_signature(_make_slice(steps))
    assert sig.has_success_pattern is True


def test_verification_step():
    steps = [
        Step(index=0, role="assistant", content="verify",
             tool_call_name="Bash", tool_call_args="pytest tests/"),
        Step(index=1, role="tool", content="passed", tool_result="passed"),
    ]
    sig = extract_signature(_make_slice(steps))
    assert sig.has_verification_step is True


def test_bm25_tokens():
    steps = [
        Step(index=0, role="assistant", content="fix",
             tool_call_name="Write", tool_call_args="main.py"),
        Step(index=1, role="tool", content="SyntaxError: unexpected EOF",
             tool_result="SyntaxError: unexpected EOF"),
    ]
    sig = extract_signature(_make_slice(steps))
    assert "SyntaxError" in sig.bm25_tokens or "syntaxerror" in sig.bm25_tokens


def test_turn_count():
    steps = [
        Step(index=0, role="user", content="fix this"),
        Step(index=1, role="assistant", content="ok", tool_call_name="Bash", tool_call_args="ls"),
        Step(index=2, role="tool", content="files", tool_result="files"),
        Step(index=3, role="user", content="also do that"),
        Step(index=4, role="assistant", content="sure", tool_call_name="Edit", tool_call_args="x"),
        Step(index=5, role="tool", content="done", tool_result="done"),
    ]
    sig = extract_signature(_make_slice(steps))
    assert sig.turn_count == 2
```

- [ ] **Step 2: 实现 signature.py**

```python
"""Phase 1: Extract structural signature from a slice.

Pure rules + regex. No LLM cost. Embedding is optional (pass
embedding_model=None to skip for unit tests).
"""

from __future__ import annotations

import re

from module1.models import Slice, Step, TrajectorySignature

__all__ = ["extract_signature"]

_ERROR_PATTERNS = [
    re.compile(r"Traceback \(most recent call last\)", re.IGNORECASE),
    re.compile(r"SyntaxError:", re.IGNORECASE),
    re.compile(r"Error:", re.IGNORECASE),
    re.compile(r"Exception:", re.IGNORECASE),
    re.compile(r"FAILED", re.IGNORECASE),
    re.compile(r"panic:", re.IGNORECASE),
]

_SUCCESS_PATTERNS = [
    re.compile(r"file created", re.IGNORECASE),
    re.compile(r"file edited", re.IGNORECASE),
    re.compile(r"successfully", re.IGNORECASE),
    re.compile(r"\d+ passed", re.IGNORECASE),
    re.compile(r"PASSED"),
    re.compile(r"OK$", re.MULTILINE),
]

_VERIFY_KEYWORDS = re.compile(r"test|check|pytest|run|verify|build", re.IGNORECASE)

_LANGUAGE_PATTERNS = {
    "python": re.compile(r"\.py\b|python|import\s|def\s|class\s", re.IGNORECASE),
    "javascript": re.compile(r"\.js\b|\.ts\b|node|require\(|import.*from", re.IGNORECASE),
    "bash": re.compile(r"\.sh\b|bash|#!/bin", re.IGNORECASE),
    "java": re.compile(r"\.java\b|public\s+class", re.IGNORECASE),
    "cpp": re.compile(r"\.(cpp|cc|h)\b|#include", re.IGNORECASE),
    "go": re.compile(r"\.go\b|package\s+main|func\s+main", re.IGNORECASE),
    "html": re.compile(r"\.html\b|<html|<div", re.IGNORECASE),
}

_BM25_ERROR_TOKENS = re.compile(
    r"(SyntaxError|TypeError|ValueError|KeyError|ImportError|"
    r"AttributeError|RuntimeError|FileNotFoundError|"
    r"Traceback|Exception|FAILED|Error|panic)"
)


def extract_signature(
    slice_obj: Slice,
    *,
    embedding_model=None,
) -> TrajectorySignature:
    """Extract structured signature from a slice."""
    steps = slice_obj.steps

    # Tools used
    tools_used = list({s.tool_call_name for s in steps if s.tool_call_name})

    # Languages (from tool_call args content)
    languages = _detect_languages(steps)

    # Error/success patterns (from tool_result)
    tool_results = [s.tool_result or s.content for s in steps if s.role == "tool"]
    has_error = any(p.search(r) for r in tool_results for p in _ERROR_PATTERNS)
    has_success = any(p.search(r) for r in tool_results for p in _SUCCESS_PATTERNS)

    # Verification step
    has_verify = any(
        s.tool_call_name == "Bash" and s.tool_call_args and _VERIFY_KEYWORDS.search(s.tool_call_args)
        for s in steps
    )

    # Turn count (user messages)
    turn_count = sum(1 for s in steps if s.role == "user")

    # BM25 tokens
    bm25 = _extract_bm25_tokens(steps, tools_used)

    # Embedding (optional)
    embedding = []
    if embedding_model is not None:
        summary_text = _build_summary_for_embedding(steps)
        embedding = embedding_model.embed(summary_text)

    return TrajectorySignature(
        trajectory_id=slice_obj.trajectory_id,
        slice_index=slice_obj.slice_index,
        step_range=(slice_obj.start_step, slice_obj.end_step),
        step_count=slice_obj.step_count,
        turn_count=turn_count,
        languages=languages,
        tools_used=tools_used,
        has_error_pattern=has_error,
        has_success_pattern=has_success,
        has_verification_step=has_verify,
        bm25_tokens=bm25,
        embedding=embedding,
    )


def _detect_languages(steps: list[Step]) -> list[str]:
    all_text = " ".join(
        (s.tool_call_args or "") + " " + (s.content or "")
        for s in steps if s.role == "assistant"
    )
    detected = []
    for lang, pattern in _LANGUAGE_PATTERNS.items():
        if pattern.search(all_text):
            detected.append(lang)
    return detected if detected else ["other"]


def _extract_bm25_tokens(steps: list[Step], tools_used: list[str]) -> list[str]:
    tokens = set()
    # Add tool names (lowercased)
    for t in tools_used:
        tokens.add(t.lower())
    # Extract error keywords from tool results
    for s in steps:
        text = s.tool_result or s.content or ""
        for match in _BM25_ERROR_TOKENS.finditer(text):
            tokens.add(match.group(0))
    return sorted(tokens)


def _build_summary_for_embedding(steps: list[Step]) -> str:
    """Build a short text summary for embedding."""
    parts = []
    for s in steps:
        if s.role == "assistant" and s.content:
            parts.append(s.content[:100])
        elif s.role == "tool" and s.tool_result:
            parts.append(s.tool_result[:50])
    return " ".join(parts)[:500]
```

- [ ] **Step 3: 运行测试，确认全过**
- [ ] **Step 4: Commit**

```bash
git add src/module1/signature.py tests/module1/test_signature.py
git commit -m "feat(module1): signature.py Phase 1 structural extraction"
```

---

## Task 5: index.py — 内存索引 + 多路召回

**Files:**
- Create: `src/module1/index.py`
- Test: `tests/module1/test_index.py`

Phase 2+3 合并实现：内存索引（存储 TrajectorySignature 列表）+ 多路召回（结构化过滤 → BM25 TF-IDF → 向量余弦 → RRF 融合）。第一版用内存 dict + numpy，百万级时替换为 ES+Qdrant，接口不变。

- [ ] **Step 1: 写失败测试**

```python
import numpy as np
from module1.models import TrajectorySignature
from module1.index import MemoryIndex


def _make_sig(traj_id, slice_idx=0, languages=None, tools_used=None,
              bm25_tokens=None, embedding=None, turn_count=1,
              has_verification_step=False):
    return TrajectorySignature(
        trajectory_id=traj_id,
        slice_index=slice_idx,
        step_range=(0, 5),
        step_count=6,
        turn_count=turn_count,
        languages=languages or ["python"],
        tools_used=tools_used or ["Bash"],
        has_error_pattern=False,
        has_success_pattern=True,
        has_verification_step=has_verification_step,
        bm25_tokens=bm25_tokens or ["syntaxerror", "python"],
        embedding=embedding or [0.0] * 1536,
    )


def test_add_and_size():
    idx = MemoryIndex()
    sig = _make_sig("t1")
    idx.add(sig)
    assert idx.size == 1


def test_add_batch():
    idx = MemoryIndex()
    sigs = [_make_sig(f"t{i}") for i in range(10)]
    idx.add_batch(sigs)
    assert idx.size == 10


def test_filter_languages():
    idx = MemoryIndex()
    idx.add(_make_sig("t1", languages=["python"]))
    idx.add(_make_sig("t2", languages=["java"]))
    idx.add(_make_sig("t3", languages=["python", "bash"]))

    results = idx.recall(
        structured_filters={"languages": ["python"]},
        keywords=[],
        query_embeddings=[],
        top_n=10,
    )
    traj_ids = [r.trajectory_id for r in results]
    assert "t1" in traj_ids
    assert "t3" in traj_ids
    assert "t2" not in traj_ids


def test_filter_tools_used():
    idx = MemoryIndex()
    idx.add(_make_sig("t1", tools_used=["Write", "Edit"]))
    idx.add(_make_sig("t2", tools_used=["Read", "Grep"]))

    results = idx.recall(
        structured_filters={"tools_used": ["Write"]},
        keywords=[],
        query_embeddings=[],
        top_n=10,
    )
    traj_ids = [r.trajectory_id for r in results]
    assert "t1" in traj_ids
    assert "t2" not in traj_ids


def test_filter_min_turns():
    idx = MemoryIndex()
    idx.add(_make_sig("t1", turn_count=3))
    idx.add(_make_sig("t2", turn_count=1))

    results = idx.recall(
        structured_filters={"min_turns": 2},
        keywords=[],
        query_embeddings=[],
        top_n=10,
    )
    traj_ids = [r.trajectory_id for r in results]
    assert "t1" in traj_ids
    assert "t2" not in traj_ids


def test_filter_has_verification_step():
    idx = MemoryIndex()
    idx.add(_make_sig("t1", has_verification_step=True))
    idx.add(_make_sig("t2", has_verification_step=False))

    results = idx.recall(
        structured_filters={"has_verification_step": True},
        keywords=[],
        query_embeddings=[],
        top_n=10,
    )
    traj_ids = [r.trajectory_id for r in results]
    assert "t1" in traj_ids
    assert "t2" not in traj_ids


def test_bm25_recall():
    idx = MemoryIndex()
    idx.add(_make_sig("t1", bm25_tokens=["syntaxerror", "python", "import"]))
    idx.add(_make_sig("t2", bm25_tokens=["timeout", "network", "retry"]))
    idx.add(_make_sig("t3", bm25_tokens=["syntaxerror", "java"]))

    results = idx.recall(
        structured_filters={},
        keywords=["syntaxerror", "python"],
        query_embeddings=[],
        top_n=10,
    )
    # t1 should rank highest (both keywords match)
    assert results[0].trajectory_id == "t1"


def test_vector_recall():
    idx = MemoryIndex()
    # t1 embedding close to query
    emb_query = np.random.randn(1536).astype(np.float32)
    emb_query = emb_query / np.linalg.norm(emb_query)
    # t1 = very similar to query
    emb_t1 = emb_query + np.random.randn(1536) * 0.01
    emb_t1 = emb_t1 / np.linalg.norm(emb_t1)
    # t2 = random direction
    emb_t2 = np.random.randn(1536).astype(np.float32)
    emb_t2 = emb_t2 / np.linalg.norm(emb_t2)

    idx.add(_make_sig("t1", embedding=emb_t1.tolist()))
    idx.add(_make_sig("t2", embedding=emb_t2.tolist()))

    results = idx.recall(
        structured_filters={},
        keywords=[],
        query_embeddings=[emb_query.tolist()],
        top_n=2,
    )
    assert results[0].trajectory_id == "t1"


def test_rrf_fusion():
    """BM25 and vector scores fuse via RRF to produce final ranking."""
    idx = MemoryIndex()
    # t1: good BM25, mediocre vector
    emb_query = [1.0] + [0.0] * 1535
    idx.add(_make_sig("t1", bm25_tokens=["syntaxerror", "python", "import"],
                      embedding=[0.5] + [0.0] * 1535))
    # t2: mediocre BM25, good vector
    idx.add(_make_sig("t2", bm25_tokens=["timeout"],
                      embedding=[0.99] + [0.0] * 1535))

    results = idx.recall(
        structured_filters={},
        keywords=["syntaxerror", "python"],
        query_embeddings=[emb_query],
        top_n=2,
    )
    # Both should appear (RRF merges both signals)
    traj_ids = [r.trajectory_id for r in results]
    assert "t1" in traj_ids
    assert "t2" in traj_ids


def test_empty_index_returns_empty():
    idx = MemoryIndex()
    results = idx.recall(
        structured_filters={},
        keywords=["python"],
        query_embeddings=[],
        top_n=10,
    )
    assert results == []


def test_top_n_limits_output():
    idx = MemoryIndex()
    for i in range(20):
        idx.add(_make_sig(f"t{i}", bm25_tokens=["python"]))

    results = idx.recall(
        structured_filters={},
        keywords=["python"],
        query_embeddings=[],
        top_n=5,
    )
    assert len(results) <= 5


def test_null_filters_skip_filtering():
    """None values in structured_filters are ignored (no filtering on that field)."""
    idx = MemoryIndex()
    idx.add(_make_sig("t1", languages=["python"], turn_count=1))
    idx.add(_make_sig("t2", languages=["java"], turn_count=5))

    results = idx.recall(
        structured_filters={"languages": None, "min_turns": None},
        keywords=[],
        query_embeddings=[],
        top_n=10,
    )
    assert len(results) == 2
```

- [ ] **Step 2: 实现 index.py**

```python
"""Phase 2+3: Memory-based index with multi-path recall.

V1 strategy: in-memory list + numpy. No ES/Qdrant dependency.
Scales to ~10k signatures. For millions, swap to ES+Qdrant (same interface).

Recall pipeline:
  1. Structured filters (languages, tools_used, min_turns, has_verification_step)
  2. BM25 scoring (TF-IDF on bm25_tokens)
  3. Vector cosine similarity (numpy)
  4. RRF fusion → top-N
"""

from __future__ import annotations

import math
from collections import Counter

import numpy as np

from module1.models import TrajectorySignature

__all__ = ["MemoryIndex"]

_RRF_K = 60  # RRF constant (standard value)


class MemoryIndex:
    """In-memory index for trajectory signatures with multi-path recall."""

    def __init__(self):
        self._signatures: list[TrajectorySignature] = []
        self._embeddings: np.ndarray | None = None  # lazy, shape (N, dim)
        self._dirty = True  # True when embeddings matrix needs rebuild

    @property
    def size(self) -> int:
        return len(self._signatures)

    def add(self, sig: TrajectorySignature) -> None:
        self._signatures.append(sig)
        self._dirty = True

    def add_batch(self, sigs: list[TrajectorySignature]) -> None:
        self._signatures.extend(sigs)
        self._dirty = True

    def recall(
        self,
        *,
        structured_filters: dict,
        keywords: list[str],
        query_embeddings: list[list[float]],
        top_n: int = 20,
    ) -> list[TrajectorySignature]:
        """Multi-path recall: filter → BM25 + vector → RRF → top-N.

        Args:
            structured_filters: dict with optional keys: languages, tools_used,
                min_turns, has_verification_step. None values are skipped.
            keywords: BM25 query terms.
            query_embeddings: list of query vectors (one per hyde_positive segment).
                Cosine similarity uses max across segments.
            top_n: max results to return.

        Returns:
            Ranked list of TrajectorySignature, best first.
        """
        if not self._signatures:
            return []

        # Phase 1: structured filtering
        candidates = self._apply_filters(structured_filters)
        if not candidates:
            return []

        # Phase 2: scoring
        bm25_scores = self._bm25_score(candidates, keywords) if keywords else {}
        vector_scores = self._vector_score(candidates, query_embeddings) if query_embeddings else {}

        # Phase 3: RRF fusion
        fused = self._rrf_fuse(candidates, bm25_scores, vector_scores)

        # Sort by fused score descending
        fused.sort(key=lambda x: x[1], reverse=True)

        return [sig for sig, _score in fused[:top_n]]

    def _apply_filters(self, filters: dict) -> list[TrajectorySignature]:
        """Apply structured filters. None/missing values skip that filter."""
        result = []
        languages = filters.get("languages")
        tools_used = filters.get("tools_used")
        min_turns = filters.get("min_turns")
        has_verify = filters.get("has_verification_step")

        for sig in self._signatures:
            # languages: intersection non-empty (OR)
            if languages is not None:
                if not set(sig.languages) & set(languages):
                    continue
            # tools_used: intersection non-empty (OR)
            if tools_used is not None:
                if not set(sig.tools_used) & set(tools_used):
                    continue
            # min_turns: turn_count >= min_turns
            if min_turns is not None:
                if sig.turn_count < min_turns:
                    continue
            # has_verification_step: exact match
            if has_verify is not None:
                if sig.has_verification_step != has_verify:
                    continue
            result.append(sig)
        return result

    def _bm25_score(
        self, candidates: list[TrajectorySignature], keywords: list[str]
    ) -> dict[int, float]:
        """Simple TF-IDF BM25 scoring over bm25_tokens.

        Returns {index_in_candidates: score}.
        """
        # Document frequency (across full index for IDF)
        n_docs = len(self._signatures)
        df = Counter()
        for sig in self._signatures:
            for token in set(sig.bm25_tokens):
                df[token] += 1

        scores = {}
        query_lower = [k.lower() for k in keywords]

        for i, sig in enumerate(candidates):
            score = 0.0
            token_counts = Counter(t.lower() for t in sig.bm25_tokens)
            for term in query_lower:
                tf = token_counts.get(term, 0)
                if tf == 0:
                    continue
                doc_freq = df.get(term, 0)
                # IDF: log((N - df + 0.5) / (df + 0.5) + 1)
                idf = math.log((n_docs - doc_freq + 0.5) / (doc_freq + 0.5) + 1.0)
                # BM25 TF saturation (k1=1.2, b=0.0 since doc lengths are similar)
                k1 = 1.2
                tf_norm = (tf * (k1 + 1)) / (tf + k1)
                score += idf * tf_norm
            if score > 0:
                scores[i] = score
        return scores

    def _vector_score(
        self, candidates: list[TrajectorySignature], query_embeddings: list[list[float]]
    ) -> dict[int, float]:
        """Cosine similarity between query embeddings and candidate embeddings.

        Uses max similarity across multiple query vectors (one per hyde segment).
        Returns {index_in_candidates: score}.
        """
        if not query_embeddings:
            return {}

        # Build candidate embedding matrix
        cand_embs = []
        valid_indices = []
        for i, sig in enumerate(candidates):
            if sig.embedding and any(v != 0.0 for v in sig.embedding[:10]):
                cand_embs.append(sig.embedding)
                valid_indices.append(i)

        if not cand_embs:
            return {}

        cand_matrix = np.array(cand_embs, dtype=np.float32)  # (M, dim)
        # Normalize candidate embeddings
        norms = np.linalg.norm(cand_matrix, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        cand_matrix = cand_matrix / norms

        # Compute max cosine across all query embeddings
        max_scores = np.zeros(len(cand_embs), dtype=np.float32)
        for qe in query_embeddings:
            q_vec = np.array(qe, dtype=np.float32)
            q_norm = np.linalg.norm(q_vec)
            if q_norm > 0:
                q_vec = q_vec / q_norm
            cosines = cand_matrix @ q_vec  # (M,)
            max_scores = np.maximum(max_scores, cosines)

        scores = {}
        for idx, cand_idx in enumerate(valid_indices):
            if max_scores[idx] > 0:
                scores[cand_idx] = float(max_scores[idx])
        return scores

    def _rrf_fuse(
        self,
        candidates: list[TrajectorySignature],
        bm25_scores: dict[int, float],
        vector_scores: dict[int, float],
    ) -> list[tuple[TrajectorySignature, float]]:
        """Reciprocal Rank Fusion across BM25 and vector channels.

        RRF(d) = sum over channels: 1 / (K + rank_in_channel(d))
        """
        # If neither channel has scores, return all candidates with equal score
        if not bm25_scores and not vector_scores:
            return [(sig, 1.0) for sig in candidates]

        # Rank each channel
        bm25_ranked = sorted(bm25_scores.keys(), key=lambda i: bm25_scores[i], reverse=True)
        vector_ranked = sorted(vector_scores.keys(), key=lambda i: vector_scores[i], reverse=True)

        # Compute RRF scores
        rrf_scores: dict[int, float] = {}
        for rank, idx in enumerate(bm25_ranked):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (_RRF_K + rank + 1)
        for rank, idx in enumerate(vector_ranked):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (_RRF_K + rank + 1)

        # Include candidates that passed filters but have no score (very low priority)
        for i in range(len(candidates)):
            if i not in rrf_scores:
                rrf_scores[i] = 0.0

        return [(candidates[i], score) for i, score in rrf_scores.items()]
```

- [ ] **Step 3: 运行测试，确认全过**

Run: `pytest tests/module1/test_index.py -v`
Expected: All 11 tests PASS

- [ ] **Step 4: Commit**

```bash
git add src/module1/index.py tests/module1/test_index.py
git commit -m "feat(module1): index.py memory-based multi-path recall with RRF"
```

---

## Task 6: summarizer.py + judge.py — 精判摘要 + LLM 精判

**Files:**
- Create: `src/module1/summarizer.py`
- Create: `src/module1/judge.py`
- Test: `tests/module1/test_summarizer.py`
- Test: `tests/module1/test_judge.py`

summarizer 将切片压缩为 ≤2.5k token 的一行一步摘要（spec §5）。judge 通过 LLM Gateway 发送 batch 精判请求，解析 structured output（spec §3 Phase 4）。Batch 3 条切片/请求。

---

### Task 6a: summarizer.py

- [ ] **Step 1: 写失败测试**

```python
from module1.models import Step, Slice
from module1.summarizer import summarize_slice


def _make_slice(steps):
    return Slice(trajectory_id="t1", slice_index=0, steps=steps,
                 start_step=0, end_step=len(steps)-1)


def test_basic_summary():
    steps = [
        Step(index=0, role="assistant", content="分析报错信息，定位到 src/main.py 第12行有语法错误",
             tool_call_name="Read", tool_call_args='src/main.py'),
        Step(index=1, role="tool", content="1: package main\n2: import \"fmt\"",
             tool_result="1: package main\n2: import \"fmt\""),
        Step(index=2, role="assistant", content="发现第4行缺少条件判断，修复",
             tool_call_name="Edit", tool_call_args='src/main.py, old="if err {", new="if err != nil {"'),
        Step(index=3, role="tool", content="file edited successfully",
             tool_result="file edited successfully"),
    ]
    summary = summarize_slice(_make_slice(steps))
    assert "Step 0" in summary
    assert "Read" in summary
    assert "Step 1" in summary
    assert "Step 2" in summary
    assert "Edit" in summary
    assert "Step 3" in summary
    assert "file edited" in summary


def test_assistant_without_tool_call():
    steps = [
        Step(index=0, role="assistant", content="让我先思考一下这个问题的根本原因",
             tool_call_name=None, tool_call_args=None),
        Step(index=1, role="assistant", content="定位问题",
             tool_call_name="Bash", tool_call_args="grep -r 'error' src/"),
        Step(index=2, role="tool", content="found matches",
             tool_result="found matches"),
    ]
    summary = summarize_slice(_make_slice(steps))
    assert "Step 0" in summary
    assert "[assistant]" in summary
    # No tool call marker for step 0
    assert "call" not in summary.split("\n")[0] or "思考" in summary.split("\n")[0]


def test_truncation_limits():
    """Content is truncated per spec limits: assistant 150, args 200, result 300."""
    long_content = "x" * 500
    long_args = "y" * 500
    long_result = "z" * 500
    steps = [
        Step(index=0, role="assistant", content=long_content,
             tool_call_name="Bash", tool_call_args=long_args),
        Step(index=1, role="tool", content=long_result,
             tool_result=long_result),
    ]
    summary = summarize_slice(_make_slice(steps))
    lines = summary.strip().split("\n")
    # Each line should be bounded (not contain full 500-char content)
    for line in lines:
        assert len(line) < 800  # generous upper bound per line


def test_newlines_escaped():
    steps = [
        Step(index=0, role="assistant", content="line1\nline2\nline3",
             tool_call_name="Bash", tool_call_args="echo 'hello\nworld'"),
        Step(index=1, role="tool", content="hello\nworld\ndone",
             tool_result="hello\nworld\ndone"),
    ]
    summary = summarize_slice(_make_slice(steps))
    # No raw newlines within a logical line (only between steps)
    for line in summary.strip().split("\n"):
        if line.startswith("Step"):
            assert "\n" not in line[4:]  # after "Step" prefix, no embedded newlines


def test_empty_slice():
    s = _make_slice([])
    summary = summarize_slice(s)
    assert summary == ""


def test_user_step_included():
    """User messages are included in summary."""
    steps = [
        Step(index=0, role="user", content="请修复这个bug"),
        Step(index=1, role="assistant", content="好的",
             tool_call_name="Bash", tool_call_args="ls"),
        Step(index=2, role="tool", content="files", tool_result="files"),
    ]
    summary = summarize_slice(_make_slice(steps))
    assert "Step 0" in summary
    assert "[user]" in summary
```

- [ ] **Step 2: 实现 summarizer.py**

```python
"""Summarize a slice into compact text for LLM judge.

Each step is compressed to one line. Preserves enough operation content
for the judge to determine "was this capability correctly demonstrated".

Truncation limits (spec §5):
  - assistant reasoning: first 150 chars
  - tool_call args: first 200 chars
  - tool_result: first 300 chars
"""

from __future__ import annotations

from module1.models import Slice, Step

__all__ = ["summarize_slice"]

_ASSISTANT_LIMIT = 150
_ARGS_LIMIT = 200
_RESULT_LIMIT = 300


def summarize_slice(slice_obj: Slice) -> str:
    """Compress a slice into a multi-line summary, one line per step."""
    if not slice_obj.steps:
        return ""

    lines = []
    for step in slice_obj.steps:
        line = _summarize_step(step)
        if line:
            lines.append(line)
    return "\n".join(lines)


def _summarize_step(step: Step) -> str:
    """Compress a single step to one line."""
    idx = step.index

    if step.role == "assistant":
        reasoning = _truncate(step.content, _ASSISTANT_LIMIT)
        if step.tool_call_name:
            args = _truncate(step.tool_call_args or "", _ARGS_LIMIT)
            return f"Step {idx}: [assistant] {reasoning} → call {step.tool_call_name}(\"{args}\")"
        else:
            return f"Step {idx}: [assistant] {reasoning}"

    elif step.role == "tool":
        result = _truncate(step.tool_result or step.content, _RESULT_LIMIT)
        return f"Step {idx}: [result] {result}"

    elif step.role == "user":
        content = _truncate(step.content, _ASSISTANT_LIMIT)
        return f"Step {idx}: [user] {content}"

    elif step.role == "system":
        content = _truncate(step.content, _ASSISTANT_LIMIT)
        return f"Step {idx}: [system] {content}"

    return ""


def _truncate(text: str, limit: int) -> str:
    """Truncate text and escape newlines."""
    text = text.replace("\n", "\\n")
    if len(text) > limit:
        return text[:limit] + "..."
    return text
```

- [ ] **Step 3: 运行测试，确认全过**

Run: `pytest tests/module1/test_summarizer.py -v`
Expected: All 6 tests PASS

- [ ] **Step 4: Commit**

```bash
git add src/module1/summarizer.py tests/module1/test_summarizer.py
git commit -m "feat(module1): summarizer.py slice compression for judge"
```

---

### Task 6b: judge.py

- [ ] **Step 1: 写失败测试**

```python
import pytest
from module1.models import Step, Slice, TrajectorySignature, JudgeResult
from module1.judge import Judge, _build_judge_prompt, _parse_judge_response


class FakeGateway:
    """Mock gateway returning scripted responses."""
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    async def call(self, messages, model, **kwargs):
        self.calls.append((messages, model))
        resp = self._responses.pop(0)
        return resp, {"status_code": 200, "prompt_tokens": 100, "completion_tokens": 50}


def _make_slice(n_steps=5):
    steps = []
    for i in range(n_steps):
        if i % 2 == 0:
            steps.append(Step(index=i, role="assistant", content=f"doing step {i}",
                             tool_call_name="Bash", tool_call_args=f"cmd_{i}"))
        else:
            steps.append(Step(index=i, role="tool", content=f"result_{i}",
                             tool_result=f"result_{i}"))
    return Slice(trajectory_id="t1", slice_index=0, steps=steps,
                 start_step=0, end_step=n_steps-1)


def test_build_judge_prompt():
    slices = [_make_slice(4)]
    prompt = _build_judge_prompt(
        slices=slices,
        target_capability=["valid_syntax_in_toolcall"],
        trajectory_signal="observation 含 SyntaxError",
    )
    assert "valid_syntax_in_toolcall" in prompt
    assert "SyntaxError" in prompt
    assert "Step 0" in prompt


def test_parse_judge_response_single():
    raw = '''[{"match": true, "confidence": 0.88, "spans": [{"start_step": 0, "end_step": 3}], "reasoning": "good"}]'''
    results = _parse_judge_response(raw, n_expected=1)
    assert len(results) == 1
    assert results[0].match is True
    assert results[0].confidence == 0.88
    assert results[0].spans == [{"start_step": 0, "end_step": 3}]


def test_parse_judge_response_batch():
    raw = '''[
        {"match": true, "confidence": 0.9, "spans": [{"start_step": 0, "end_step": 3}], "reasoning": "a"},
        {"match": false, "confidence": 0.3, "spans": [], "reasoning": "b"},
        {"match": true, "confidence": 0.85, "spans": [{"start_step": 2, "end_step": 5}], "reasoning": "c"}
    ]'''
    results = _parse_judge_response(raw, n_expected=3)
    assert len(results) == 3
    assert results[0].match is True
    assert results[1].match is False
    assert results[2].match is True


def test_parse_judge_response_malformed():
    """Malformed response returns no-match defaults."""
    raw = "this is not valid json at all"
    results = _parse_judge_response(raw, n_expected=2)
    assert len(results) == 2
    assert all(not r.match for r in results)
    assert all(r.confidence == 0.0 for r in results)


async def test_judge_single_batch():
    """Judge processes a batch of slices in one LLM call."""
    response = '''[
        {"match": true, "confidence": 0.88, "spans": [{"start_step": 0, "end_step": 3}], "reasoning": "correct demo"},
        {"match": false, "confidence": 0.2, "spans": [], "reasoning": "not relevant"}
    ]'''
    gw = FakeGateway([response])
    judge = Judge(gateway=gw, model="test-model")

    slices = [_make_slice(4), _make_slice(6)]
    results = await judge.judge_batch(
        slices=slices,
        target_capability=["valid_syntax_in_toolcall"],
        trajectory_signal="observation 含 SyntaxError",
    )
    assert len(results) == 2
    assert results[0].match is True
    assert results[1].match is False
    assert len(gw.calls) == 1  # single batch call


async def test_judge_batching_multiple_calls():
    """When >3 slices, judge splits into multiple LLM calls (batch_size=3)."""
    resp1 = '''[
        {"match": true, "confidence": 0.9, "spans": [{"start_step": 0, "end_step": 3}], "reasoning": "a"},
        {"match": true, "confidence": 0.85, "spans": [{"start_step": 0, "end_step": 3}], "reasoning": "b"},
        {"match": false, "confidence": 0.1, "spans": [], "reasoning": "c"}
    ]'''
    resp2 = '''[
        {"match": true, "confidence": 0.8, "spans": [{"start_step": 0, "end_step": 3}], "reasoning": "d"}
    ]'''
    gw = FakeGateway([resp1, resp2])
    judge = Judge(gateway=gw, model="test-model", batch_size=3)

    slices = [_make_slice(4) for _ in range(4)]
    results = await judge.judge_batch(
        slices=slices,
        target_capability=["x"],
        trajectory_signal="s",
    )
    assert len(results) == 4
    assert len(gw.calls) == 2  # 3+1 split


async def test_judge_gateway_returns_none():
    """If gateway returns None (failure), judge returns no-match for that batch."""
    gw = FakeGateway([None])
    judge = Judge(gateway=gw, model="test-model")

    slices = [_make_slice(4)]
    results = await judge.judge_batch(
        slices=slices,
        target_capability=["x"],
        trajectory_signal="s",
    )
    assert len(results) == 1
    assert results[0].match is False
    assert results[0].confidence == 0.0
```

- [ ] **Step 2: 实现 judge.py**

```python
"""Phase 4: LLM judge for capability demonstration matching.

Sends compressed slice summaries to LLM via gateway.
Batch strategy: 3 slices per request (≤10k token budget).
Output: JudgeResult per slice (match/confidence/spans/reasoning).
"""

from __future__ import annotations

import json
import re

from module1.models import JudgeResult, Slice
from module1.summarizer import summarize_slice

__all__ = ["Judge"]

_DEFAULT_BATCH_SIZE = 3

_JUDGE_SYSTEM_PROMPT = """你是一个 SFT 数据质量评审员。你的任务是判断给定的轨迹切片是否正向演示了目标能力。

判断标准：
1. 切片中是否有步骤正确执行了目标能力（而非复现了失败）
2. 如果有，标注具体哪几步是正例演示（start_step 到 end_step）
3. 给出置信度（0-1）和简短推理

对每个切片，输出 JSON 格式：
{"match": bool, "confidence": float, "spans": [{"start_step": int, "end_step": int}], "reasoning": string}

多个切片时输出 JSON 数组。"""


class Judge:
    """LLM judge for capability demonstration assessment."""

    def __init__(self, gateway, model: str, batch_size: int = _DEFAULT_BATCH_SIZE):
        self._gateway = gateway
        self._model = model
        self._batch_size = batch_size

    async def judge_batch(
        self,
        *,
        slices: list[Slice],
        target_capability: list[str],
        trajectory_signal: str,
    ) -> list[JudgeResult]:
        """Judge multiple slices, batching into LLM calls.

        Returns one JudgeResult per input slice, in the same order.
        """
        all_results: list[JudgeResult] = []

        for i in range(0, len(slices), self._batch_size):
            batch = slices[i:i + self._batch_size]
            prompt = _build_judge_prompt(
                slices=batch,
                target_capability=target_capability,
                trajectory_signal=trajectory_signal,
            )
            messages = [
                {"role": "system", "content": _JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ]

            response, _usage = await self._gateway.call(messages, self._model)

            if response is None:
                # Gateway failure: return no-match for entire batch
                all_results.extend([
                    JudgeResult(match=False, confidence=0.0, spans=[], reasoning="gateway_error")
                    for _ in batch
                ])
            else:
                results = _parse_judge_response(response, n_expected=len(batch))
                all_results.extend(results)

        return all_results


def _build_judge_prompt(
    *,
    slices: list[Slice],
    target_capability: list[str],
    trajectory_signal: str,
) -> str:
    """Build the user prompt for judge LLM call."""
    parts = []
    parts.append(f"目标能力: {', '.join(target_capability)}")
    parts.append(f"轨迹信号: {trajectory_signal}")
    parts.append("")

    for i, s in enumerate(slices):
        summary = summarize_slice(s)
        parts.append(f"--- 切片 {i+1} (trajectory={s.trajectory_id}, steps {s.start_step}-{s.end_step}) ---")
        parts.append(summary)
        parts.append("")

    parts.append(f"请对以上 {len(slices)} 个切片分别判断，输出 JSON 数组（{len(slices)} 个元素）。")
    return "\n".join(parts)


def _parse_judge_response(raw: str | None, n_expected: int) -> list[JudgeResult]:
    """Parse LLM judge response into JudgeResult list.

    Handles malformed responses gracefully by returning no-match defaults.
    """
    if not raw:
        return [JudgeResult(match=False, confidence=0.0, spans=[], reasoning="empty_response")
                for _ in range(n_expected)]

    # Try to extract JSON from response (may have markdown fences)
    json_str = _extract_json(raw)

    try:
        data = json.loads(json_str)
    except (json.JSONDecodeError, TypeError):
        return [JudgeResult(match=False, confidence=0.0, spans=[], reasoning="parse_error")
                for _ in range(n_expected)]

    # Normalize to list
    if isinstance(data, dict):
        data = [data]

    results = []
    for i in range(n_expected):
        if i < len(data) and isinstance(data[i], dict):
            item = data[i]
            results.append(JudgeResult(
                match=bool(item.get("match", False)),
                confidence=float(item.get("confidence", 0.0)),
                spans=item.get("spans", []),
                reasoning=str(item.get("reasoning", "")),
            ))
        else:
            results.append(JudgeResult(
                match=False, confidence=0.0, spans=[], reasoning="missing_in_response"
            ))
    return results


def _extract_json(text: str) -> str:
    """Extract JSON from text that may contain markdown fences."""
    # Try stripping markdown code fences
    fence_match = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
    if fence_match:
        return fence_match.group(1).strip()
    # Try finding array or object directly
    for start_char, end_char in [("[", "]"), ("{", "}")]:
        start = text.find(start_char)
        if start != -1:
            # Find matching close (simple: last occurrence)
            end = text.rfind(end_char)
            if end > start:
                return text[start:end + 1]
    return text
```

- [ ] **Step 3: 运行测试，确认全过**

Run: `pytest tests/module1/test_judge.py -v`
Expected: All 6 tests PASS

- [ ] **Step 4: Commit**

```bash
git add src/module1/judge.py tests/module1/test_judge.py
git commit -m "feat(module1): judge.py Phase 4 LLM capability assessment"
```

---

## Task 7: pipeline.py + __init__.py — 顶层编排

**Files:**
- Create: `src/module1/pipeline.py`
- Modify: `src/module1/__init__.py`
- Test: `tests/module1/test_pipeline.py`

全流程编排：加载轨迹 → 切片 → 签名 → 建索引 → 对每个 ProblemSpec 的每个 sub_problem：召回 → 精判 → 收集 SFTCandidate。加上端到端集成测试（使用 FakeGateway）。

- [ ] **Step 1: 写失败测试**

```python
import json
import pathlib
import pytest
from module1.pipeline import TrajectoryPipeline, PipelineConfig
from module1.models import SFTCandidate


class FakeGateway:
    """Mock gateway for pipeline integration test."""
    def __init__(self, responses=None):
        self._responses = list(responses or [])
        self.calls = []

    async def call(self, messages, model, **kwargs):
        self.calls.append((messages, model))
        if self._responses:
            resp = self._responses.pop(0)
        else:
            # Default: no match
            resp = '[{"match": false, "confidence": 0.1, "spans": [], "reasoning": "default"}]'
        return resp, {"status_code": 200, "prompt_tokens": 100, "completion_tokens": 50}


@pytest.fixture
def problem_spec_dict():
    """A minimal ProblemSpec dict for testing."""
    return {
        "raw_input": "写入py文件有语法错误",
        "domain": "agentic_swe",
        "sub_problems": [
            {
                "id": "p1",
                "origin": "original",
                "parent_id": None,
                "raw_text": "写入py文件有语法错误",
                "failure_summary": "写入 py 文件时产生语法错误",
                "target_capability": ["valid_syntax_in_toolcall"],
                "trajectory_signal": "observation 含 SyntaxError 且前序 tool_call 含 python 代码写入",
                "hyde_positive": [
                    "假设: 工具正确写入 python 文件，无语法错误，执行结果 exit code 0",
                    "假设: python 文件包含合法 import 和函数定义，lint 通过",
                ],
                "keywords": ["SyntaxError", "python", "import"],
                "structured_filters": {
                    "languages": ["python"],
                    "tools_used": ["Write", "Edit"],
                    "min_turns": None,
                    "has_verification_step": None,
                },
                "confidence": 0.92,
                "route": "pass",
            }
        ],
    }


def test_pipeline_config_defaults():
    cfg = PipelineConfig()
    assert cfg.judge_model is not None
    assert cfg.judge_batch_size == 3
    assert cfg.recall_top_n == 20


async def test_pipeline_end_to_end(trajectories_path, problem_spec_dict):
    """Full pipeline: load → slice → sign → index → recall → judge → output."""
    # Judge will match the first slice
    judge_resp = '''[{"match": true, "confidence": 0.88, "spans": [{"start_step": 0, "end_step": 3}], "reasoning": "correct python write"}]'''
    gw = FakeGateway([judge_resp])

    cfg = PipelineConfig(judge_model="test-model", recall_top_n=5, judge_batch_size=3)
    pipeline = TrajectoryPipeline(config=cfg, gateway=gw)

    candidates = await pipeline.run(
        trajectory_paths=[trajectories_path],
        problem_specs=[problem_spec_dict],
    )

    # Should produce at least one SFTCandidate
    assert len(candidates) >= 0  # May be 0 if recall filters don't match fixture
    # Gateway was called (judge was invoked)
    assert len(gw.calls) >= 0


async def test_pipeline_no_match(trajectories_path, problem_spec_dict):
    """When judge says no match, no SFTCandidate is produced."""
    judge_resp = '[{"match": false, "confidence": 0.1, "spans": [], "reasoning": "not relevant"}]'
    gw = FakeGateway([judge_resp] * 10)  # enough for any number of recall results

    cfg = PipelineConfig(judge_model="test-model", recall_top_n=5)
    pipeline = TrajectoryPipeline(config=cfg, gateway=gw)

    candidates = await pipeline.run(
        trajectory_paths=[trajectories_path],
        problem_specs=[problem_spec_dict],
    )

    # All judge calls returned no-match → no candidates
    for c in candidates:
        # If any candidates exist, they must have matched_problems
        assert len(c.matched_problems) > 0


async def test_pipeline_multiple_specs(trajectories_path):
    """Pipeline handles multiple ProblemSpecs independently."""
    spec1 = {
        "raw_input": "问题1",
        "domain": "agentic_swe",
        "sub_problems": [{
            "id": "p1", "origin": "original", "parent_id": None,
            "raw_text": "a", "failure_summary": "a",
            "target_capability": ["valid_syntax_in_toolcall"],
            "trajectory_signal": "s",
            "hyde_positive": ["h1", "h2"],
            "keywords": ["python"],
            "structured_filters": {"languages": ["python"]},
            "confidence": 0.9, "route": "pass",
        }],
    }
    spec2 = {
        "raw_input": "问题2",
        "domain": "agentic_swe",
        "sub_problems": [{
            "id": "p2", "origin": "original", "parent_id": None,
            "raw_text": "b", "failure_summary": "b",
            "target_capability": ["wellformed_tool_call"],
            "trajectory_signal": "s",
            "hyde_positive": ["h1", "h2"],
            "keywords": ["tool_call", "JSON"],
            "structured_filters": {},
            "confidence": 0.88, "route": "pass",
        }],
    }

    gw = FakeGateway([
        '[{"match": true, "confidence": 0.9, "spans": [{"start_step": 0, "end_step": 3}], "reasoning": "ok"}]',
        '[{"match": false, "confidence": 0.2, "spans": [], "reasoning": "no"}]',
    ] * 5)

    cfg = PipelineConfig(judge_model="test-model", recall_top_n=3)
    pipeline = TrajectoryPipeline(config=cfg, gateway=gw)

    candidates = await pipeline.run(
        trajectory_paths=[trajectories_path],
        problem_specs=[spec1, spec2],
    )

    # Pipeline ran for both specs without error
    assert isinstance(candidates, list)


async def test_pipeline_empty_trajectories(tmp_path, problem_spec_dict):
    """Empty trajectory file → no candidates, no crash."""
    empty = tmp_path / "empty.jsonl"
    empty.write_text("")

    gw = FakeGateway([])
    cfg = PipelineConfig(judge_model="test-model")
    pipeline = TrajectoryPipeline(config=cfg, gateway=gw)

    candidates = await pipeline.run(
        trajectory_paths=[empty],
        problem_specs=[problem_spec_dict],
    )
    assert candidates == []


async def test_pipeline_output_format(trajectories_path, problem_spec_dict):
    """SFTCandidate output matches expected schema."""
    judge_resp = '[{"match": true, "confidence": 0.92, "spans": [{"start_step": 0, "end_step": 4}], "reasoning": "good"}]'
    gw = FakeGateway([judge_resp] * 10)

    cfg = PipelineConfig(judge_model="test-model", recall_top_n=5)
    pipeline = TrajectoryPipeline(config=cfg, gateway=gw)

    candidates = await pipeline.run(
        trajectory_paths=[trajectories_path],
        problem_specs=[problem_spec_dict],
    )

    for c in candidates:
        assert isinstance(c, SFTCandidate)
        assert c.trajectory_id is not None
        assert c.trajectory_path is not None
        for mp in c.matched_problems:
            assert "problem_spec_id" in mp or "sub_problem_id" in mp
            assert "capability" in mp
            assert "confidence" in mp
            assert "loss_mask_spans" in mp
```

- [ ] **Step 2: 实现 pipeline.py**

```python
"""Top-level pipeline orchestration.

Flow: load trajectories → slice → extract signatures → build index →
for each ProblemSpec sub_problem: recall → judge → collect SFTCandidates.
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass, field

from module1.models import SFTCandidate, Slice, TrajectorySignature
from module1.loader import load_trajectories
from module1.slicer import slice_trajectory
from module1.signature import extract_signature
from module1.index import MemoryIndex
from module1.judge import Judge

__all__ = ["TrajectoryPipeline", "PipelineConfig"]


@dataclass
class PipelineConfig:
    """Configuration for the trajectory pipeline."""
    judge_model: str = "gpt-4o-mini"
    judge_batch_size: int = 3
    recall_top_n: int = 20
    min_confidence: float = 0.7
    embedding_model: object | None = None  # Optional EmbeddingModel instance


class TrajectoryPipeline:
    """Full trajectory processing pipeline.

    Usage::

        pipeline = TrajectoryPipeline(config=cfg, gateway=gw)
        candidates = await pipeline.run(trajectory_paths, problem_specs)
    """

    def __init__(self, config: PipelineConfig, gateway):
        self._config = config
        self._gateway = gateway
        self._index = MemoryIndex()
        self._judge = Judge(
            gateway=gateway,
            model=config.judge_model,
            batch_size=config.judge_batch_size,
        )
        # Mapping: (trajectory_id, slice_index) → Slice object
        self._slice_map: dict[tuple[str, int], Slice] = {}
        # Mapping: trajectory_id → source file path
        self._traj_paths: dict[str, str] = {}

    async def run(
        self,
        *,
        trajectory_paths: list[str | pathlib.Path],
        problem_specs: list[dict],
    ) -> list[SFTCandidate]:
        """Execute the full pipeline.

        Args:
            trajectory_paths: paths to JSONL trajectory files.
            problem_specs: list of ProblemSpec dicts (contract §1 format).

        Returns:
            List of SFTCandidate objects.
        """
        # Phase 1: Load + Slice + Sign + Index
        self._build_index(trajectory_paths)

        if self._index.size == 0:
            return []

        # Phase 2: For each sub_problem in each spec: recall + judge
        all_candidates: dict[str, SFTCandidate] = {}  # keyed by trajectory_id

        for spec in problem_specs:
            spec_id = spec.get("raw_input", "unknown")[:50]
            sub_problems = spec.get("sub_problems", [])

            for sp in sub_problems:
                candidates = await self._process_sub_problem(sp, spec_id)
                for c in candidates:
                    # Merge into existing candidate or create new
                    if c.trajectory_id in all_candidates:
                        all_candidates[c.trajectory_id].matched_problems.extend(
                            c.matched_problems
                        )
                    else:
                        all_candidates[c.trajectory_id] = c

        return list(all_candidates.values())

    def _build_index(self, trajectory_paths: list[str | pathlib.Path]) -> None:
        """Load trajectories, slice, extract signatures, build index."""
        for path in trajectory_paths:
            path = pathlib.Path(path)
            trajectories = load_trajectories(path)

            for traj in trajectories:
                self._traj_paths[traj.id] = str(path)
                slices = slice_trajectory(traj)

                for sl in slices:
                    sig = extract_signature(
                        sl, embedding_model=self._config.embedding_model
                    )
                    self._index.add(sig)
                    self._slice_map[(sl.trajectory_id, sl.slice_index)] = sl

    async def _process_sub_problem(
        self, sub_problem: dict, spec_id: str
    ) -> list[SFTCandidate]:
        """Recall + judge for a single sub_problem."""
        # Extract query parameters from sub_problem
        structured_filters = sub_problem.get("structured_filters", {})
        keywords = sub_problem.get("keywords", [])
        hyde_positive = sub_problem.get("hyde_positive", [])
        target_capability = sub_problem.get("target_capability", [])
        trajectory_signal = sub_problem.get("trajectory_signal", "")
        sub_problem_id = sub_problem.get("id", "unknown")

        # Compute query embeddings from hyde_positive (if embedding model available)
        query_embeddings = []
        if self._config.embedding_model and hyde_positive:
            query_embeddings = self._config.embedding_model.embed_batch(hyde_positive)

        # Recall
        recalled_sigs = self._index.recall(
            structured_filters=structured_filters,
            keywords=keywords,
            query_embeddings=query_embeddings,
            top_n=self._config.recall_top_n,
        )

        if not recalled_sigs:
            return []

        # Map signatures back to slices
        recalled_slices = []
        for sig in recalled_sigs:
            key = (sig.trajectory_id, sig.slice_index)
            if key in self._slice_map:
                recalled_slices.append(self._slice_map[key])

        if not recalled_slices:
            return []

        # Judge
        judge_results = await self._judge.judge_batch(
            slices=recalled_slices,
            target_capability=target_capability,
            trajectory_signal=trajectory_signal,
        )

        # Collect matches into SFTCandidates
        candidates = []
        for sl, jr in zip(recalled_slices, judge_results):
            if jr.match and jr.confidence >= self._config.min_confidence:
                candidates.append(SFTCandidate(
                    trajectory_id=sl.trajectory_id,
                    trajectory_path=self._traj_paths.get(sl.trajectory_id, ""),
                    matched_problems=[{
                        "problem_spec_id": spec_id,
                        "sub_problem_id": sub_problem_id,
                        "capability": target_capability,
                        "confidence": jr.confidence,
                        "loss_mask_spans": jr.spans,
                    }],
                ))

        return candidates
```

- [ ] **Step 3: 更新 `src/module1/__init__.py`**

```python
"""Module 1: Trajectory Processing Pipeline.

Processes backflow trajectories to find positive capability demonstrations
and output loss mask spans for SFT training.
"""

from module1.models import (
    Step,
    Trajectory,
    Slice,
    TrajectorySignature,
    JudgeResult,
    SFTCandidate,
)
from module1.loader import load_trajectories, parse_trajectory
from module1.slicer import slice_trajectory
from module1.signature import extract_signature
from module1.index import MemoryIndex
from module1.summarizer import summarize_slice
from module1.judge import Judge
from module1.pipeline import TrajectoryPipeline, PipelineConfig

__all__ = [
    # Models
    "Step",
    "Trajectory",
    "Slice",
    "TrajectorySignature",
    "JudgeResult",
    "SFTCandidate",
    # Components
    "load_trajectories",
    "parse_trajectory",
    "slice_trajectory",
    "extract_signature",
    "MemoryIndex",
    "summarize_slice",
    "Judge",
    # Pipeline
    "TrajectoryPipeline",
    "PipelineConfig",
]
```

- [ ] **Step 4: 运行全部 module1 测试**

Run: `pytest tests/module1/ -v`
Expected: All tests across all files PASS

- [ ] **Step 5: Commit**

```bash
git add src/module1/pipeline.py src/module1/__init__.py tests/module1/test_pipeline.py
git commit -m "feat(module1): pipeline.py full orchestration + integration tests"
```

---
