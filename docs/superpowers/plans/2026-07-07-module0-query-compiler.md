# Module 0: Query Compiler 实现计划 (Plan A)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** 把用户的自然语言问题清单编译成结构化、可检索的 Problem Spec。核心是 4 次 LLM 调用的编排（Call 1 分解 → Call 2 批量打标+自评 → 条件触发 Call 3 澄清 → Call 2' 重评），通过已实现的 `llm_gateway` 发起，产出严格符合接口契约的 Problem Spec。

**Scope（Plan A，本计划）：** Query Compiler 编排 + Taxonomy 读取层 + 本地 Embedding 计算 + Problem Spec 产出与校验。可独立跑通、可与 gateway 联调。

**Out of scope（留给 Plan B）：** Taxonomy 演化（新标签入库/去重/挂载）、异步回填 pipeline、ES/Qdrant 索引。本计划只读取 taxonomy 文件用于 prompt 注入，不写回、不演化。

**权威文档（严格遵守，不漂移）：**
- 设计 Spec：[`../specs/2026-07-06-module0-query-compiler-design.md`](../specs/2026-07-06-module0-query-compiler-design.md)
- 接口契约：[`../specs/interface-contract.md`](../specs/interface-contract.md)（Problem Spec schema / StructuredFilters 枚举 / Taxonomy schema 以此为准）

**Tech Stack:** Python 3.11+, llm_gateway（本仓已实现）, sentence-transformers（Qwen3-Embedding 本地推理，Apple M5 走 MPS）, pytest + pytest-asyncio。

---

## 与契约的关键锁定项（实现时逐条核对，防漂移）

1. **Problem Spec 顶层**：`raw_input`(str) / `domain`(str，固定 `"agentic_swe"`) / `sub_problems`(SubProblem[]，仅 route=="pass")。
2. **SubProblem 12 字段**：`id` / `origin`(original|clarified) / `parent_id`(clarified 必填否则 null) / `raw_text` / `failure_summary` / `target_capability`(1-3) / `trajectory_signal` / `hyde_positive`(2-3 段) / `keywords` / `structured_filters` / `confidence`(0-1) / `route`(恒 "pass")。
3. **StructuredFilters 5 字段 + 冻结枚举**：
   - `languages`: 枚举 `python/cpp/java/javascript/go/bash/html/other`
   - `tools_used`: 枚举 `Bash/Read/Write/Edit/Glob/Grep/WebFetch/WebSearch/Task/TodoWrite/NotebookEdit/other`
   - `outcome_transition`: 枚举 `failed→success/success_only/failed_only/no_execution`
   - `min_turns`: int ≥1
   - `has_verification_step`: bool
   - 全部可选（可为 null）。
4. **drop_reason 枚举**：`ambiguous`(触发 Part 6.5) / `not_applicable` / `label_diverged` / `other`。被 drop 的子问题**不进入** `sub_problems[]`，单独存审计。
5. **调用上限恒为 4**：Call 3 + Call 2' 是原子对，至多一次，不递归。澄清产出的子问题不再输出 drop_reason、不再触发 Part 6.5。
6. **Taxonomy**：冷启动空词表；v0 快照 5 顶层 12 叶子（契约附录 A）。标签命名：小写英文+下划线、动宾结构、描述"正确做法"。
7. **Embedding**：Qwen3-Embedding，1536 维，L2 归一化。仅对 `hyde_positive` 各段做 embedding（查询锚）。

---

## 文件结构

包位于 `src/module0/`。每个文件单一职责。

| 文件 | 职责 | 依赖 |
|------|------|------|
| `src/module0/schema.py` | Problem Spec / SubProblem / StructuredFilters 的 dataclass + 枚举 + 校验 | (无) |
| `src/module0/taxonomy.py` | 加载 taxonomy JSON，提供 prompt 注入文本 + 标签查询（只读，不演化） | schema |
| `src/module0/prompts.py` | Call 1/2/3/2' 的 prompt 模板构建 | taxonomy |
| `src/module0/embedding.py` | Qwen3-Embedding 本地推理封装（HyDE → 1536-d L2） | (无) |
| `src/module0/parsing.py` | LLM 原始文本 → 结构化对象（JSON 解析 + schema 校验 + 容错重试判定） | schema |
| `src/module0/compiler.py` | QueryCompiler 编排类：串联 Call 1/2/3/2'，产出 Problem Spec | 以上全部 + llm_gateway |
| `src/module0/__init__.py` | 公共导出 | compiler, schema |

测试在 `tests/module0/` 一一对应。Fixtures 在 `fixtures/`（taxonomy_v0.json + 测试 Problem Spec）。

---

## 依赖关系

```
schema.py       ← 叶子（纯数据 + 校验）
embedding.py    ← 叶子（独立，仅依赖 sentence-transformers）
taxonomy.py     ← schema
prompts.py      ← taxonomy
parsing.py      ← schema
compiler.py     ← schema, taxonomy, prompts, embedding, parsing, llm_gateway
```

**环境：** venv 已存在于 `.venv`（Python 3.11）。所有命令假定已激活：`source .venv/bin/activate`。

---

## Task 0: 基础设施 + fixtures

**Files:**
- Modify: `pyproject.toml`（加依赖 + module0 包路径）
- Create: `fixtures/taxonomy_v0.json`
- Create: `fixtures/problem_specs/test_01.json`
- Create: `tests/module0/__init__.py`
- Create: `tests/module0/conftest.py`

- [ ] **Step 1: 修改 pyproject.toml**

在 `[project] dependencies` 加入：
```toml
dependencies = [
    "httpx>=0.27",
    "sentence-transformers>=3.0",
]
```

在 `[tool.hatch.build.targets.wheel]` 改为：
```toml
packages = ["src/llm_gateway", "src/module0"]
```

- [ ] **Step 2: 创建 fixtures/taxonomy_v0.json**

严格按契约附录 A，JSON 格式（schema 对齐契约 §5.1 TaxonomyLabel）：

```json
{
  "version": "0.1.0",
  "updated_at": "2026-07-06T00:00:00Z",
  "labels": [
    {"label": "code_generation", "parent": null, "new_root": false, "description": "代码生成相关能力", "keywords": ["code", "generate", "write"], "description_embedding": [], "taxonomy_extension": false, "created_at": "2026-07-06T00:00:00Z"},
    {"label": "valid_syntax_in_toolcall", "parent": "code_generation", "new_root": false, "description": "工具调用中生成合法代码（含 import）", "keywords": ["syntax", "import", "SyntaxError"], "description_embedding": [], "taxonomy_extension": false, "created_at": "2026-07-06T00:00:00Z"},
    {"label": "correct_indentation", "parent": "code_generation", "new_root": false, "description": "写入文件时缩进正确", "keywords": ["indentation", "indent", "whitespace", "tab"], "description_embedding": [], "taxonomy_extension": false, "created_at": "2026-07-06T00:00:00Z"},
    {"label": "correct_shell_embedding", "parent": "code_generation", "new_root": false, "description": "bash 中正确嵌入其他语言", "keywords": ["bash", "shell", "heredoc", "escape"], "description_embedding": [], "taxonomy_extension": false, "created_at": "2026-07-06T00:00:00Z"},
    {"label": "file_localization_and_edit", "parent": "code_generation", "new_root": false, "description": "正确定位目标文件并成功编辑", "keywords": ["file", "edit", "locate", "path"], "description_embedding": [], "taxonomy_extension": false, "created_at": "2026-07-06T00:00:00Z"},
    {"label": "tool_use", "parent": null, "new_root": false, "description": "工具使用相关能力", "keywords": ["tool", "function_call", "tool_call"], "description_embedding": [], "taxonomy_extension": false, "created_at": "2026-07-06T00:00:00Z"},
    {"label": "wellformed_tool_call", "parent": "tool_use", "new_root": false, "description": "tool_call 结构正确，不错位到 thinking 块", "keywords": ["tool_call", "JSON", "wellformed", "malformed"], "description_embedding": [], "taxonomy_extension": false, "created_at": "2026-07-06T00:00:00Z"},
    {"label": "planning", "parent": null, "new_root": false, "description": "规划相关能力", "keywords": ["plan", "requirement", "analysis"], "description_embedding": [], "taxonomy_extension": false, "created_at": "2026-07-06T00:00:00Z"},
    {"label": "requirement_completeness", "parent": "planning", "new_root": false, "description": "覆盖需求中所有明确功能点", "keywords": ["requirement", "complete", "coverage", "feature"], "description_embedding": [], "taxonomy_extension": false, "created_at": "2026-07-06T00:00:00Z"},
    {"label": "requirement_analysis_before_coding", "parent": "planning", "new_root": false, "description": "复杂任务先分析需求再编码", "keywords": ["analysis", "understand", "before", "coding"], "description_embedding": [], "taxonomy_extension": false, "created_at": "2026-07-06T00:00:00Z"},
    {"label": "execution_control", "parent": null, "new_root": false, "description": "执行控制相关能力", "keywords": ["execution", "control", "loop", "strategy"], "description_embedding": [], "taxonomy_extension": false, "created_at": "2026-07-06T00:00:00Z"},
    {"label": "reproduce_before_fix", "parent": "execution_control", "new_root": false, "description": "修改前先复现问题", "keywords": ["reproduce", "verify", "before", "fix"], "description_embedding": [], "taxonomy_extension": false, "created_at": "2026-07-06T00:00:00Z"},
    {"label": "avoid_redundant_repetition", "parent": "execution_control", "new_root": false, "description": "避免重复动作循环，及时切换策略", "keywords": ["repeat", "loop", "stuck", "strategy"], "description_embedding": [], "taxonomy_extension": false, "created_at": "2026-07-06T00:00:00Z"},
    {"label": "self_verification", "parent": "execution_control", "new_root": false, "description": "实现后自主验证，不依赖用户提示才发现错误", "keywords": ["verify", "test", "check", "self"], "description_embedding": [], "taxonomy_extension": false, "created_at": "2026-07-06T00:00:00Z"},
    {"label": "error_recovery", "parent": null, "new_root": false, "description": "错误恢复相关能力", "keywords": ["error", "fix", "recovery", "debug"], "description_embedding": [], "taxonomy_extension": false, "created_at": "2026-07-06T00:00:00Z"},
    {"label": "effective_error_fix", "parent": "error_recovery", "new_root": false, "description": "用户指出后修复真正生效", "keywords": ["fix", "resolve", "effective", "bug"], "description_embedding": [], "taxonomy_extension": false, "created_at": "2026-07-06T00:00:00Z"}
  ]
}
```

注意：`description_embedding` 暂为空数组（Task embedding 实现后填充）。5 顶层 + 12 叶子 = 17 条目（含 5 个 parent-only 节点）。对齐契约附录 A 的完整内容。

- [ ] **Step 3: 创建 fixtures/problem_specs/test_01.json**

一个 golden fixture，用于端到端测试验证 schema 合规：

```json
{
  "raw_input": "写入py文件有语法错误，并且工具调用结构经常出错",
  "domain": "agentic_swe",
  "sub_problems": [
    {
      "id": "p1",
      "origin": "original",
      "parent_id": null,
      "raw_text": "写入py文件有语法错误",
      "failure_summary": "写入 py 文件时产生语法错误",
      "target_capability": ["valid_syntax_in_toolcall"],
      "trajectory_signal": "observation 含 SyntaxError 且前序 tool_call 含 python 代码写入",
      "hyde_positive": ["<假设: 工具正确写入 python 文件，无语法错误，执行结果 exit code 0>", "<假设: python 文件包含合法 import 和函数定义，lint 通过>"],
      "keywords": ["SyntaxError", "python", "import"],
      "structured_filters": {"languages": ["python"], "tools_used": ["Write", "Edit"], "outcome_transition": ["failed→success"], "min_turns": null, "has_verification_step": null},
      "confidence": 0.92,
      "route": "pass"
    },
    {
      "id": "p2",
      "origin": "original",
      "parent_id": null,
      "raw_text": "工具调用结构经常出错",
      "failure_summary": "tool_call JSON 结构畸形或错位到 thinking 块",
      "target_capability": ["wellformed_tool_call"],
      "trajectory_signal": "tool_call 块 JSON 解析失败或 tool_call 内容出现在 thinking/content 字段",
      "hyde_positive": ["<假设: tool_call 正确使用 JSON 格式，包含 name + arguments 字段>", "<假设: thinking 块不包含 tool_call 内容，工具调用独立且结构清晰>"],
      "keywords": ["tool_call", "JSON", "malformed", "thinking"],
      "structured_filters": {"languages": null, "tools_used": null, "outcome_transition": ["failed→success"], "min_turns": null, "has_verification_step": null},
      "confidence": 0.88,
      "route": "pass"
    }
  ]
}
```

- [ ] **Step 4: 创建 tests/module0/conftest.py**

```python
import json
import pathlib
import pytest

FIXTURES_DIR = pathlib.Path(__file__).parent.parent.parent / "fixtures"


@pytest.fixture
def taxonomy_v0_path():
    return FIXTURES_DIR / "taxonomy_v0.json"


@pytest.fixture
def taxonomy_v0(taxonomy_v0_path):
    with open(taxonomy_v0_path) as f:
        return json.load(f)


@pytest.fixture
def golden_problem_spec():
    path = FIXTURES_DIR / "problem_specs" / "test_01.json"
    with open(path) as f:
        return json.load(f)
```

- [ ] **Step 5: pip install + commit**

```bash
pip install -e ".[dev]"
git add pyproject.toml fixtures/ tests/module0/
git commit -m "chore(module0): add fixtures, test scaffold, and sentence-transformers dep"
```

---

## Task 1: schema.py — Problem Spec 数据模型 + 校验

**Files:**
- Create: `src/module0/schema.py`
- Create: `src/module0/__init__.py`（占位，最终导出等 Task 6）
- Test: `tests/module0/test_schema.py`

严格对齐契约 §1 Problem Spec Schema + §2 StructuredFilters。枚举值从契约冻结，不允许实现者自由发挥。

- [ ] **Step 1: 写失败测试** `tests/module0/test_schema.py`

```python
import pytest
from module0.schema import (
    ProblemSpec,
    SubProblem,
    StructuredFilters,
    DroppedSubProblem,
    LANGUAGES,
    TOOLS_USED,
    OUTCOME_TRANSITIONS,
    DROP_REASONS,
    validate_problem_spec,
)


def test_structured_filters_enums():
    assert "python" in LANGUAGES
    assert "other" in LANGUAGES
    assert len(LANGUAGES) == 8
    assert "Bash" in TOOLS_USED
    assert "other" in TOOLS_USED
    assert len(TOOLS_USED) == 12
    assert "failed→success" in OUTCOME_TRANSITIONS
    assert len(OUTCOME_TRANSITIONS) == 4
    assert "ambiguous" in DROP_REASONS
    assert len(DROP_REASONS) == 4


def test_sub_problem_valid():
    sp = SubProblem(
        id="p1",
        origin="original",
        parent_id=None,
        raw_text="test",
        failure_summary="test summary",
        target_capability=["valid_syntax_in_toolcall"],
        trajectory_signal="signal",
        hyde_positive=["hyp1", "hyp2"],
        keywords=["kw1"],
        structured_filters=StructuredFilters(languages=["python"]),
        confidence=0.9,
        route="pass",
    )
    assert sp.id == "p1"
    assert sp.origin == "original"


def test_sub_problem_clarified_requires_parent_id():
    with pytest.raises(ValueError, match="parent_id"):
        SubProblem(
            id="p1a",
            origin="clarified",
            parent_id=None,  # must not be None for clarified
            raw_text="t",
            failure_summary="s",
            target_capability=["x"],
            trajectory_signal="s",
            hyde_positive=["h1", "h2"],
            keywords=["k"],
            structured_filters=StructuredFilters(),
            confidence=0.85,
            route="pass",
        )


def test_sub_problem_target_capability_1_to_3():
    with pytest.raises(ValueError, match="target_capability"):
        SubProblem(
            id="p1", origin="original", parent_id=None,
            raw_text="t", failure_summary="s",
            target_capability=[],  # must be 1-3
            trajectory_signal="s", hyde_positive=["h1", "h2"],
            keywords=["k"], structured_filters=StructuredFilters(),
            confidence=0.9, route="pass",
        )


def test_structured_filters_validates_enums():
    with pytest.raises(ValueError, match="languages"):
        StructuredFilters(languages=["invalid_lang"])


def test_problem_spec_valid(golden_problem_spec):
    ps = validate_problem_spec(golden_problem_spec)
    assert ps.domain == "agentic_swe"
    assert len(ps.sub_problems) == 2
    assert ps.sub_problems[0].id == "p1"


def test_problem_spec_rejects_drop_in_sub_problems():
    with pytest.raises(ValueError, match="route"):
        validate_problem_spec({
            "raw_input": "test",
            "domain": "agentic_swe",
            "sub_problems": [{"id": "p1", "route": "drop"}],
        })


def test_dropped_sub_problem():
    d = DroppedSubProblem(
        id="p5", origin="original", parent_id=None,
        raw_text="t", failure_summary="s",
        target_capability=["x"], trajectory_signal="s",
        hyde_positive=["h1", "h2"], keywords=["k"],
        structured_filters=StructuredFilters(),
        confidence=0.5, route="drop",
        drop_reason="not_applicable",
    )
    assert d.route == "drop"
    assert d.drop_reason == "not_applicable"
```

- [ ] **Step 2: 运行测试，确认失败**

```bash
pytest tests/module0/test_schema.py -v
```
预期：`ModuleNotFoundError: No module named 'module0'`

- [ ] **Step 3: 实现 schema.py**

创建 `src/module0/__init__.py`（空占位）和 `src/module0/schema.py`：

```python
"""Problem Spec schema — strict alignment with interface-contract.md.

All enumerations, field names, and validation rules are frozen per contract.
Do NOT add, rename, or relax without updating the contract document.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "ProblemSpec",
    "SubProblem",
    "StructuredFilters",
    "DroppedSubProblem",
    "LANGUAGES",
    "TOOLS_USED",
    "OUTCOME_TRANSITIONS",
    "DROP_REASONS",
    "validate_problem_spec",
]

# ── 契约 §2 冻结枚举 ──

LANGUAGES = frozenset([
    "python", "cpp", "java", "javascript", "go", "bash", "html", "other",
])

TOOLS_USED = frozenset([
    "Bash", "Read", "Write", "Edit", "Glob", "Grep",
    "WebFetch", "WebSearch", "Task", "TodoWrite", "NotebookEdit", "other",
])

OUTCOME_TRANSITIONS = frozenset([
    "failed→success", "success_only", "failed_only", "no_execution",
])

DROP_REASONS = frozenset([
    "ambiguous", "not_applicable", "label_diverged", "other",
])


@dataclass
class StructuredFilters:
    """契约 §1.3 StructuredFilters — 5 fields, all optional."""
    languages: list[str] | None = None
    tools_used: list[str] | None = None
    outcome_transition: list[str] | None = None
    min_turns: int | None = None
    has_verification_step: bool | None = None

    def __post_init__(self):
        if self.languages:
            for v in self.languages:
                if v not in LANGUAGES:
                    raise ValueError(f"languages: invalid value '{v}'. Must be one of {sorted(LANGUAGES)}")
        if self.tools_used:
            for v in self.tools_used:
                if v not in TOOLS_USED:
                    raise ValueError(f"tools_used: invalid value '{v}'. Must be one of {sorted(TOOLS_USED)}")
        if self.outcome_transition:
            for v in self.outcome_transition:
                if v not in OUTCOME_TRANSITIONS:
                    raise ValueError(f"outcome_transition: invalid value '{v}'. Must be one of {sorted(OUTCOME_TRANSITIONS)}")
        if self.min_turns is not None and self.min_turns < 1:
            raise ValueError("min_turns must be >= 1")


@dataclass
class SubProblem:
    """契约 §1.2 SubProblem — 12 required fields."""
    id: str
    origin: str  # "original" | "clarified"
    parent_id: str | None
    raw_text: str
    failure_summary: str
    target_capability: list[str]  # 1-3 labels
    trajectory_signal: str
    hyde_positive: list[str]  # 2-3 segments
    keywords: list[str]
    structured_filters: StructuredFilters
    confidence: float  # 0-1
    route: str  # "pass" in sub_problems[]

    def __post_init__(self):
        if self.origin not in ("original", "clarified"):
            raise ValueError(f"origin must be 'original' or 'clarified', got '{self.origin}'")
        if self.origin == "clarified" and not self.parent_id:
            raise ValueError("parent_id is required when origin=='clarified'")
        if not (1 <= len(self.target_capability) <= 3):
            raise ValueError(f"target_capability must have 1-3 items, got {len(self.target_capability)}")
        if not (2 <= len(self.hyde_positive) <= 3):
            raise ValueError(f"hyde_positive must have 2-3 segments, got {len(self.hyde_positive)}")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"confidence must be 0-1, got {self.confidence}")
        if self.route != "pass":
            raise ValueError(f"route must be 'pass' in sub_problems[], got '{self.route}'")


@dataclass
class DroppedSubProblem:
    """契约 §1.4 — dropped sub-problems for audit (not in sub_problems[])."""
    id: str
    origin: str
    parent_id: str | None
    raw_text: str
    failure_summary: str
    target_capability: list[str]
    trajectory_signal: str
    hyde_positive: list[str]
    keywords: list[str]
    structured_filters: StructuredFilters
    confidence: float
    route: str  # "drop"
    drop_reason: str  # one of DROP_REASONS

    def __post_init__(self):
        if self.route != "drop":
            raise ValueError(f"route must be 'drop' for DroppedSubProblem, got '{self.route}'")
        if self.drop_reason not in DROP_REASONS:
            raise ValueError(f"drop_reason must be one of {sorted(DROP_REASONS)}, got '{self.drop_reason}'")


@dataclass
class ProblemSpec:
    """契约 §1.1 顶层结构。"""
    raw_input: str
    domain: str  # 固定 "agentic_swe"
    sub_problems: list[SubProblem]


def _parse_structured_filters(d: dict | None) -> StructuredFilters:
    if not d:
        return StructuredFilters()
    return StructuredFilters(
        languages=d.get("languages"),
        tools_used=d.get("tools_used"),
        outcome_transition=d.get("outcome_transition"),
        min_turns=d.get("min_turns"),
        has_verification_step=d.get("has_verification_step"),
    )


def validate_problem_spec(data: dict) -> ProblemSpec:
    """Parse and validate a raw dict into a ProblemSpec.

    Raises ValueError on any schema violation.
    """
    sub_problems = []
    for sp_data in data.get("sub_problems", []):
        if sp_data.get("route") != "pass":
            raise ValueError(f"route must be 'pass' in sub_problems[], got '{sp_data.get('route')}'")
        sub_problems.append(SubProblem(
            id=sp_data["id"],
            origin=sp_data["origin"],
            parent_id=sp_data.get("parent_id"),
            raw_text=sp_data["raw_text"],
            failure_summary=sp_data["failure_summary"],
            target_capability=sp_data["target_capability"],
            trajectory_signal=sp_data["trajectory_signal"],
            hyde_positive=sp_data["hyde_positive"],
            keywords=sp_data["keywords"],
            structured_filters=_parse_structured_filters(sp_data.get("structured_filters")),
            confidence=sp_data["confidence"],
            route=sp_data["route"],
        ))
    return ProblemSpec(
        raw_input=data["raw_input"],
        domain=data["domain"],
        sub_problems=sub_problems,
    )
```

- [ ] **Step 4: 运行测试，确认全过**

```bash
pip install -e ".[dev]"  # 确保 module0 可 import
pytest tests/module0/test_schema.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/module0/ tests/module0/test_schema.py
git commit -m "feat(module0): schema.py with Problem Spec dataclasses + validation"
```

---

## Task 2: taxonomy.py — Taxonomy 只读层

**Files:**
- Create: `src/module0/taxonomy.py`
- Test: `tests/module0/test_taxonomy.py`

提供 taxonomy JSON 加载 + prompt 注入文本生成 + 标签查询。只读，不写回。对齐契约 §5 TaxonomyLabel schema。

- [ ] **Step 1: 写失败测试** `tests/module0/test_taxonomy.py`

```python
import pytest
from module0.taxonomy import Taxonomy


def test_load_from_file(taxonomy_v0_path):
    t = Taxonomy.load(taxonomy_v0_path)
    assert t.version == "0.1.0"
    assert len(t.labels) == 17  # 5 parent + 12 leaf


def test_leaf_labels(taxonomy_v0_path):
    t = Taxonomy.load(taxonomy_v0_path)
    leaves = t.leaf_labels()
    assert len(leaves) == 12
    assert "valid_syntax_in_toolcall" in [l.label for l in leaves]


def test_get_label(taxonomy_v0_path):
    t = Taxonomy.load(taxonomy_v0_path)
    label = t.get("valid_syntax_in_toolcall")
    assert label is not None
    assert label.parent == "code_generation"


def test_get_nonexistent(taxonomy_v0_path):
    t = Taxonomy.load(taxonomy_v0_path)
    assert t.get("nonexistent_label") is None


def test_prompt_injection_text(taxonomy_v0_path):
    t = Taxonomy.load(taxonomy_v0_path)
    text = t.to_prompt_text()
    # Should contain indented tree-like structure
    assert "code_generation" in text
    assert "valid_syntax_in_toolcall" in text
    # Leaf labels should include description
    assert "工具调用中生成合法代码" in text


def test_empty_taxonomy():
    t = Taxonomy.from_dict({"version": "0.0.0", "updated_at": "2026-01-01T00:00:00Z", "labels": []})
    assert len(t.labels) == 0
    assert t.to_prompt_text() == ""
    assert t.leaf_labels() == []


def test_is_empty_vs_non_empty(taxonomy_v0_path):
    t = Taxonomy.load(taxonomy_v0_path)
    assert not t.is_empty
    empty = Taxonomy.from_dict({"version": "0.0.0", "updated_at": "", "labels": []})
    assert empty.is_empty
```

- [ ] **Step 2: 运行测试，确认失败**

- [ ] **Step 3: 实现 taxonomy.py**

```python
"""Taxonomy read-only layer.

Loads a taxonomy JSON file (contract §5 schema), provides:
- Label lookup by name
- Leaf vs parent classification
- Prompt injection text generation (indented tree for LLM context)

Does NOT write back or evolve taxonomy (that's Plan B / module 0.5).
"""

from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass

__all__ = ["Taxonomy", "TaxonomyLabel"]


@dataclass(frozen=True)
class TaxonomyLabel:
    """契约 §5.1 TaxonomyLabel（只读视图）。"""
    label: str
    parent: str | None
    new_root: bool
    description: str
    keywords: list[str]
    description_embedding: list[float]
    taxonomy_extension: bool
    created_at: str


class Taxonomy:
    """Read-only taxonomy store."""

    def __init__(self, version: str, updated_at: str, labels: list[TaxonomyLabel]):
        self.version = version
        self.updated_at = updated_at
        self.labels = labels
        self._by_name: dict[str, TaxonomyLabel] = {l.label: l for l in labels}

    @classmethod
    def load(cls, path: str | pathlib.Path) -> "Taxonomy":
        with open(path) as f:
            data = json.load(f)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict) -> "Taxonomy":
        labels = []
        for entry in data.get("labels", []):
            labels.append(TaxonomyLabel(
                label=entry["label"],
                parent=entry.get("parent"),
                new_root=entry.get("new_root", False),
                description=entry["description"],
                keywords=entry.get("keywords", []),
                description_embedding=entry.get("description_embedding", []),
                taxonomy_extension=entry.get("taxonomy_extension", False),
                created_at=entry.get("created_at", ""),
            ))
        return cls(
            version=data.get("version", "0.0.0"),
            updated_at=data.get("updated_at", ""),
            labels=labels,
        )

    @property
    def is_empty(self) -> bool:
        return len(self.labels) == 0

    def get(self, label_name: str) -> TaxonomyLabel | None:
        return self._by_name.get(label_name)

    def leaf_labels(self) -> list[TaxonomyLabel]:
        """Labels that are not parent of any other label."""
        parents = {l.parent for l in self.labels if l.parent}
        return [l for l in self.labels if l.label not in parents]

    def to_prompt_text(self) -> str:
        """Generate indented tree text for LLM prompt injection.

        Format (per spec Part 2):
          parent_label (parent: null)
            ├── child_label  中文description
        """
        if not self.labels:
            return ""

        # Group children by parent
        children: dict[str | None, list[TaxonomyLabel]] = {}
        for l in self.labels:
            children.setdefault(l.parent, []).append(l)

        lines = []
        # Top-level (parent=None)
        for root in children.get(None, []):
            lines.append(f"{root.label}  ({root.description})")
            for child in children.get(root.label, []):
                lines.append(f"  ├── {child.label}  {child.description}")
        return "\n".join(lines)
```

- [ ] **Step 4: 运行测试，确认全过**

```bash
pytest tests/module0/test_taxonomy.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/module0/taxonomy.py tests/module0/test_taxonomy.py
git commit -m "feat(module0): taxonomy.py read-only layer with prompt injection"
```

---

## Task 3: embedding.py — Qwen3-Embedding 本地推理

**Files:**
- Create: `src/module0/embedding.py`
- Test: `tests/module0/test_embedding.py`

对 `hyde_positive` 各段做 embedding，产出 1536-d L2 归一化向量。用 sentence-transformers 加载 Qwen3-Embedding-0.6B。Apple M5 走 MPS 后端。

注意：首次 `import` 时加载模型需要下载 ~1.2GB。测试时用 mock 或很短文本验证维度和归一化。

- [ ] **Step 1: 写失败测试** `tests/module0/test_embedding.py`

```python
import numpy as np
import pytest

from module0.embedding import EmbeddingModel


@pytest.fixture(scope="module")
def model():
    """Load model once for all tests in this module (expensive init)."""
    return EmbeddingModel()


def test_model_loads(model):
    assert model is not None
    assert model.dimension == 1536


def test_embed_single(model):
    vec = model.embed("hello world")
    assert len(vec) == 1536
    # L2 normalized — norm should be ~1.0
    norm = np.linalg.norm(vec)
    assert abs(norm - 1.0) < 0.01


def test_embed_batch(model):
    texts = ["first text", "second text", "third text"]
    vecs = model.embed_batch(texts)
    assert len(vecs) == 3
    assert all(len(v) == 1536 for v in vecs)
    # Each vector L2 normalized
    for v in vecs:
        assert abs(np.linalg.norm(v) - 1.0) < 0.01


def test_embed_hyde_positive(model):
    """Simulate embedding hyde_positive segments (2-3 per sub-problem)."""
    segments = [
        "工具正确写入 python 文件，无语法错误，执行结果 exit code 0",
        "python 文件包含合法 import 和函数定义，lint 通过",
    ]
    vecs = model.embed_batch(segments)
    assert len(vecs) == 2
    # Vectors should be different (not degenerate)
    cos_sim = np.dot(vecs[0], vecs[1])
    assert cos_sim < 0.99  # related but not identical
```

- [ ] **Step 2: 运行测试，确认失败**

- [ ] **Step 3: 实现 embedding.py**

```python
"""Local embedding via Qwen3-Embedding (sentence-transformers).

Produces 1536-d L2-normalized vectors. Runs on MPS (Apple Silicon) or CPU.
Model loaded lazily on first call to avoid import-time overhead.

Contract §4: both sides MUST use the same model + same dimension.
"""

from __future__ import annotations

import numpy as np

__all__ = ["EmbeddingModel"]

# Model identifier — pinned per contract §4/§6
_MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
_DIMENSION = 1536


class EmbeddingModel:
    """Wrapper for local Qwen3-Embedding inference."""

    def __init__(self, model_name: str = _MODEL_NAME, device: str | None = None):
        from sentence_transformers import SentenceTransformer
        import torch

        if device is None:
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"

        self._model = SentenceTransformer(model_name, device=device)
        self._dimension = _DIMENSION

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, text: str) -> list[float]:
        """Embed a single text. Returns L2-normalized 1536-d vector."""
        vec = self._model.encode(
            text,
            normalize_embeddings=True,
            output_value="sentence_embedding",
        )
        # Truncate/pad to target dimension (MRL support)
        vec = self._ensure_dimension(vec)
        return vec.tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts. Returns list of L2-normalized 1536-d vectors."""
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
        """Truncate or zero-pad to self._dimension, then re-normalize."""
        if len(vec) >= self._dimension:
            vec = vec[:self._dimension]
        else:
            vec = np.pad(vec, (0, self._dimension - len(vec)))
        # Re-normalize after truncation (MRL best practice)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec
```

- [ ] **Step 4: 添加 numpy 到 dev 依赖（test 用）**

在 pyproject.toml `[project.optional-dependencies]` dev 列表加 `"numpy"`.

- [ ] **Step 5: 运行测试，确认全过**

```bash
pip install -e ".[dev]"
pytest tests/module0/test_embedding.py -v
```

注意：首次跑会下载模型。如果网络不通或模型名不可用，标 BLOCKED 报告。

- [ ] **Step 6: Commit**

```bash
git add src/module0/embedding.py tests/module0/test_embedding.py pyproject.toml
git commit -m "feat(module0): embedding.py Qwen3-Embedding local inference (1536-d, L2)"
```

---

## Task 4: prompts.py — LLM Prompt 模板构建

**Files:**
- Create: `src/module0/prompts.py`
- Test: `tests/module0/test_prompts.py`

构建 Call 1/2/3/2' 的 system+user messages。Prompt 设计严格遵循 spec §2 各 Part 要求。词表通过 `{taxonomy_injection}` 注入（spec Part 2 契约 §5.3）。

- [ ] **Step 1: 写失败测试** `tests/module0/test_prompts.py`

```python
import pytest
from module0.prompts import build_call1_messages, build_call2_messages, build_call3_messages
from module0.taxonomy import Taxonomy


def test_call1_messages_structure():
    msgs = build_call1_messages("写入py文件有语法错误，并且工具调用结构经常出错")
    assert len(msgs) >= 2  # system + user
    assert msgs[0]["role"] == "system"
    assert msgs[-1]["role"] == "user"
    assert "写入py文件有语法错误" in msgs[-1]["content"]


def test_call1_system_prompt_contains_rules():
    msgs = build_call1_messages("test input")
    system = msgs[0]["content"]
    # Spec Part 1: decomposition rules
    assert "子问题" in system or "sub-problem" in system.lower()
    assert "JSON" in system


def test_call2_messages_with_taxonomy(taxonomy_v0_path):
    taxonomy = Taxonomy.load(taxonomy_v0_path)
    sub_problems = [
        {"id": "p1", "raw_text": "写入py文件有语法错误", "failure_summary": "写入 py 文件语法错误"},
    ]
    msgs = build_call2_messages(sub_problems, taxonomy)
    assert len(msgs) >= 2
    system = msgs[0]["content"]
    # Taxonomy injected into prompt
    assert "valid_syntax_in_toolcall" in system
    # Spec: StructuredFilters fields mentioned
    assert "structured_filters" in system or "languages" in system


def test_call2_messages_empty_taxonomy():
    taxonomy = Taxonomy.from_dict({"version": "0.0.0", "updated_at": "", "labels": []})
    sub_problems = [
        {"id": "p1", "raw_text": "test", "failure_summary": "test"},
    ]
    msgs = build_call2_messages(sub_problems, taxonomy)
    system = msgs[0]["content"]
    # Spec Part 2: empty taxonomy -> free proposal mode
    assert "自由提议" in system or "propose" in system.lower() or "freely" in system.lower()


def test_call3_messages():
    ambiguous_problems = [
        {"id": "p12", "raw_text": "多步任务未完成即终止", "failure_summary": "多步任务中途停止",
         "target_capability": ["persist_through_truncation"]},
    ]
    msgs = build_call3_messages(ambiguous_problems)
    assert len(msgs) >= 2
    system = msgs[0]["content"]
    # Spec Part 6.5: disambiguation
    assert "消歧" in system or "disambiguat" in system.lower()
    user = msgs[-1]["content"]
    assert "p12" in user


def test_call2_output_schema_mentioned():
    """Prompt must instruct LLM to output structured JSON per spec."""
    taxonomy = Taxonomy.from_dict({"version": "0.0.0", "updated_at": "", "labels": []})
    msgs = build_call2_messages([{"id": "p1", "raw_text": "t", "failure_summary": "s"}], taxonomy)
    system = msgs[0]["content"]
    # Must mention key output fields
    for field in ["target_capability", "trajectory_signal", "hyde_positive", "keywords", "confidence", "route"]:
        assert field in system, f"Missing field '{field}' in Call 2 system prompt"
```

- [ ] **Step 2: 运行测试，确认失败**

- [ ] **Step 3: 实现 prompts.py**

```python
"""LLM prompt templates for Call 1/2/3/2'.

Each function returns a list[dict] (OpenAI messages format) ready to pass
to gateway.call(). Prompt design follows spec §2 Part 1-6.5 exactly.

Key design decisions:
- Call 2 system prompt injects taxonomy via Taxonomy.to_prompt_text()
- Empty taxonomy triggers "free proposal" mode (spec Part 2 table)
- Call 2' reuses Call 2 prompt but with explicit "不输出 drop_reason" constraint
- All prompts request JSON output with specified schema
"""

from __future__ import annotations

from module0.taxonomy import Taxonomy

__all__ = [
    "build_call1_messages",
    "build_call2_messages",
    "build_call3_messages",
    "build_call2_prime_messages",
]

# ── Call 1: 子问题分解 (Part 1) ──

_CALL1_SYSTEM = """你是一个问题分解专家。任务：把用户的问题描述拆分成互不重叠的原子子问题。

## 分解规则
- 按逗号/分号/语义转折切分
- 合并重复描述
- 每条必须是单一失败模式
- 整句只描述一个问题则输出单条
- 无法确定是否该拆时保守不拆，标 needs_review: true

## 输出格式（严格 JSON）
```json
{
  "sub_problems": [
    {
      "id": "p1",
      "raw_text": "从原始描述切出的片段",
      "failure_summary": "一句话标准化失败描述"
    }
  ]
}
```

只输出 JSON，不要其他解释。"""


def build_call1_messages(raw_input: str) -> list[dict]:
    return [
        {"role": "system", "content": _CALL1_SYSTEM},
        {"role": "user", "content": raw_input},
    ]


# ── Call 2: 批量打标 + 自评 (Part 2-6) ──

_CALL2_SYSTEM_WITH_TAXONOMY = """你是一个 SWE 问题分析专家。对每个子问题完成以下全部字段：

## 你可用的能力标签词表
{taxonomy_injection}

优先从上述词表选取标签（1-3个）。确无匹配才提议新标签（小写英文+下划线，动宾结构，描述"正确做法"，附 description + parent 建议，标 taxonomy_extension: true）。

## 结构化过滤条件 structured_filters
字段（全部可选，无则设 null）：
- languages: {languages_enum}
- tools_used: {tools_enum}
- outcome_transition: {outcome_enum}
- min_turns: 整数 ≥1
- has_verification_step: 布尔值

## 输出格式（严格 JSON 数组）
对每个子问题输出：
```json
{{
  "id": "p1",
  "target_capability": ["label1"],
  "trajectory_signal": "在轨迹中应匹配什么模式",
  "hyde_positive": ["假设正例片段1(200-500token)", "假设正例片段2"],
  "keywords": ["关键词1", "关键词2"],
  "structured_filters": {{"languages": ["python"], "outcome_transition": ["failed→success"], ...}},
  "confidence": 0.0-1.0,
  "route": "pass" 或 "drop",
  "drop_reason": "ambiguous|not_applicable|label_diverged|other（仅 route==drop 时填）"
}}
```

## 置信度评估维度
- 子问题是否有歧义（高权重）
- 能力标签是否唯一命中（中）
- trajectory_signal 是否可操作（中）
- HyDE 变体是否一致（低）

## 分流规则
- confidence ≥ 0.8 → route: "pass"
- confidence < 0.8 → route: "drop"，并标注 drop_reason

只输出 JSON 数组，不要其他解释。"""

_CALL2_SYSTEM_EMPTY_TAXONOMY = """你是一个 SWE 问题分析专家。对每个子问题完成以下全部字段：

## 能力标签
当前词表为空（冷启动）。请自由提议标签（1-3个）：
- 命名规则：小写英文+下划线，动宾结构，描述"正确做法"
- 每个标签附：description（一句话中文）、parent 建议（无则 null）
- 标 taxonomy_extension: true

## 结构化过滤条件 structured_filters
字段（全部可选，无则设 null）：
- languages: {languages_enum}
- tools_used: {tools_enum}
- outcome_transition: {outcome_enum}
- min_turns: 整数 ≥1
- has_verification_step: 布尔值

## 输出格式（严格 JSON 数组）
对每个子问题输出：
```json
{{
  "id": "p1",
  "target_capability": ["label1"],
  "trajectory_signal": "在轨迹中应匹配什么模式",
  "hyde_positive": ["假设正例片段1(200-500token)", "假设正例片段2"],
  "keywords": ["关键词1", "关键词2"],
  "structured_filters": {{"languages": ["python"], "outcome_transition": ["failed→success"], ...}},
  "confidence": 0.0-1.0,
  "route": "pass" 或 "drop",
  "drop_reason": "ambiguous|not_applicable|label_diverged|other（仅 route==drop 时填）"
}}
```

## 置信度评估维度
- 子问题是否有歧义（高权重）
- 能力标签是否唯一命中（中）
- trajectory_signal 是否可操作（中）
- HyDE 变体是否一致（低）

## 分流规则
- confidence ≥ 0.8 → route: "pass"
- confidence < 0.8 → route: "drop"，并标注 drop_reason

只输出 JSON 数组，不要其他解释。"""


def build_call2_messages(sub_problems: list[dict], taxonomy: Taxonomy) -> list[dict]:
    from module0.schema import LANGUAGES, TOOLS_USED, OUTCOME_TRANSITIONS

    format_kwargs = {
        "languages_enum": sorted(LANGUAGES),
        "tools_enum": sorted(TOOLS_USED),
        "outcome_enum": sorted(OUTCOME_TRANSITIONS),
    }

    if taxonomy.is_empty:
        system = _CALL2_SYSTEM_EMPTY_TAXONOMY.format(**format_kwargs)
    else:
        system = _CALL2_SYSTEM_WITH_TAXONOMY.format(
            taxonomy_injection=taxonomy.to_prompt_text(),
            **format_kwargs,
        )

    user_content = "请对以下子问题逐一分析：\n\n"
    for sp in sub_problems:
        user_content += f"- id: {sp['id']}\n  raw_text: {sp['raw_text']}\n  failure_summary: {sp['failure_summary']}\n\n"

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content},
    ]


# ── Call 3: 歧义澄清 (Part 6.5) ──

_CALL3_SYSTEM = """你是一个语义消歧专家。对每个被标记为"ambiguous"的子问题：

## 任务
列出该子问题所有合理的互斥解释，每条解释产出一条独立的被消歧新子问题。

## 约束
- 每条原始子问题最多拆 4 条消歧解释
- 新 id 格式：原 id + 字母后缀（如 p12 → p12a, p12b, p12c）
- 每条消歧子问题必须包含 failure_summary（重新描述，清晰无歧义）

## 输出格式（严格 JSON 数组）
```json
[
  {
    "original_id": "p12",
    "clarified": [
      {"id": "p12a", "raw_text": "原始文本", "failure_summary": "消歧后的明确描述"},
      {"id": "p12b", "raw_text": "原始文本", "failure_summary": "另一种消歧描述"}
    ]
  }
]
```

只输出 JSON，不要其他解释。"""


def build_call3_messages(ambiguous_problems: list[dict]) -> list[dict]:
    user_content = "以下子问题因语义歧义被标记为 drop。请消歧：\n\n"
    for sp in ambiguous_problems:
        user_content += f"- id: {sp['id']}\n  raw_text: {sp['raw_text']}\n  failure_summary: {sp['failure_summary']}\n"
        if sp.get("target_capability"):
            user_content += f"  已有标签猜测: {sp['target_capability']}\n"
        user_content += "\n"

    return [
        {"role": "system", "content": _CALL3_SYSTEM},
        {"role": "user", "content": user_content},
    ]


# ── Call 2': 消歧子问题重评 ──

_CALL2_PRIME_EXTRA = """

## 额外约束（Call 2' 专用）
- 这些是经过消歧的子问题，不再输出 drop_reason
- 不再触发任何进一步澄清
- confidence < 0.8 的直接丢弃，不再进入后续流程
- 输出同 Call 2 格式，但 route 只可能是 "pass"（confidence ≥ 0.8 时）或省略"""


def build_call2_prime_messages(clarified_sub_problems: list[dict], taxonomy: Taxonomy) -> list[dict]:
    """Same as Call 2 but with additional constraints for no further disambiguation."""
    base_msgs = build_call2_messages(clarified_sub_problems, taxonomy)
    # Append the extra constraint to system prompt
    base_msgs[0]["content"] += _CALL2_PRIME_EXTRA
    return base_msgs
```

注意：prompt 的中文措辞/格式可能需微调才能让真实 LLM 稳定输出。但结构性要素（必须包含的字段名、枚举值、JSON schema 指令）必须严格保留。实现时逐字核对 spec 各 Part 的要求。

- [ ] **Step 4: 运行测试，确认全过**

```bash
pytest tests/module0/test_prompts.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/module0/prompts.py tests/module0/test_prompts.py
git commit -m "feat(module0): prompts.py LLM prompt templates for Call 1/2/3/2'"
```

---

## Task 5: parsing.py — LLM 输出解析 + schema 校验

**Files:**
- Create: `src/module0/parsing.py`
- Test: `tests/module0/test_parsing.py`

从 LLM 原始文本中提取 JSON，校验 schema，返回结构化对象。容错处理：markdown 代码块包裹、多余文本、部分字段缺失。

- [ ] **Step 1: 写失败测试** `tests/module0/test_parsing.py`

```python
import pytest
from module0.parsing import (
    parse_call1_response,
    parse_call2_response,
    parse_call3_response,
    ParseError,
)


def test_parse_call1_success():
    raw = '{"sub_problems": [{"id": "p1", "raw_text": "写入py文件有语法错误", "failure_summary": "写入 py 文件语法错误"}]}'
    result = parse_call1_response(raw)
    assert len(result) == 1
    assert result[0]["id"] == "p1"


def test_parse_call1_with_markdown_fence():
    raw = '```json\n{"sub_problems": [{"id": "p1", "raw_text": "test", "failure_summary": "test"}]}\n```'
    result = parse_call1_response(raw)
    assert len(result) == 1


def test_parse_call1_invalid_json():
    with pytest.raises(ParseError):
        parse_call1_response("not json at all")


def test_parse_call1_missing_field():
    raw = '{"sub_problems": [{"id": "p1"}]}'  # missing raw_text, failure_summary
    with pytest.raises(ParseError, match="raw_text|failure_summary"):
        parse_call1_response(raw)


def test_parse_call2_success():
    raw = '''[{
        "id": "p1",
        "target_capability": ["valid_syntax_in_toolcall"],
        "trajectory_signal": "observation 含 SyntaxError",
        "hyde_positive": ["hyp1 text here over 20 chars for validity", "hyp2 text here over 20 chars"],
        "keywords": ["SyntaxError", "python"],
        "structured_filters": {"languages": ["python"], "outcome_transition": ["failed→success"]},
        "confidence": 0.92,
        "route": "pass"
    }]'''
    result = parse_call2_response(raw)
    assert len(result) == 1
    assert result[0]["confidence"] == 0.92
    assert result[0]["route"] == "pass"


def test_parse_call2_with_drop():
    raw = '''[{
        "id": "p5",
        "target_capability": ["x"],
        "trajectory_signal": "s",
        "hyde_positive": ["h1 padding text", "h2 padding text"],
        "keywords": ["k"],
        "structured_filters": {},
        "confidence": 0.3,
        "route": "drop",
        "drop_reason": "not_applicable"
    }]'''
    result = parse_call2_response(raw)
    assert result[0]["route"] == "drop"
    assert result[0]["drop_reason"] == "not_applicable"


def test_parse_call3_success():
    raw = '''[{
        "original_id": "p12",
        "clarified": [
            {"id": "p12a", "raw_text": "text", "failure_summary": "被 max_turns 截断"},
            {"id": "p12b", "raw_text": "text", "failure_summary": "主动收尾误判"}
        ]
    }]'''
    result = parse_call3_response(raw)
    assert len(result) == 1
    assert result[0]["original_id"] == "p12"
    assert len(result[0]["clarified"]) == 2


def test_parse_call3_max_4_clarified():
    """Spec says max K=4 clarified per original problem."""
    raw = '''[{
        "original_id": "p1",
        "clarified": [
            {"id": "p1a", "raw_text": "t", "failure_summary": "s1"},
            {"id": "p1b", "raw_text": "t", "failure_summary": "s2"},
            {"id": "p1c", "raw_text": "t", "failure_summary": "s3"},
            {"id": "p1d", "raw_text": "t", "failure_summary": "s4"},
            {"id": "p1e", "raw_text": "t", "failure_summary": "s5"}
        ]
    }]'''
    result = parse_call3_response(raw)
    # Should truncate to 4
    assert len(result[0]["clarified"]) <= 4
```

- [ ] **Step 2: 运行测试，确认失败**

- [ ] **Step 3: 实现 parsing.py**

```python
"""Parse LLM raw text responses into structured dicts.

Handles: markdown fence stripping, JSON extraction, field validation,
and graceful error reporting. Does NOT do schema-level validation
(SubProblem/StructuredFilters) — that's compiler.py's job when assembling
the final ProblemSpec.
"""

from __future__ import annotations

import json
import re

__all__ = [
    "parse_call1_response",
    "parse_call2_response",
    "parse_call3_response",
    "ParseError",
]

_MAX_CLARIFIED_PER_PROBLEM = 4  # Spec Part 6.5: K=4


class ParseError(ValueError):
    """Raised when LLM output cannot be parsed into expected structure."""
    pass


def _extract_json(text: str) -> str:
    """Strip markdown fences and surrounding text to find JSON."""
    if not text:
        raise ParseError("Empty response")
    # Try stripping markdown code fence
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Try finding first { or [
    for i, c in enumerate(text):
        if c in ('{', '['):
            # Find matching close
            depth = 0
            open_c = c
            close_c = '}' if c == '{' else ']'
            for j in range(i, len(text)):
                if text[j] == open_c:
                    depth += 1
                elif text[j] == close_c:
                    depth -= 1
                    if depth == 0:
                        return text[i:j+1]
            break
    raise ParseError(f"No valid JSON found in response: {text[:200]}...")


def parse_call1_response(raw: str) -> list[dict]:
    """Parse Call 1 output: {sub_problems: [{id, raw_text, failure_summary}]}."""
    json_str = _extract_json(raw)
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ParseError(f"Invalid JSON: {e}") from e

    if isinstance(data, dict):
        items = data.get("sub_problems", [])
    elif isinstance(data, list):
        items = data
    else:
        raise ParseError(f"Expected dict or list, got {type(data).__name__}")

    for item in items:
        for field in ("id", "raw_text", "failure_summary"):
            if field not in item:
                raise ParseError(f"Missing required field '{field}' in Call 1 output")
    return items


def parse_call2_response(raw: str) -> list[dict]:
    """Parse Call 2/2' output: [{id, target_capability, ..., confidence, route}]."""
    json_str = _extract_json(raw)
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ParseError(f"Invalid JSON: {e}") from e

    if isinstance(data, dict) and "results" in data:
        items = data["results"]
    elif isinstance(data, list):
        items = data
    else:
        raise ParseError(f"Expected list, got {type(data).__name__}")

    required = ("id", "target_capability", "trajectory_signal", "hyde_positive",
                "keywords", "structured_filters", "confidence", "route")
    for item in items:
        missing = [f for f in required if f not in item]
        if missing:
            raise ParseError(f"Missing fields in Call 2 output: {missing}")
    return items


def parse_call3_response(raw: str) -> list[dict]:
    """Parse Call 3 output: [{original_id, clarified: [{id, raw_text, failure_summary}]}]."""
    json_str = _extract_json(raw)
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ParseError(f"Invalid JSON: {e}") from e

    if not isinstance(data, list):
        raise ParseError(f"Expected list, got {type(data).__name__}")

    for item in data:
        if "original_id" not in item or "clarified" not in item:
            raise ParseError("Missing 'original_id' or 'clarified' in Call 3 output")
        # Enforce max K
        if len(item["clarified"]) > _MAX_CLARIFIED_PER_PROBLEM:
            item["clarified"] = item["clarified"][:_MAX_CLARIFIED_PER_PROBLEM]
        for c in item["clarified"]:
            for field in ("id", "raw_text", "failure_summary"):
                if field not in c:
                    raise ParseError(f"Missing '{field}' in clarified sub-problem")
    return data
```

- [ ] **Step 4: 运行测试，确认全过**

```bash
pytest tests/module0/test_parsing.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/module0/parsing.py tests/module0/test_parsing.py
git commit -m "feat(module0): parsing.py LLM response JSON extraction + validation"
```

---

## Task 6: compiler.py — QueryCompiler 编排核心

**Files:**
- Create: `src/module0/compiler.py`
- Modify: `src/module0/__init__.py`（导出 QueryCompiler）
- Test: `tests/module0/test_compiler.py`

这是核心编排：串联 Call 1 → Call 2 →（条件）Call 3 → Call 2'，组装 Problem Spec。通过 `llm_gateway.LLMGateway.call()` 发起调用。用 mock gateway 测试编排逻辑（真实 LLM 联调在 Task 7）。

**关键约束（spec §3）：**
- 调用上限恒为 4：Call 3 + Call 2' 至多一次，不递归
- route==pass 进 Problem Spec；route==drop 且 reason!=ambiguous 直接丢弃；route==drop 且 reason==ambiguous 进 Part 6.5
- 澄清产出的子问题：origin="clarified"，parent_id 指向原始 id，重评后不再输出 drop_reason
- Call 2' 后 confidence≥0.8 进 Spec，<0.8 最终丢弃
- 对每条 pass 子问题的 hyde_positive 做 embedding（可选，见下）

- [ ] **Step 1: 写失败测试** `tests/module0/test_compiler.py`

```python
import pytest
from unittest.mock import AsyncMock
from module0.compiler import QueryCompiler
from module0.taxonomy import Taxonomy


class FakeGateway:
    """Mock gateway returning scripted responses in call order."""
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    async def call(self, messages, model, **kwargs):
        self.calls.append((messages, model))
        resp = self._responses.pop(0)
        return resp, {"status_code": 200, "prompt_tokens": 10, "completion_tokens": 5}


@pytest.fixture
def taxonomy(taxonomy_v0_path):
    return Taxonomy.load(taxonomy_v0_path)


async def test_two_call_happy_path(taxonomy):
    """No ambiguous drops → exactly 2 calls (Call 1 + Call 2)."""
    call1_resp = '{"sub_problems": [{"id": "p1", "raw_text": "写入py文件有语法错误", "failure_summary": "写入 py 语法错误"}]}'
    call2_resp = '''[{
        "id": "p1",
        "target_capability": ["valid_syntax_in_toolcall"],
        "trajectory_signal": "observation 含 SyntaxError",
        "hyde_positive": ["正例片段一，超过二十字符的假设轨迹", "正例片段二，超过二十字符的假设轨迹"],
        "keywords": ["SyntaxError", "python"],
        "structured_filters": {"languages": ["python"], "outcome_transition": ["failed→success"]},
        "confidence": 0.92,
        "route": "pass"
    }]'''
    gw = FakeGateway([call1_resp, call2_resp])
    compiler = QueryCompiler(gateway=gw, taxonomy=taxonomy, model="test-model", embedding_model=None)

    spec = await compiler.compile("写入py文件有语法错误")

    assert len(gw.calls) == 2  # only Call 1 + Call 2
    assert spec.domain == "agentic_swe"
    assert len(spec.sub_problems) == 1
    assert spec.sub_problems[0].id == "p1"
    assert spec.sub_problems[0].origin == "original"
    assert spec.sub_problems[0].parent_id is None


async def test_drop_non_ambiguous_excluded(taxonomy):
    """route=drop, reason!=ambiguous → excluded, no Call 3."""
    call1_resp = '{"sub_problems": [{"id": "p1", "raw_text": "前端样式问题", "failure_summary": "前端视觉问题"}]}'
    call2_resp = '''[{
        "id": "p1",
        "target_capability": ["x"],
        "trajectory_signal": "s",
        "hyde_positive": ["hyp padding one text", "hyp padding two text"],
        "keywords": ["k"],
        "structured_filters": {},
        "confidence": 0.3,
        "route": "drop",
        "drop_reason": "not_applicable"
    }]'''
    gw = FakeGateway([call1_resp, call2_resp])
    compiler = QueryCompiler(gateway=gw, taxonomy=taxonomy, model="test-model", embedding_model=None)

    spec = await compiler.compile("前端样式问题")

    assert len(gw.calls) == 2  # no Call 3
    assert len(spec.sub_problems) == 0  # dropped, not in spec


async def test_ambiguous_triggers_clarification(taxonomy):
    """route=drop, reason=ambiguous → Call 3 + Call 2' (4 calls total)."""
    call1_resp = '{"sub_problems": [{"id": "p12", "raw_text": "解题过程中途停止", "failure_summary": "多步任务中途停止"}]}'
    call2_resp = '''[{
        "id": "p12",
        "target_capability": ["persist_through_truncation"],
        "trajectory_signal": "s",
        "hyde_positive": ["hyp padding one text", "hyp padding two text"],
        "keywords": ["k"],
        "structured_filters": {},
        "confidence": 0.70,
        "route": "drop",
        "drop_reason": "ambiguous"
    }]'''
    call3_resp = '''[{
        "original_id": "p12",
        "clarified": [
            {"id": "p12a", "raw_text": "解题过程中途停止", "failure_summary": "被 max_turns 截断而未完成"},
            {"id": "p12b", "raw_text": "解题过程中途停止", "failure_summary": "主动收尾误判为完成"}
        ]
    }]'''
    call2prime_resp = '''[
        {"id": "p12a", "target_capability": ["persist_through_truncation"], "trajectory_signal": "末轮命中 max_turns",
         "hyde_positive": ["hyp padding one text", "hyp padding two text"], "keywords": ["max_turns"],
         "structured_filters": {"min_turns": 5}, "confidence": 0.84, "route": "pass"},
        {"id": "p12b", "target_capability": ["x"], "trajectory_signal": "s",
         "hyde_positive": ["hyp padding one text", "hyp padding two text"], "keywords": ["k"],
         "structured_filters": {}, "confidence": 0.60, "route": "pass"}
    ]'''
    gw = FakeGateway([call1_resp, call2_resp, call3_resp, call2prime_resp])
    compiler = QueryCompiler(gateway=gw, taxonomy=taxonomy, model="test-model", embedding_model=None)

    spec = await compiler.compile("解题过程中途停止")

    assert len(gw.calls) == 4  # Call 1 + Call 2 + Call 3 + Call 2'
    # p12a (0.84) passes, p12b (0.60) dropped
    assert len(spec.sub_problems) == 1
    assert spec.sub_problems[0].id == "p12a"
    assert spec.sub_problems[0].origin == "clarified"
    assert spec.sub_problems[0].parent_id == "p12"


async def test_no_recursion_after_clarification(taxonomy):
    """Call 2' output never triggers another Call 3, even if it says ambiguous."""
    call1_resp = '{"sub_problems": [{"id": "p1", "raw_text": "模糊问题", "failure_summary": "模糊"}]}'
    call2_resp = '[{"id": "p1", "target_capability": ["x"], "trajectory_signal": "s", "hyde_positive": ["padding one text here", "padding two text here"], "keywords": ["k"], "structured_filters": {}, "confidence": 0.5, "route": "drop", "drop_reason": "ambiguous"}]'
    call3_resp = '[{"original_id": "p1", "clarified": [{"id": "p1a", "raw_text": "t", "failure_summary": "clarified"}]}]'
    # Call 2' returns low confidence — must be dropped, NOT re-clarified
    call2prime_resp = '[{"id": "p1a", "target_capability": ["x"], "trajectory_signal": "s", "hyde_positive": ["padding one text here", "padding two text here"], "keywords": ["k"], "structured_filters": {}, "confidence": 0.4, "route": "pass"}]'
    gw = FakeGateway([call1_resp, call2_resp, call3_resp, call2prime_resp])
    compiler = QueryCompiler(gateway=gw, taxonomy=taxonomy, model="test-model", embedding_model=None)

    spec = await compiler.compile("模糊问题")

    assert len(gw.calls) == 4  # exactly 4, no 5th call
    assert len(spec.sub_problems) == 0  # p1a dropped (0.4 < 0.8)


async def test_dropped_audit_records(taxonomy):
    """Dropped sub-problems are accessible for audit (not in sub_problems[])."""
    call1_resp = '{"sub_problems": [{"id": "p1", "raw_text": "环境问题", "failure_summary": "基础设施问题"}]}'
    call2_resp = '[{"id": "p1", "target_capability": ["x"], "trajectory_signal": "s", "hyde_positive": ["padding one text here", "padding two text here"], "keywords": ["k"], "structured_filters": {}, "confidence": 0.3, "route": "drop", "drop_reason": "not_applicable"}]'
    gw = FakeGateway([call1_resp, call2_resp])
    compiler = QueryCompiler(gateway=gw, taxonomy=taxonomy, model="test-model", embedding_model=None)

    spec = await compiler.compile("环境问题")
    dropped = compiler.dropped_records

    assert len(spec.sub_problems) == 0
    assert len(dropped) == 1
    assert dropped[0].drop_reason == "not_applicable"
```

- [ ] **Step 2: 运行测试，确认失败**

- [ ] **Step 3: 实现 compiler.py**

```python
"""QueryCompiler — orchestrates Call 1/2/3/2' into a Problem Spec.

Flow (spec §3.2):
  Call 1 (decompose) → Call 2 (label + self-eval)
    ├── route==pass → Problem Spec
    ├── route==drop, reason!=ambiguous → dropped (audit)
    └── route==drop, reason==ambiguous → Call 3 (clarify) → Call 2' (re-eval)
          ├── confidence >= 0.8 → Problem Spec (origin=clarified)
          └── confidence < 0.8 → dropped (final)

Hard invariant: at most 4 LLM calls, Call 3+2' at most once, no recursion.
"""

from __future__ import annotations

from module0.parsing import (
    parse_call1_response,
    parse_call2_response,
    parse_call3_response,
    ParseError,
)
from module0.prompts import (
    build_call1_messages,
    build_call2_messages,
    build_call3_messages,
    build_call2_prime_messages,
)
from module0.schema import (
    ProblemSpec,
    SubProblem,
    DroppedSubProblem,
    StructuredFilters,
)
from module0.taxonomy import Taxonomy

__all__ = ["QueryCompiler"]

_PASS_THRESHOLD = 0.8  # spec Part 6 分流规则


class QueryCompiler:
    def __init__(self, *, gateway, taxonomy: Taxonomy, model: str,
                 embedding_model=None, max_tokens: int = 4000):
        self._gateway = gateway
        self._taxonomy = taxonomy
        self._model = model
        self._embedding_model = embedding_model
        self._max_tokens = max_tokens
        self.dropped_records: list[DroppedSubProblem] = []

    async def compile(self, raw_input: str) -> ProblemSpec:
        self.dropped_records = []

        # ── Call 1: decompose ──
        c1_text, _ = await self._gateway.call(
            build_call1_messages(raw_input), self._model, max_tokens=self._max_tokens,
        )
        sub_problems_raw = parse_call1_response(c1_text)

        # ── Call 2: label + self-eval ──
        c2_text, _ = await self._gateway.call(
            build_call2_messages(sub_problems_raw, self._taxonomy),
            self._model, max_tokens=self._max_tokens,
        )
        c2_results = parse_call2_response(c2_text)

        # Index raw sub-problems by id for merging
        raw_by_id = {sp["id"]: sp for sp in sub_problems_raw}

        passed: list[SubProblem] = []
        ambiguous: list[dict] = []

        for result in c2_results:
            sp_id = result["id"]
            raw = raw_by_id.get(sp_id, {})
            route = result.get("route")
            if route == "pass" and result.get("confidence", 0) >= _PASS_THRESHOLD:
                passed.append(self._build_sub_problem(result, raw, origin="original", parent_id=None))
            else:
                # drop
                reason = result.get("drop_reason", "other")
                self.dropped_records.append(self._build_dropped(result, raw, reason))
                if reason == "ambiguous":
                    ambiguous.append({
                        "id": sp_id,
                        "raw_text": raw.get("raw_text", ""),
                        "failure_summary": raw.get("failure_summary", ""),
                        "target_capability": result.get("target_capability", []),
                    })

        # ── Call 3 + Call 2' (conditional, at most once, no recursion) ──
        if ambiguous:
            passed.extend(await self._clarify_and_reeval(ambiguous))

        # ── Embedding (optional) ──
        if self._embedding_model is not None:
            for sp in passed:
                # embed hyde_positive segments (stored separately or attached)
                # Plan A: compute but the vectors go to the retrieval layer (Plan B/module 1)
                # Here we just validate they can be computed without error.
                self._embedding_model.embed_batch(sp.hyde_positive)

        return ProblemSpec(raw_input=raw_input, domain="agentic_swe", sub_problems=passed)

    async def _clarify_and_reeval(self, ambiguous: list[dict]) -> list[SubProblem]:
        """Call 3 (disambiguate) + Call 2' (re-eval). At most once, no recursion."""
        # ── Call 3 ──
        c3_text, _ = await self._gateway.call(
            build_call3_messages(ambiguous), self._model, max_tokens=self._max_tokens,
        )
        c3_results = parse_call3_response(c3_text)

        # Flatten clarified sub-problems, track parent_id
        clarified_raw: list[dict] = []
        parent_map: dict[str, str] = {}
        for group in c3_results:
            original_id = group["original_id"]
            for c in group["clarified"]:
                clarified_raw.append(c)
                parent_map[c["id"]] = original_id

        if not clarified_raw:
            return []

        # ── Call 2' ──
        c2p_text, _ = await self._gateway.call(
            build_call2_prime_messages(clarified_raw, self._taxonomy),
            self._model, max_tokens=self._max_tokens,
        )
        c2p_results = parse_call2_response(c2p_text)

        clarified_by_id = {c["id"]: c for c in clarified_raw}
        recovered: list[SubProblem] = []
        for result in c2p_results:
            sp_id = result["id"]
            raw = clarified_by_id.get(sp_id, {})
            # Call 2' 后：confidence >= 0.8 进 spec，否则最终丢弃（不再触发澄清）
            if result.get("confidence", 0) >= _PASS_THRESHOLD:
                recovered.append(self._build_sub_problem(
                    result, raw, origin="clarified", parent_id=parent_map.get(sp_id),
                ))
            # else: final drop, no audit record needed (or add one — spec doesn't require)
        return recovered

    def _build_sub_problem(self, result: dict, raw: dict, *, origin: str,
                           parent_id: str | None) -> SubProblem:
        return SubProblem(
            id=result["id"],
            origin=origin,
            parent_id=parent_id,
            raw_text=raw.get("raw_text", ""),
            failure_summary=raw.get("failure_summary", ""),
            target_capability=result["target_capability"],
            trajectory_signal=result["trajectory_signal"],
            hyde_positive=result["hyde_positive"],
            keywords=result["keywords"],
            structured_filters=self._build_filters(result.get("structured_filters")),
            confidence=result["confidence"],
            route="pass",
        )

    def _build_dropped(self, result: dict, raw: dict, reason: str) -> DroppedSubProblem:
        return DroppedSubProblem(
            id=result["id"],
            origin="original",
            parent_id=None,
            raw_text=raw.get("raw_text", ""),
            failure_summary=raw.get("failure_summary", ""),
            target_capability=result.get("target_capability", ["unknown"]),
            trajectory_signal=result.get("trajectory_signal", ""),
            hyde_positive=result.get("hyde_positive", ["", ""]),
            keywords=result.get("keywords", []),
            structured_filters=self._build_filters(result.get("structured_filters")),
            confidence=result.get("confidence", 0.0),
            route="drop",
            drop_reason=reason,
        )

    def _build_filters(self, d: dict | None) -> StructuredFilters:
        if not d:
            return StructuredFilters()
        return StructuredFilters(
            languages=d.get("languages"),
            tools_used=d.get("tools_used"),
            outcome_transition=d.get("outcome_transition"),
            min_turns=d.get("min_turns"),
            has_verification_step=d.get("has_verification_step"),
        )
```

注意 `_build_dropped` 里对 DroppedSubProblem 的构造：DroppedSubProblem 的 `__post_init__` 会校验 hyde_positive 无长度约束（DroppedSubProblem 没有继承 SubProblem 的校验）。确认 schema.py 中 DroppedSubProblem 不强制 hyde_positive 2-3 段（因为 drop 的可能字段不全）。若 schema 强制了，需在 schema.py 放宽 DroppedSubProblem 的校验。**实现时核对这一点。**

- [ ] **Step 4: 更新 __init__.py**

```python
"""Module 0 — Query Compiler."""

from module0.compiler import QueryCompiler
from module0.schema import ProblemSpec, SubProblem, StructuredFilters
from module0.taxonomy import Taxonomy

__all__ = ["QueryCompiler", "ProblemSpec", "SubProblem", "StructuredFilters", "Taxonomy"]
```

- [ ] **Step 5: 运行所有测试**

```bash
pytest tests/module0/ -v
```
预期全过（schema + taxonomy + prompts + parsing + compiler；embedding 需模型下载可能跳过）。

- [ ] **Step 6: Commit**

```bash
git add src/module0/compiler.py src/module0/__init__.py tests/module0/test_compiler.py
git commit -m "feat(module0): QueryCompiler orchestration (Call 1/2/3/2', 4-call cap)"
```

---

## Task 7: 联调 —— Module 0 → LLMGateway → LiteLLM

**Files:**
- Create: `tests/module0/test_integration.py`（标记 `@pytest.mark.integration`，默认跳过）
- Possibly modify: `src/llm_gateway/gateway.py` + `transport.py`（若联调暴露接口缺口）
- Create: `scripts/module0_smoke.py`（手动联调脚本）

这是本计划的核心价值：用真实 LLM 跑通 Module 0 完整链路，验证 gateway 对接。**预期会暴露 gateway 缺口**（见下）。

### 已知的 gateway 接口缺口（联调前预判）

探索发现 `gateway.call()` 目前**不支持 structured output / JSON mode**：payload 只有 model/messages/temperature/max_tokens。Module 0 全是 JSON 输出。两种应对：

- **方案 A（推荐，改动小）**：不动 gateway。Module 0 靠 prompt 里的"只输出 JSON"指令 + parsing.py 的容错提取（已实现 markdown 剥离 + JSON 定位）。LiteLLM 后端多数模型遵循指令良好。
- **方案 B（若方案 A 不稳定）**：给 gateway.call() + async_llm_call() 加 `response_format: dict | None = None` 参数，透传到 payload（OpenAI `{"type": "json_object"}`）。这是对 gateway 的增强，需回到 gateway 分支加测试。

**决策留到联调时**：先按方案 A 跑，若 JSON 解析失败率高再上方案 B。

- [ ] **Step 1: 写集成测试** `tests/module0/test_integration.py`

```python
import os
import pytest

from llm_gateway import LLMGateway, GatewayConfig
from module0 import QueryCompiler, Taxonomy
from module0.schema import validate_problem_spec

# Skip unless explicitly enabled (needs running LiteLLM)
pytestmark = pytest.mark.skipif(
    not os.environ.get("MODULE0_INTEGRATION"),
    reason="set MODULE0_INTEGRATION=1 and a running LiteLLM to run",
)

MODEL = os.environ.get("MODULE0_TEST_MODEL", "gpt-4o-mini")


@pytest.fixture
def taxonomy():
    import pathlib
    p = pathlib.Path(__file__).parent.parent.parent / "fixtures" / "taxonomy_v0.json"
    return Taxonomy.load(p)


async def test_end_to_end_simple(taxonomy):
    """Real LLM: single clear problem → 2 calls → valid Problem Spec."""
    config = GatewayConfig(
        litellm_base=os.environ.get("LITELLM_BASE", "http://localhost:4000/v1"),
        litellm_key=os.environ.get("LITELLM_KEY", ""),
        concurrency=5,
        transport_stuck_seconds=0,
    )
    async with LLMGateway(config) as gw:
        compiler = QueryCompiler(gateway=gw, taxonomy=taxonomy, model=MODEL, embedding_model=None)
        spec = await compiler.compile("写入py文件经常有语法错误")

        # Validate the produced spec round-trips through contract validation
        assert spec.domain == "agentic_swe"
        assert len(spec.sub_problems) >= 1
        for sp in spec.sub_problems:
            assert 1 <= len(sp.target_capability) <= 3
            assert 2 <= len(sp.hyde_positive) <= 3
            assert sp.confidence >= 0.8
            assert sp.route == "pass"


async def test_end_to_end_ambiguous(taxonomy):
    """Real LLM: the p12 '中途停止' benchmark case → clarification triggers."""
    config = GatewayConfig(
        litellm_base=os.environ.get("LITELLM_BASE", "http://localhost:4000/v1"),
        litellm_key=os.environ.get("LITELLM_KEY", ""),
        transport_stuck_seconds=0,
    )
    async with LLMGateway(config) as gw:
        compiler = QueryCompiler(gateway=gw, taxonomy=taxonomy, model=MODEL, embedding_model=None)
        spec = await compiler.compile("解题过程中途停止")
        # Whatever the outcome, must not exceed 4 calls and must be valid
        # (clarified sub-problems, if any, carry origin=clarified + parent_id)
        for sp in spec.sub_problems:
            if sp.origin == "clarified":
                assert sp.parent_id is not None


async def test_embedding_integration(taxonomy):
    """Real LLM + real embedding model end-to-end."""
    from module0.embedding import EmbeddingModel
    config = GatewayConfig(
        litellm_base=os.environ.get("LITELLM_BASE", "http://localhost:4000/v1"),
        litellm_key=os.environ.get("LITELLM_KEY", ""),
        transport_stuck_seconds=0,
    )
    emb = EmbeddingModel()
    async with LLMGateway(config) as gw:
        compiler = QueryCompiler(gateway=gw, taxonomy=taxonomy, model=MODEL, embedding_model=emb)
        spec = await compiler.compile("写入py文件经常有语法错误")
        assert spec is not None
```

- [ ] **Step 2: 写手动 smoke 脚本** `scripts/module0_smoke.py`

```python
"""Manual smoke test: Module 0 → LLMGateway → LiteLLM.

Usage:
    export LITELLM_BASE=http://localhost:4000/v1
    export LITELLM_KEY=sk-...
    python scripts/module0_smoke.py "写入py文件有语法错误，工具调用结构经常出错"
"""

import asyncio
import json
import os
import sys
import pathlib

from llm_gateway import LLMGateway, GatewayConfig
from module0 import QueryCompiler, Taxonomy


async def main():
    raw_input = sys.argv[1] if len(sys.argv) > 1 else "写入py文件有语法错误"
    model = os.environ.get("MODULE0_TEST_MODEL", "gpt-4o-mini")

    taxonomy_path = pathlib.Path(__file__).parent.parent / "fixtures" / "taxonomy_v0.json"
    taxonomy = Taxonomy.load(taxonomy_path)

    config = GatewayConfig(
        litellm_base=os.environ.get("LITELLM_BASE", "http://localhost:4000/v1"),
        litellm_key=os.environ.get("LITELLM_KEY", ""),
        transport_stuck_seconds=0,
    )
    async with LLMGateway(config) as gw:
        compiler = QueryCompiler(gateway=gw, taxonomy=taxonomy, model=model, embedding_model=None)
        spec = await compiler.compile(raw_input)

        print("=== Problem Spec ===")
        print(json.dumps({
            "raw_input": spec.raw_input,
            "domain": spec.domain,
            "sub_problems": [
                {
                    "id": sp.id, "origin": sp.origin, "parent_id": sp.parent_id,
                    "failure_summary": sp.failure_summary,
                    "target_capability": sp.target_capability,
                    "confidence": sp.confidence,
                } for sp in spec.sub_problems
            ],
        }, ensure_ascii=False, indent=2))
        print(f"\n=== Dropped (audit): {len(compiler.dropped_records)} ===")
        for d in compiler.dropped_records:
            print(f"  {d.id}: {d.drop_reason} (conf={d.confidence})")
        print(f"\n=== Gateway stats: {gw.http_stats} ===")


if __name__ == "__main__":
    asyncio.run(main())
```

- [ ] **Step 3: 跑联调（需要你启动 LiteLLM）**

```bash
export LITELLM_BASE=http://localhost:4000/v1
export LITELLM_KEY=sk-...
export MODULE0_TEST_MODEL=<你的模型>

# 手动 smoke
python scripts/module0_smoke.py "写入py文件有语法错误，工具调用结构经常出错"

# 自动集成测试
MODULE0_INTEGRATION=1 pytest tests/module0/test_integration.py -v
```

- [ ] **Step 4: 记录联调结果 + 决策 response_format**

观察：
- JSON 解析成功率（若频繁 ParseError → 上方案 B 给 gateway 加 response_format）
- 4-call 上限是否被遵守
- p12 "中途停止" 是否正确触发澄清
- Problem Spec 是否通过契约校验

若需方案 B，切回 gateway 分支给 `call()` + `async_llm_call()` 加 `response_format` 参数（附测试），再回来重跑。

- [ ] **Step 5: Commit**

```bash
git add tests/module0/test_integration.py scripts/module0_smoke.py
git commit -m "test(module0): integration tests + smoke script for gateway联调"
```

---

## Self-Review Checklist

1. **契约合规**：Problem Spec schema（§1）/ StructuredFilters 枚举（§2）/ drop_reason 枚举 全部在 schema.py 冻结，validate_problem_spec 强校验。
2. **调用上限**：compiler.py 的 `_clarify_and_reeval` 只被调一次（非循环），Call 2' 输出不再走 drop_reason 分支 → 4-call 上限硬保证。test_no_recursion 验证。
3. **origin/parent_id**：original→parent_id=null；clarified→parent_id 指向原始 id。test_ambiguous 验证。
4. **占位符扫描**：无 TODO/TBD。prompt 措辞标注"实现时可微调"但结构要素锁定。
5. **类型一致性**：compiler 调用 gateway.call(messages, model, max_tokens=) 与已实现签名一致；返回 (text, usage) 解包正确。
6. **Embedding 可选**：embedding_model=None 时跳过（测试用），非阻塞编排。

---

## 依赖 Plan B 的部分（本计划显式不做）

- `hyde_positive` 的 embedding 向量最终去向（写入 Qdrant）——Plan B 的检索层负责。本计划只验证 embedding 能算出来。
- 新标签入库/去重/挂载——Plan B（taxonomy 演化）。本计划 taxonomy 只读。
- structured_filters 落到真实切片签名的召回——Plan B / 模块 1。

---

## Task 依赖图（执行顺序）

```
Task 0 (基础设施/fixtures)
   ├─→ Task 1 (schema)  ──┐
   ├─→ Task 3 (embedding) │  (独立，可并行)
   │                      ▼
   ├─→ Task 2 (taxonomy) ─→ Task 4 (prompts)
   │                       Task 5 (parsing) ←── Task 1
   │                              │
   └──────────────────────────────┴─→ Task 6 (compiler) ─→ Task 7 (联调)
```

- Task 1 / Task 3 可并行（都是叶子）
- Task 2 依赖 Task 1；Task 4 依赖 Task 2；Task 5 依赖 Task 1
- Task 6 依赖 1-5 全部；Task 7 依赖 Task 6 + 已实现的 llm_gateway
