# 模块 0 LLM 健壮性补强 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让 `QueryCompiler.compile()` 在 LLM 偶发返回空响应 / 坏 JSON / 缺字段时不崩溃——通过「每步重试 1 次 + 分层降级」。

**Architecture:** 抽一个 `_call_and_parse` 私有方法收敛「调用→检查空→解析→重试」；`compile()` 按步骤处理重试用尽：Call 1 硬失败抛公开 `CompileError`，Call 2/3/2' 降级为空。结构化 `robustness_report` 记录重试/降级。`parsing.py` 与 Problem Spec 契约不动。

**Tech Stack:** Python 3.11, pytest, pytest-asyncio。纯逻辑测试(FakeGateway,零真实 LLM/网络)。

**上游 spec:** `docs/superpowers/specs/2026-07-10-module0-llm-robustness-design.md`

---

## 前置说明(实现者必读)

- **运行测试用** `/Users/deng/开发/UXFlow/.venv/bin/python -m pytest ...`(系统 `python` 是 3.9,会 import 失败;venv 是 3.11)。
- 工作目录:`/Users/deng/开发/UXFlow`,当前分支 `main`(本计划应在新分支上实现——由执行方按 subagent-driven 流程建分支)。
- **FakeGateway 无需改结构**:现有 `tests/module0/test_compiler.py` 的 `FakeGateway` 按列表顺序 pop 返回响应,`None` 和坏字符串都能作为列表元素直接注入。重试会多消耗一次响应,所以坏响应测试只需在响应列表里多放对应条目。
- **关键:重试不改变成功路径的调用次数**。成功即返回,不重试。所以现有 happy-path 测试(脚本恰好条数)仍通过。
- `compiler.py` 当前**未导入** `ParseError`,只导入了 `parse_call1_response/parse_call2_response/parse_call3_response`。Task 2 需补充导入。
- 现有 `_build_sub_problem` 的单 item try/except 降级(`compiler.py:92-95` 主路径、`:152-157` Call 2' 路径)**保持不动**,本计划只加步骤级容错。

## 文件结构

- **改 `src/module0/compiler.py`**:新增异常类 `RetryableParseError`、`_StepFailed`、`CompileError`;新增 `_call_and_parse` 方法、`_robustness` 状态、`robustness_report` 属性;改 4 处调用点。
- **改 `src/module0/__init__.py`**:导出 `CompileError`。
- **改 `tests/module0/test_compiler.py`**:新增容错测试(§6 of spec)。
- **不动**:`parsing.py`、`prompts.py`、`schema.py`。

---

## Task 1: 新增异常类 + 导出 CompileError

**Files:**
- Modify: `src/module0/compiler.py`（顶部加异常类）
- Modify: `src/module0/__init__.py`（导出）
- Test: `tests/module0/test_compiler.py`

- [ ] **Step 1: 写失败测试**

在 `tests/module0/test_compiler.py` 末尾追加:

```python
def test_compile_error_is_importable_and_is_exception():
    from module0 import CompileError
    assert issubclass(CompileError, Exception)
```

- [ ] **Step 2: 运行确认失败**

Run: `.venv/bin/python -m pytest tests/module0/test_compiler.py::test_compile_error_is_importable_and_is_exception -v`
Expected: FAIL — `ImportError: cannot import name 'CompileError' from 'module0'`

- [ ] **Step 3: 加异常类**

在 `src/module0/compiler.py` 的 `__all__ = ["QueryCompiler"]` 行**下方**加入:

```python
__all__ = ["QueryCompiler", "CompileError"]

_PASS_THRESHOLD = 0.8  # spec Part 6 分流规则


class CompileError(Exception):
    """Raised when compilation cannot proceed (Call 1 failed after retries)."""


class RetryableParseError(Exception):
    """Internal signal: this step should be retried (empty response)."""


class _StepFailed(Exception):
    """Internal: a step exhausted its retries. Caught by compile() to degrade."""

    def __init__(self, step_name: str, last_error):
        self.step_name = step_name
        self.last_error = last_error
        super().__init__(f"{step_name} failed after retries: {last_error}")
```

注意:原文件已有 `__all__ = ["QueryCompiler"]` 和 `_PASS_THRESHOLD = 0.8` 两行(在 import 块之后)。用上面内容**替换**这两行所在位置(即把 `__all__` 改为含 `CompileError`,并在 `_PASS_THRESHOLD` 之后追加三个类)。不要重复定义 `_PASS_THRESHOLD`。

- [ ] **Step 4: 导出**

修改 `src/module0/__init__.py`,把 `from module0.compiler import QueryCompiler` 改为:

```python
from module0.compiler import QueryCompiler, CompileError
```

并把 `__all__` 改为:

```python
__all__ = ["QueryCompiler", "CompileError", "ProblemSpec", "SubProblem", "StructuredFilters", "Taxonomy"]
```

- [ ] **Step 5: 运行确认通过**

Run: `.venv/bin/python -m pytest tests/module0/test_compiler.py::test_compile_error_is_importable_and_is_exception -v`
Expected: PASS

- [ ] **Step 6: 回归——现有 compiler 测试不受影响**

Run: `.venv/bin/python -m pytest tests/module0/test_compiler.py -q`
Expected: PASS(全部现有测试 + 新测试)

- [ ] **Step 7: 提交**

```bash
git add src/module0/compiler.py src/module0/__init__.py tests/module0/test_compiler.py
git commit -m "feat(module0): add CompileError + internal retry/degrade exception types"
```

---

## Task 2: `_call_and_parse` helper + robustness 状态 + 接入 Call 1(硬失败)

**Files:**
- Modify: `src/module0/compiler.py`
- Test: `tests/module0/test_compiler.py`

- [ ] **Step 1: 写失败测试(两条:Call 1 重试成功、Call 1 硬失败抛 CompileError)**

在 `tests/module0/test_compiler.py` 末尾追加:

```python
_C2_OK = '''[{
    "id": "p1",
    "target_capability": ["valid_syntax_in_toolcall"],
    "trajectory_signal": "observation 含 SyntaxError",
    "hyde_positive": ["正例片段一,超过二十字符的假设轨迹", "正例片段二,超过二十字符的假设轨迹"],
    "keywords": ["SyntaxError", "python"],
    "structured_filters": {"languages": ["python"]},
    "confidence": 0.92,
    "route": "pass"
}]'''

_C1_OK = '{"sub_problems": [{"id": "p1", "raw_text": "写入py有语法错误", "failure_summary": "语法错误"}]}'


async def test_call1_empty_then_retry_succeeds(taxonomy):
    # Call 1 returns None first (empty), retry returns valid; then Call 2 ok.
    gw = FakeGateway([None, _C1_OK, _C2_OK])
    compiler = QueryCompiler(gateway=gw, taxonomy=taxonomy, model="test-model", embedding_model=None)
    spec = await compiler.compile("写入py文件有语法错误")
    assert len(spec.sub_problems) == 1
    assert compiler.robustness_report["retries"]["call1"] == 1


async def test_call1_hard_fail_raises_compile_error(taxonomy):
    from module0 import CompileError
    # Call 1 empty on both the initial attempt and the retry → give up.
    gw = FakeGateway([None, None])
    compiler = QueryCompiler(gateway=gw, taxonomy=taxonomy, model="test-model", embedding_model=None)
    with pytest.raises(CompileError):
        await compiler.compile("写入py文件有语法错误")
```

- [ ] **Step 2: 运行确认失败**

Run: `.venv/bin/python -m pytest tests/module0/test_compiler.py::test_call1_empty_then_retry_succeeds tests/module0/test_compiler.py::test_call1_hard_fail_raises_compile_error -v`
Expected: FAIL(当前 Call 1 裸解析,`None` 会抛 `ParseError` 而非重试;且无 `robustness_report`)

- [ ] **Step 3: 补 ParseError 导入**

在 `src/module0/compiler.py` 的 import 块中,把:

```python
from module0.parsing import (
    parse_call1_response,
    parse_call2_response,
    parse_call3_response,
)
```

改为:

```python
from module0.parsing import (
    ParseError,
    parse_call1_response,
    parse_call2_response,
    parse_call3_response,
)
```

- [ ] **Step 4: 加 `_robustness` 初始化 + `robustness_report` 属性 + `_call_and_parse` 方法**

在 `QueryCompiler.__init__` 末尾(现有 `self.hyde_embeddings = {}` 之后)加:

```python
        self._robustness = {
            "retries": {"call1": 0, "call2": 0, "call3": 0, "call2prime": 0},
            "degraded": [],
        }
```

在 `__init__` 之后、`compile` 之前加属性:

```python
    @property
    def robustness_report(self) -> dict:
        """Retry/degrade stats for the last compile() run."""
        return self._robustness
```

在类中(建议放 `compile` 之后、`_clarify_and_reeval` 之前)加方法:

```python
    async def _call_and_parse(self, build_messages_fn, parser, *, step_name, max_attempts=2):
        """Call the LLM and parse; retry once on empty/malformed response.

        Raises _StepFailed when all attempts are exhausted; the caller decides
        how to degrade. Does not mutate parser behavior — parsing stays strict.
        """
        last_err = None
        for attempt in range(max_attempts):
            content, _ = await self._gateway.call(
                build_messages_fn(), self._model, max_tokens=self._max_tokens
            )
            try:
                if not content or not content.strip():
                    raise RetryableParseError(f"{step_name}: empty response")
                return parser(content)
            except (ParseError, RetryableParseError) as e:
                last_err = e
                if attempt + 1 < max_attempts:
                    self._robustness["retries"][step_name] += 1
                continue
        raise _StepFailed(step_name, last_err)
```

- [ ] **Step 5: 在 `compile()` 开头重置 `_robustness`,并把 Call 1 接入 helper**

在 `compile()` 方法开头,把:

```python
        self.dropped_records = []
        self.hyde_embeddings = {}
```

改为:

```python
        self.dropped_records = []
        self.hyde_embeddings = {}
        self._robustness = {
            "retries": {"call1": 0, "call2": 0, "call3": 0, "call2prime": 0},
            "degraded": [],
        }
```

然后把 Call 1 的两行:

```python
        c1_text, _ = await self._gateway.call(
            build_call1_messages(raw_input), self._model, max_tokens=self._max_tokens,
        )
        sub_problems_raw = parse_call1_response(c1_text)
```

替换为:

```python
        try:
            sub_problems_raw = await self._call_and_parse(
                lambda: build_call1_messages(raw_input),
                parse_call1_response, step_name="call1",
            )
        except _StepFailed as e:
            raise CompileError(f"Call 1 failed after retries: {e}") from e
```

- [ ] **Step 6: 运行确认通过**

Run: `.venv/bin/python -m pytest tests/module0/test_compiler.py::test_call1_empty_then_retry_succeeds tests/module0/test_compiler.py::test_call1_hard_fail_raises_compile_error -v`
Expected: PASS(2 passed)

- [ ] **Step 7: 回归**

Run: `.venv/bin/python -m pytest tests/module0/test_compiler.py -q`
Expected: PASS(现有 happy-path 测试不受影响——成功路径无重试)

- [ ] **Step 8: 提交**

```bash
git add src/module0/compiler.py tests/module0/test_compiler.py
git commit -m "feat(module0): add _call_and_parse retry helper; wire Call 1 with CompileError"
```

---

## Task 3: 接入 Call 2(降级为空)

**Files:**
- Modify: `src/module0/compiler.py`
- Test: `tests/module0/test_compiler.py`

- [ ] **Step 1: 写失败测试(Call 2 空响应重试成功、Call 2 重试用尽降级为空 spec)**

在 `tests/module0/test_compiler.py` 末尾追加:

```python
async def test_call2_empty_then_retry_succeeds(taxonomy):
    # Call 1 ok; Call 2 empty then valid.
    gw = FakeGateway([_C1_OK, None, _C2_OK])
    compiler = QueryCompiler(gateway=gw, taxonomy=taxonomy, model="test-model", embedding_model=None)
    spec = await compiler.compile("写入py文件有语法错误")
    assert len(spec.sub_problems) == 1
    assert compiler.robustness_report["retries"]["call2"] == 1


async def test_call2_exhausted_degrades_to_empty_spec(taxonomy):
    # Call 1 ok; Call 2 empty on both attempts → degrade, empty spec, no crash.
    gw = FakeGateway([_C1_OK, None, None])
    compiler = QueryCompiler(gateway=gw, taxonomy=taxonomy, model="test-model", embedding_model=None)
    spec = await compiler.compile("写入py文件有语法错误")
    assert spec.sub_problems == []
    assert spec.domain == "agentic_swe"
    assert "call2" in compiler.robustness_report["degraded"]
```

- [ ] **Step 2: 运行确认失败**

Run: `.venv/bin/python -m pytest tests/module0/test_compiler.py::test_call2_empty_then_retry_succeeds tests/module0/test_compiler.py::test_call2_exhausted_degrades_to_empty_spec -v`
Expected: FAIL(当前 Call 2 裸解析,`None` 抛 `ParseError` 崩溃)

- [ ] **Step 3: 把 Call 2 接入 helper**

在 `compile()` 中,把 Call 2 的两行:

```python
        c2_text, _ = await self._gateway.call(
            build_call2_messages(sub_problems_raw, self._taxonomy),
            self._model, max_tokens=self._max_tokens,
        )
        c2_results = parse_call2_response(c2_text)
```

替换为:

```python
        try:
            c2_results = await self._call_and_parse(
                lambda: build_call2_messages(sub_problems_raw, self._taxonomy),
                parse_call2_response, step_name="call2",
            )
        except _StepFailed:
            c2_results = []
            self._robustness["degraded"].append("call2")
```

- [ ] **Step 4: 运行确认通过**

Run: `.venv/bin/python -m pytest tests/module0/test_compiler.py::test_call2_empty_then_retry_succeeds tests/module0/test_compiler.py::test_call2_exhausted_degrades_to_empty_spec -v`
Expected: PASS(2 passed)

- [ ] **Step 5: 回归**

Run: `.venv/bin/python -m pytest tests/module0/test_compiler.py -q`
Expected: PASS

- [ ] **Step 6: 提交**

```bash
git add src/module0/compiler.py tests/module0/test_compiler.py
git commit -m "feat(module0): wire Call 2 with degrade-to-empty on exhausted retries"
```

---

## Task 4: 接入 Call 3 + Call 2'(澄清路径降级,复现最初崩溃)

**Files:**
- Modify: `src/module0/compiler.py`
- Test: `tests/module0/test_compiler.py`

- [ ] **Step 1: 写失败测试(Call 2' 缺 route 重试成功、Call 2' 用尽降级不崩)**

在 `tests/module0/test_compiler.py` 末尾追加。这些测试触发澄清链路:Call 2 把子问题判为 `ambiguous`(route=drop, drop_reason=ambiguous)→ Call 3 澄清 → Call 2' 重评。

```python
# Call 2 that drops p1 as ambiguous (triggers Call 3 + Call 2').
_C2_AMBIGUOUS = '''[{
    "id": "p1",
    "target_capability": ["valid_syntax_in_toolcall"],
    "trajectory_signal": "含糊",
    "hyde_positive": ["片段一超过二十字符的假设正例轨迹", "片段二超过二十字符的假设正例轨迹"],
    "keywords": ["python"],
    "structured_filters": {"languages": ["python"]},
    "confidence": 0.3,
    "route": "drop",
    "drop_reason": "ambiguous"
}]'''

# Call 3 clarifies p1 into p1a.
_C3_OK = '[{"original_id": "p1", "clarified": [{"id": "p1a", "raw_text": "澄清后的问题", "failure_summary": "澄清"}]}]'

# Call 2' valid response recovering p1a.
_C2P_OK = '''[{
    "id": "p1a",
    "target_capability": ["valid_syntax_in_toolcall"],
    "trajectory_signal": "observation 含 SyntaxError",
    "hyde_positive": ["片段一超过二十字符的假设正例轨迹", "片段二超过二十字符的假设正例轨迹"],
    "keywords": ["SyntaxError"],
    "structured_filters": {"languages": ["python"]},
    "confidence": 0.9,
    "route": "pass"
}]'''

# Call 2' response missing the required "route" field (the field the live LLM dropped).
_C2P_MISSING_ROUTE = '''[{
    "id": "p1a",
    "target_capability": ["valid_syntax_in_toolcall"],
    "trajectory_signal": "observation 含 SyntaxError",
    "hyde_positive": ["片段一超过二十字符的假设正例轨迹", "片段二超过二十字符的假设正例轨迹"],
    "keywords": ["SyntaxError"],
    "structured_filters": {"languages": ["python"]},
    "confidence": 0.9
}]'''


async def test_call2prime_missing_route_then_retry_succeeds(taxonomy):
    # Call1 ok → Call2 ambiguous → Call3 ok → Call2' missing route, retry valid.
    gw = FakeGateway([_C1_OK, _C2_AMBIGUOUS, _C3_OK, _C2P_MISSING_ROUTE, _C2P_OK])
    compiler = QueryCompiler(gateway=gw, taxonomy=taxonomy, model="test-model", embedding_model=None)
    spec = await compiler.compile("解题过程中途停止")
    assert any(sp.id == "p1a" for sp in spec.sub_problems)
    assert compiler.robustness_report["retries"]["call2prime"] == 1


async def test_call2prime_exhausted_degrades_without_crash(taxonomy):
    # Call2' missing route on both attempts → degrade; no crash; p1a not recovered.
    gw = FakeGateway([_C1_OK, _C2_AMBIGUOUS, _C3_OK, _C2P_MISSING_ROUTE, _C2P_MISSING_ROUTE])
    compiler = QueryCompiler(gateway=gw, taxonomy=taxonomy, model="test-model", embedding_model=None)
    spec = await compiler.compile("解题过程中途停止")
    assert all(sp.id != "p1a" for sp in spec.sub_problems)
    assert "clarify" in compiler.robustness_report["degraded"]
```

- [ ] **Step 2: 运行确认失败**

Run: `.venv/bin/python -m pytest tests/module0/test_compiler.py::test_call2prime_missing_route_then_retry_succeeds tests/module0/test_compiler.py::test_call2prime_exhausted_degrades_without_crash -v`
Expected: FAIL(当前 `_clarify_and_reeval` 裸解析 Call 3/2',缺 route 抛 `ParseError` 崩溃)

- [ ] **Step 3: 把 Call 3 与 Call 2' 接入 helper**

在 `_clarify_and_reeval` 中,把 Call 3 的两行:

```python
        c3_text, _ = await self._gateway.call(
            build_call3_messages(ambiguous), self._model, max_tokens=self._max_tokens,
        )
        c3_results = parse_call3_response(c3_text)
```

替换为:

```python
        c3_results = await self._call_and_parse(
            lambda: build_call3_messages(ambiguous),
            parse_call3_response, step_name="call3",
        )
```

再把 Call 2' 的两行:

```python
        c2p_text, _ = await self._gateway.call(
            build_call2_prime_messages(clarified_raw, self._taxonomy),
            self._model, max_tokens=self._max_tokens,
        )
        c2p_results = parse_call2_response(c2p_text)
```

替换为:

```python
        c2p_results = await self._call_and_parse(
            lambda: build_call2_prime_messages(clarified_raw, self._taxonomy),
            parse_call2_response, step_name="call2prime",
        )
```

- [ ] **Step 4: 在 `compile()` 中给澄清调用加降级捕获**

在 `compile()` 中,把:

```python
        # ── Call 3 + Call 2' (conditional, at most once, no recursion) ──
        if ambiguous:
            passed.extend(await self._clarify_and_reeval(ambiguous))
```

替换为:

```python
        # ── Call 3 + Call 2' (conditional, at most once, no recursion) ──
        if ambiguous:
            try:
                passed.extend(await self._clarify_and_reeval(ambiguous))
            except _StepFailed:
                self._robustness["degraded"].append("clarify")
                # Clarify path failed: ambiguous items stay in dropped_records,
                # main flow continues (no recover, no crash).
```

- [ ] **Step 5: 运行确认通过**

Run: `.venv/bin/python -m pytest tests/module0/test_compiler.py::test_call2prime_missing_route_then_retry_succeeds tests/module0/test_compiler.py::test_call2prime_exhausted_degrades_without_crash -v`
Expected: PASS(2 passed)

- [ ] **Step 6: 回归**

Run: `.venv/bin/python -m pytest tests/module0/test_compiler.py -q`
Expected: PASS

- [ ] **Step 7: 提交**

```bash
git add src/module0/compiler.py tests/module0/test_compiler.py
git commit -m "fix(module0): degrade clarify path (Call 3/2') instead of crashing"
```

---

## Task 5: 横切测试——统一坏 JSON + happy-path 无重试回归

**Files:**
- Test: `tests/module0/test_compiler.py`

- [ ] **Step 1: 写测试(坏 JSON 与空响应统一重试;正常路径零重试零降级)**

在 `tests/module0/test_compiler.py` 末尾追加:

```python
async def test_bad_json_and_empty_both_retried(taxonomy):
    # Call 2 returns non-JSON garbage first, then valid → should retry & succeed.
    gw = FakeGateway([_C1_OK, "not json {{{", _C2_OK])
    compiler = QueryCompiler(gateway=gw, taxonomy=taxonomy, model="test-model", embedding_model=None)
    spec = await compiler.compile("写入py文件有语法错误")
    assert len(spec.sub_problems) == 1
    assert compiler.robustness_report["retries"]["call2"] == 1


async def test_happy_path_records_no_retries_no_degrade(taxonomy):
    gw = FakeGateway([_C1_OK, _C2_OK])
    compiler = QueryCompiler(gateway=gw, taxonomy=taxonomy, model="test-model", embedding_model=None)
    spec = await compiler.compile("写入py文件有语法错误")
    assert len(spec.sub_problems) == 1
    report = compiler.robustness_report
    assert report["retries"] == {"call1": 0, "call2": 0, "call3": 0, "call2prime": 0}
    assert report["degraded"] == []
```

- [ ] **Step 2: 运行确认通过(实现已在 Task 2-4 完成,这里是覆盖补强)**

Run: `.venv/bin/python -m pytest tests/module0/test_compiler.py::test_bad_json_and_empty_both_retried tests/module0/test_compiler.py::test_happy_path_records_no_retries_no_degrade -v`
Expected: PASS(2 passed)

> 若 `test_bad_json_and_empty_both_retried` 失败,检查 `_call_and_parse` 的 `except (ParseError, RetryableParseError)` 是否确实捕获了坏 JSON 抛出的 `ParseError`(应当捕获)。

- [ ] **Step 3: 全 module0 回归**

Run: `.venv/bin/python -m pytest tests/module0/ -m "not requires_model" -q`
Expected: PASS(全部纯逻辑测试绿;`requires_model` 的 live 测试被排除)

- [ ] **Step 4: 提交**

```bash
git add tests/module0/test_compiler.py
git commit -m "test(module0): cover unified retry for bad JSON + happy-path regression"
```

---

## Self-Review(已执行)

**1. Spec coverage** — 逐条对照 spec:
- §2 决策1 重试+降级 → Task 2(helper)+ Task 3/4(降级) ✓
- §2 决策2 每步重试1次(max_attempts=2) → Task 2 `_call_and_parse` 默认 ✓
- §2 决策3 分层降级(Call1抛/其它空) → Task 2(Call1→CompileError)、Task 3(Call2→空)、Task 4(clarify→空) ✓
- §2 决策4 空响应+坏JSON统一重试 → Task 2 helper 的 `except (ParseError, RetryableParseError)`;Task 5 坏JSON测试 ✓
- §2 决策5 结构化计数属性 → Task 2 `_robustness` + `robustness_report` ✓
- §3 `_call_and_parse` + callable build_messages_fn → Task 2 ✓
- §4.1/4.2/4.3 各步骤降级 → Task 2/3/4 ✓
- §5 可观测性重置 → Task 2 Step 5(compile 开头重置) ✓
- §6 测试表 7 条 → Task 2(2条)+Task3(2)+Task4(2)+Task5(2)=8 条(happy-path 回归额外拆了一条,覆盖 ✓)
- §7 文件改动(compiler/__init__/test) → 全覆盖;parsing/prompts/schema 不动 ✓

**2. Placeholder scan** — 无 TBD/TODO;每步含完整代码与确切命令。

**3. Type consistency** — `_call_and_parse(build_messages_fn, parser, *, step_name, max_attempts=2)` 签名在 Task 2 定义,Task 3/4 调用一致;`step_name` 取值 `call1/call2/call3/call2prime` 与 `_robustness["retries"]` 的键完全对应;`_StepFailed`/`CompileError`/`RetryableParseError` 在 Task 1 定义,后续引用一致;`degraded` 追加值 `"call2"`/`"clarify"` 与测试断言一致。

**一处实现者需注意的顺序依赖**:Task 2 引入 helper 并接 Call 1,但 Call 2 仍是裸解析——**Task 2 结束时若 Call 2 收到坏响应仍会崩**。这是 TDD 增量的正常中间态,Task 3 接管 Call 2 后消除。各 Task 的回归只跑 `test_compiler.py`,现有 happy-path 用合法响应,不受影响。

---

## 落地顺序与验证门

- **Task 1 门**:`CompileError` 可从 module0 导入;现有测试全绿。
- **Task 2 门**:Call 1 重试/硬失败两测试过;happy-path 回归绿。
- **Task 3 门**:Call 2 重试/降级两测试过。
- **Task 4 门**:Call 2' 重试/降级两测试过(复现最初崩溃场景现在不崩)。
- **Task 5 门**:`pytest tests/module0/ -m "not requires_model"` 全绿。
- 全部完成后:最初 flaky 的 `test_end_to_end_ambiguous` 的偶发坏响应会被重试/降级吸收(该 live 测试仍靠 .env 触发,不在 CI 路径)。
