# 模块 0 LLM 健壮性补强 设计规格

> **状态**：设计定稿，待评审。
> **范围**：让 `QueryCompiler.compile()`（`src/module0/compiler.py`）在 LLM 偶发返回空响应 / 坏 JSON / 缺字段时不崩溃——通过「重试 + 分层降级」。不改 Problem Spec 契约、不改 `parsing.py` 的严格解析职责。
> **上游依据**：`interface-contract.md`；`docs/superpowers/specs/2026-07-06-module0-query-compiler-design.md`。
> **触发**：开源化分支收尾时，集成测试 `test_end_to_end_ambiguous` 连跑 6 次 4 过 2 败暴露此缺口。

---

## 1. 背景与根因（实测）

### 1.1 暴露的现象

集成测试 `tests/module0/test_integration.py::test_end_to_end_ambiguous`（输入 `"解题过程中途停止"`，故意歧义，触发 Call 3 澄清 + Call 2' 重评的最复杂链路）连跑 6 次：**4 过 2 败**。两次失败是两种不同的 LLM 输出问题：

- `Missing fields in Call 2 output: ['route']` —— LLM 在 Call 2' 返回的 JSON 缺 `route` 字段。
- `Empty response` —— LLM 返回空内容。

两次都发生在 `_clarify_and_reeval`（`compiler.py:144`）。

### 1.2 根因（不是代码逻辑错，是健壮性缺口）

这是 **LLM 输出的固有非确定性**：同一 prompt 大多数返回合规 JSON，偶尔返回空 / 缺字段。`parse_*` 正确地检测并抛 `ParseError`——问题在于 **`compile()` 对此毫无防护**。

核对代码确认缺口是**系统性的**，不止 Call 2'：

| 缺口 | 位置 | 现状 |
|------|------|------|
| 1. 空响应从不检查 | `compiler.py` 4 处 `gateway.call` 后 | `gateway.call` 返回 `(content \| None, usage)`，失败时 `content=None`；compiler 直接把它喂给 parser → `_extract_json` 的 `if not text` 抛 `ParseError` |
| 2. 裸解析、无重试 | Call 1(:67）、Call 2(:74）、Call 3(:125）、Call 2'(:144) | 任一步解析失败即抛穿，整个 `compile()` 崩溃 |
| 3. Call 2' 路径无 item 容错 | `_clarify_and_reeval` | 主 Call 2 路径有单 item try/except 降级（:92-95），Call 2' 的批级 `parse_call2_response(c2p_text)` 裸调用 |

### 1.3 gateway 不负责内容级重试

`llm_gateway` 有断路器 / 自适应并发，但它重试的是**传输层**故障；「LLM 返回了缺字段的 JSON」对它是一次成功调用。**内容级重试必须在 compiler 层做。**

---

## 2. 已确认决策

| # | 决策点 | 选定 |
|---|--------|------|
| 1 | 核心策略 | **重试 + 分层降级** 组合 |
| 2 | 重试预算 | 每个逻辑步骤额外重试 **1 次**（最多 2 次/步）；旧「最多 4 次 LLM 调用」约束重表述为「4 个逻辑步骤，每步最多 2 次调用」，最坏 8 次 |
| 3 | 降级语义 | **步骤级降级**：单 item 坏→drop；整步坏→空列表；**唯 Call 1 彻底失败→抛 `CompileError`** |
| 4 | 失败分类 | 空响应（content=None）、空串、坏 JSON / 缺字段 **统一视为可重试** |
| 5 | 可观测性 | **结构化计数属性**（side-channel 风格），不引入 logging 框架 |

---

## 3. 核心抽象：带重试的调用+解析封装

现状是 4 处重复的 `text = await gateway.call(...); results = parse_xxx(text)`，全部裸调用。抽出一个私有辅助方法收敛「调用 → 检查空 → 解析 → 失败重试」：

```python
class RetryableParseError(Exception):
    """内部信号：此步应重试（空响应或解析失败）。不外泄。"""


async def _call_and_parse(
    self,
    build_messages_fn,      # 无参 callable，返回 messages（重试时重新构造）
    parser,                 # parse_call1_response / parse_call2_response / ...
    *,
    step_name: str,         # "call1"/"call2"/"call3"/"call2prime"，用于计数
    max_attempts: int = 2,  # 1 次初始 + 1 次重试
):
    last_err = None
    for attempt in range(max_attempts):
        content, _ = await self._gateway.call(
            build_messages_fn(), self._model, max_tokens=self._max_tokens
        )
        try:
            if not content or not content.strip():
                raise RetryableParseError(f"{step_name}: empty response")
            return parser(content)          # 成功
        except (ParseError, RetryableParseError) as e:
            last_err = e
            if attempt + 1 < max_attempts:
                self._robustness["retries"][step_name] += 1
            continue
    raise _StepFailed(step_name, last_err)  # 重试用尽 → 交调用方降级
```

**关键设计点：**
- **统一入口**：空响应（None / 空串）和 `ParseError`（坏 JSON / 缺字段）都被捕获为「可重试」（决策 4）。
- **`build_messages_fn` 用 callable 而非现成 messages**：重试时重新构造消息，保持接口干净，并为「未来重试时微调 prompt」预留 seam。
- **不改 `parse_*` 函数**：它们保持「严格校验、坏就抛 `ParseError`」的纯粹职责；容错逻辑全在 compiler 层。
- **`_StepFailed` 内部异常**：标记「这步重试用尽」，由 `compile()` 按步骤决定降级，不外泄。

---

## 4. 分层降级行为

`compile()` 调用 `_call_and_parse` 后，按步骤重要性处理 `_StepFailed`。

### 4.1 Call 1（拆子问题）—— 硬失败抛异常

```python
try:
    sub_problems_raw = await self._call_and_parse(
        lambda: build_call1_messages(raw_input),
        parse_call1_response, step_name="call1")
except _StepFailed as e:
    raise CompileError(f"Call 1 failed after retries: {e}") from e
```

`CompileError` 是**公开自定义异常**（`module0` 导出）。理由：Call 1 拆不出任何子问题 = 整个编译无从进行，调用方必须区分「彻底失败」与「没匹配到」。

### 4.2 Call 2（打标+自评）—— 步骤降级为空

```python
try:
    c2_results = await self._call_and_parse(
        lambda: build_call2_messages(sub_problems_raw, self._taxonomy),
        parse_call2_response, step_name="call2")
except _StepFailed:
    c2_results = []
    self._robustness["degraded"].append("call2")
```

`c2_results=[]` → 无 passed、无 ambiguous → 后续自然产出空 `ProblemSpec`（合法）。不抛：Call 1 已成功，返回空 spec 是合理的「没筛出东西」，非「系统坏了」。
**单 item 降级保持现状**（`compiler.py:92-95` 的 try/except → drop），不动。

### 4.3 Call 3 + Call 2'（澄清路径）—— 步骤降级为空

```python
if ambiguous:
    try:
        passed.extend(await self._clarify_and_reeval(ambiguous))
    except _StepFailed:
        self._robustness["degraded"].append("clarify")
        # 澄清失败 → 这些 ambiguous 停留在 dropped_records，不 recover，不崩
```

`_clarify_and_reeval` 内部 Call 3、Call 2' 各自用 `_call_and_parse`；任一步重试用尽 → 抛 `_StepFailed` → 被此处接住降级。**这正是最初崩溃的路径**——现在优雅降级：歧义子问题保留审计记录，主流程继续。
Call 2' 的**单 item 降级保持现状**（`compiler.py:152-157` 静默跳过），不动。

### 4.4 降级层级总览

| 失败点 | 行为 | compile() 结果 |
|--------|------|----------------|
| Call 1 重试用尽 | 抛 `CompileError` | 异常（调用方感知彻底失败） |
| Call 2 整批重试用尽 | 空 c2_results | 空 ProblemSpec |
| Call 2 单 item 坏 | 该 item → drop（现状） | 少一个子问题 |
| Call 3/2' 重试用尽 | 该批 ambiguous 不 recover | 少若干 clarified 子问题 |
| Call 2' 单 item 坏 | 该 item 静默跳过（现状） | 少一个子问题 |

---

## 5. 可观测性

`__init__` 初始化结构化字典，与 `dropped_records` / `hyde_embeddings` 的 side-channel 风格一致：

```python
def __init__(self, ...):
    ...
    self.dropped_records = []
    self.hyde_embeddings = {}
    self._robustness = {
        "retries": {"call1": 0, "call2": 0, "call3": 0, "call2prime": 0},
        "degraded": [],   # 发生降级的步骤，如 ["call2", "clarify"]
    }

@property
def robustness_report(self) -> dict:
    """本次 compile 的重试/降级统计（供运维/测试读取）。"""
    return self._robustness
```

- 每次 `compile()` 开头**重置**（同 `dropped_records`，每次运行独立）。
- 通过 `@property` 暴露，不引入 logging——保持 module0 无新横切依赖。
- **产品级余量**：结构已机器可读，未来接监控只需在此 property 之上加导出层。

---

## 6. 测试策略

全部**纯逻辑测试**（假 gateway，零真实 LLM、零网络），会在 `pytest -m "not requires_model"` 与 CI 全平台跑。扩展现有 `test_compiler.py` 的 `FakeGateway` 支持坏响应注入。

| 测试 | 构造（FakeGateway 依次返回） | 断言 |
|------|------------------------------|------|
| Call 2 空响应重试后成功 | `None` → 合法 JSON | compile 成功；`robustness_report["retries"]["call2"]==1` |
| Call 2 重试用尽降级 | `None`、`None` | 返回空 ProblemSpec；`"call2" in degraded` |
| Call 2' 缺 route 重试后成功 | 缺字段 JSON → 合法 JSON | clarified 子问题被 recover；retry 计数+1 |
| Call 2' 重试用尽降级（**复现最初崩溃**） | 连续缺字段 | 不抛；ambiguous 停留 dropped_records；`"clarify" in degraded` |
| Call 1 硬失败抛 CompileError | 连续 `None` | `pytest.raises(CompileError)` |
| 坏 JSON 与空响应统一重试 | `"not json{"` → 合法 JSON | 成功；retry 计数+1 |
| 正常路径无重试（回归保护） | 全部合法 | `retries` 全 0、`degraded` 空 |

- 现有 live 集成测试 `test_end_to_end_*` 保持不动，但其 flaky 现被容错吸收——偶发坏响应被重试/降级，不再崩。

---

## 7. 文件改动清单

**改动**
- `src/module0/compiler.py`：新增 `_call_and_parse`、`RetryableParseError`、`_StepFailed`、`CompileError`、`_robustness` + `robustness_report`；改 Call 1/2 的调用点与 `_clarify_and_reeval` 内 Call 3/2' 的调用点。
- `src/module0/__init__.py`：导出 `CompileError`。
- `tests/module0/test_compiler.py`：扩展 `FakeGateway` 支持坏响应注入 + §6 新测试。

**不动**
- `src/module0/parsing.py`：保持严格解析（坏就抛 `ParseError`）。
- `src/module0/prompts.py`、`schema.py`：无关。
- Problem Spec 契约、下游模块 1/2/3：无影响。

---

## 8. 不做（本次）

- 重试时改写 prompt（如追加「务必包含 route 字段」）——`build_messages_fn` 已预留 seam，本次不实现。
- 区分空响应 vs 坏 JSON 的差异化重试策略——决策 4 统一处理。
- 每步重试次数 > 1 或全局重试配额——决策 2 定为每步 1 次。
- 接入 logging / 监控导出——决策 5 只做结构化属性。

---

## 附录：开放点

无。所有决策已在 §2 确认。
