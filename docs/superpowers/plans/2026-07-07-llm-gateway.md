# LLM Gateway 实现计划

> 本文件是 LLM Gateway 的实现计划，与设计 spec 平行。
> **设计 Spec**：[`../specs/2026-07-07-llm-gateway-design.md`](../specs/2026-07-07-llm-gateway-design.md)
> **参考实现**：`sft-label` 仓库的 `src/sft_label/llm/` 及周边文件

---

## Context

UXFlow 项目当前是纯文档状态（零代码、零基础设施）。LLM Gateway 是第一个要实现的代码模块，作为模块 0（查询编译）和模块 1（轨迹离线处理）的共享基础设施。

本计划从零搭建 Python 包骨架，然后按依赖顺序实现 gateway 的 7 个源文件 + 对应测试。gateway 解决三类上游拦截：单请求过长、滑动窗口限速、并发过大。

---

## 包布局

```
UXFlow/
├── .gitignore
├── pyproject.toml
├── docs/superpowers/
│   ├── specs/2026-07-07-llm-gateway-design.md
│   └── plans/2026-07-07-llm-gateway.md   ← 本文件
├── src/
│   └── llm_gateway/
│       ├── __init__.py
│       ├── outcomes.py
│       ├── config.py
│       ├── truncation.py
│       ├── runtime.py
│       ├── transport.py
│       ├── recovery.py
│       └── gateway.py
└── tests/
    ├── __init__.py
    ├── conftest.py
    ├── test_outcomes.py
    ├── test_config.py
    ├── test_truncation.py
    ├── test_runtime.py
    ├── test_transport.py
    ├── test_recovery.py
    └── test_gateway.py
```

---

## 依赖关系

```
outcomes.py      ← 无内部依赖（叶子）
config.py        ← 无内部依赖（叶子）
truncation.py    ← 无内部依赖（叶子）
runtime.py       ← 依赖 outcomes
transport.py     ← 依赖 outcomes, config
recovery.py      ← 依赖 runtime（类型引用）
gateway.py       ← 依赖以上全部（门面）
```

---

## 实现步骤

### Step 0：基础设施

**创建文件**：
- `.gitignore`（Python 标准 + .DS_Store + .venv + .env）
- `pyproject.toml`

**动作**：
- `git init`（仓库尚未初始化）
- `pip install -e ".[dev]"`
- commit: "chore: init project with pyproject.toml"

**pyproject.toml 要点**：
- `requires-python = ">=3.11"`
- 唯一运行时依赖：`httpx>=0.27`
- dev 依赖：`pytest>=7.0`、`pytest-asyncio>=0.23`、`ruff`
- build system: hatchling
- `[tool.hatch.build.targets.wheel] packages = ["src/llm_gateway"]`
- `[tool.pytest.ini_options] testpaths = ["tests"]`, `asyncio_mode = "auto"`

**tests/conftest.py 内容**：
```python
import pytest
from llm_gateway.config import GatewayConfig


@pytest.fixture
def default_config():
    """测试友好的 config：低超时、小窗口、快冷却。"""
    return GatewayConfig(
        litellm_base="http://localhost:9999/v1",
        litellm_key="test-key",
        request_timeout=5,
        request_timeout_escalation=[3, 5, 8],
        concurrency=10,
        rps_limit=100.0,
        adaptive_window_requests=5,
        adaptive_window_seconds=2.0,
        adaptive_open_base_cooldown=1.0,
        adaptive_open_max_cooldown=3.0,
        adaptive_min_observations_degraded=2,
        adaptive_min_observations_open=4,
        adaptive_min_failures_degraded=1,
        adaptive_min_failures_open=2,
        fatal_abort_global_rate_min_obs=10,
    )
```

**测试时间策略**：
- 时间敏感测试（RateLimiter 时序、cooldown）通过 monkeypatch `time.monotonic` + mock `asyncio.sleep` 实现确定性
- Gate/Runtime 并发测试用真实 asyncio event loop + 短超时
- Transport 测试用 `httpx.MockTransport`（httpx 内建，无额外依赖）

---

### Step 1：outcomes.py

**参考**：sft-label `src/sft_label/llm/runtime.py` 行 23-202

**实现**：
- `OutcomeClass(str, Enum)` — 9 个值，与 sft-label 一致
- `RequestOutcome` dataclass + 属性 `is_success / is_retryable / is_infra_failure / is_overload / is_abnormal`
- `classify_http_result(status_code, *, error_text, parse_error=False, content_filtered, latency_ms)` → RequestOutcome
- `classify_exception(exc, *, latency_ms)` → RequestOutcome
- 模块级 `__all__`

**与 sft-label 差异**：
- 去掉 `validation_error` 参数（gateway 不做业务 JSON 校验）
- **保留 `parse_error` 参数** — 用于 "200 但 body 缺 `choices[0].message.content` / 结构异常" → ABNORMAL_RESPONSE 的判定（spec §5.1）。这与 validation_error 不同：parse_error 是响应结构层面的异常，不是业务 schema 校验
- 去掉 `monitor_to_outcome_class()`（pipeline 专用）
- 去掉 `PipelineAbortError`（用 `should_abort` 替代）

**测试**：`tests/test_outcomes.py` — 覆盖每个 status code 分支 + 异常分类

**commit**: "feat(outcomes): OutcomeClass enum + classify functions"

---

### Step 2：config.py

**参考**：spec §6 + sft-label `src/sft_label/http_limits.py`

**实现**：
- `GatewayConfig` dataclass，字段完全对齐 spec §6 + 补充以下遗漏字段：
  - `adaptive_probe_success_required: int = 3`（probing 连续成功次数 → degraded）
  - `adaptive_probe_failure_tolerance: int = 3`（probing 连续失败次数 → 回 open）
- `_soft_nofile_limit()` 私有函数
- `resolve_httpx_connection_limits(requested_concurrency, extra_connections=10, reserve_fds=64)` → (max_connections, max_keepalive, capped)
- 模块级 `__all__`

**设计决策**：
- 将 http_limits 合并进 config.py（gateway 内唯一使用处，不需要单独文件）
- `request_timeout_escalation` 用 `field(default_factory=lambda: [60, 90, 120])`

**测试**：`tests/test_config.py` — config 默认值验证 + FD 限制函数在正常/极端 soft limit 下的行为

**commit**: "feat(config): GatewayConfig dataclass + connection limit calculation"

---

### Step 3：truncation.py

**参考**：sft-label `src/sft_label/preprocessing.py` 截断逻辑，但**重写**适配 OpenAI 消息格式

**实现**：
- `TRUNCATION_MARKER = "\n\n[... content truncated ...]\n\n"`
- `estimate_tokens(text: str) -> int` — `len(text) // 4`
- `_truncate_text(text, max_chars, keep_head_ratio=0.3) -> str` — 首尾保留 + marker
- `truncate_messages(messages, *, max_tokens, head_ratio, last_response_ratio, per_turn_ratio) -> list[dict]`

**核心算法**：
1. 分离 system messages（role=="system"），整段保留不截
2. 估算剩余消息总 token，未超预算 → 原样返回
3. 超预算 → 分配：首条 user（head_ratio）、末条 assistant（last_response_ratio）、中间轮均分（per_turn_ratio cap）
4. 超额单条用 `_truncate_text()` 中间截断

**与 sft-label 差异**：输入格式从 ShareGPT (`from/value`) 改为 OpenAI (`role/content`)。算法同源，格式适配。

**测试**：`tests/test_truncation.py` — 不超预算无变化、system 不截、多轮按比例、单消息过长、空列表边界

**commit**: "feat(truncation): message budget truncation for OpenAI format"

---

### Step 4：runtime.py（最复杂）

**参考**：sft-label `src/sft_label/llm/runtime.py` 行 222-989

**实现（近乎逐字移植，不要重写）**：
- `_GatePermit` — 辅助类
- `DynamicConcurrencyGate` — asyncio.Condition 信号量，支持运行时 set_limit / pause / resume
- `AdaptiveRateLimiter` — 令牌桶，支持 set_target_rps / pause_for。**去掉 `probe` 参数**（sft-label 里是 no-op）
- `RuntimePermit` dataclass
- `AdaptiveLLMRuntime` — 状态机 healthy/degraded/open/probing + acquire/observe/snapshot
- `FatalFailureMonitor` — 双信号 abort（streak + 全局率），**去掉文件 I/O**
- 模块级 `__all__`

**移植策略**：DynamicConcurrencyGate 和 AdaptiveLLMRuntime 有微妙的 asyncio 并发正确性（GC 问题、Condition notify race），必须逐字移植 sft-label 的 `_background_tasks` 集合模式和 `_notify_waiters` 实现。

**⚠️ 关键移植注意事项（review 发现）**：
1. `_window` 必须用 `collections.deque`（GIL 保证 append/popleft 原子性）。`observe()` 必须保持同步（不含 await），否则窗口数据有竞态。
2. `DynamicConcurrencyGate.release()` 的 `_background_tasks` 强引用集合必须逐字保留——否则 GC 回收 task 导致 `_in_flight` 永不归零，gate 死锁。
3. `FatalFailureMonitor` 的 `global_rate_min_observations` 默认值用 config 传入（200），不用类自身默认（sft-label 是 20）。

**FatalFailureMonitor 简化**：
- 去掉 `failure_log_path` / `_write_failure_record` / `close()`
- 保留 `record()` / `should_abort` / `abort_reason` / `to_dict()`

**测试**：`tests/test_runtime.py`：
- Gate acquire/release/shrink/pause + 高并发 GC 压力测试（`gc.collect()` between releases）
- RateLimiter 令牌时序（mock `time.monotonic`）
- Runtime **完整状态机周期**：healthy → degraded → open → cooldown → probing → degraded → healthy
- FatalMonitor 双信号触发 + 成功重置 streak

**commit**: "feat(runtime): circuit breaker, dynamic gate, rate limiter, fatal monitor"

---

### Step 5：transport.py

**参考**：sft-label `src/sft_label/llm/transport.py`

**实现**：
- `RequestStats` — 逐字移植
- `async_llm_call(http_client, messages, model, *, temperature, max_tokens, config: GatewayConfig, adaptive_mode: bool) -> tuple[str | None, dict]`
- 模块级 `__all__`

**与 sft-label 差异**：
- 返回 `(content_text | None, usage_dict)` 而非 `(parsed_json, raw, usage)`
- **不做 json.loads** — 只返回 `choices[0].message.content` 原始字符串
- 仍剥离 markdown ``` 外壳 — 移植 sft-label 的 `in_block` 状态机（lines 317-329），作为**独立文本操作**而非 JSON 解析的一部分
- **去掉 `rate_limiter` 参数** — 两种模式都在 gateway 层限流（见 Step 7 限流统一决策）
- `config` 必传（非 Optional），简化所有 fallback
- `adaptive_mode` 作为显式 bool 参数（不用 sft-label 的 `config._adaptive_runtime` 隐藏属性探测）

**⚠️ 关键实现细节（review 发现）**：
1. **双 timeout 模式必须保留**：`asyncio.wait_for(http_client.post(..., timeout=T), timeout=T)`。httpx 的 timeout 可能卡在 DNS/connect，`wait_for` 提供硬性 wall-clock cap。sft-label lines 155-163 是这个模式。
2. **`choices[0].message.content` 提取**：用 try/except `(KeyError, IndexError, TypeError)` 包裹，失败时返回 `(None, usage_dict)` 并标 `parse_error=True`，由 gateway observe 时分类为 ABNORMAL_RESPONSE。
3. **markdown 剥离**：必须用 sft-label 的 `in_block` 状态机，不能简化为 regex（内容含 ``` 会误匹配）。

**保留的 sft-label 逻辑**：
- reasoning model 检测（o1/o3/gpt-5 → 去 temperature，用 max_completion_tokens）
- 403 诊断 header 抽取
- 429 "No deployments available" 检测 + provider_cooldown
- adaptive 模式：429/5xx → 立即返回不本地重试
- 非 adaptive 模式：指数退避 `min(2^attempt * 3 + 2, 60)` + jitter
- 400 non-retryable 关键词检测
- 超时递增（request_timeout_escalation）

**测试**：`tests/test_transport.py` — 使用 `httpx.MockTransport` 模拟各种响应。覆盖：200 正常、200+markdown 包裹、200 body 缺 choices（→ ABNORMAL）、429 adaptive/non-adaptive、403 诊断、401/402、400 content_filter、timeout、双 timeout 验证

**commit**: "feat(transport): async HTTP transport with adaptive retry logic"

---

### Step 6：recovery.py

**参考**：sft-label `src/sft_label/transport_recovery.py`（逐字移植）

**实现**：
- `SwappableAsyncClient` — httpx.AsyncClient 透明代理，支持 `swap()`
- `transport_recovery_loop(*, runtime, swappable, threshold_seconds, cancel_in_flight, on_after_swap, pprint, label)` — 后台协程

**与 sft-label 差异**：无实质差异，该模块抽象干净，无 pipeline 业务耦合。

**测试**：`tests/test_recovery.py` — SwappableAsyncClient 生命周期 + recovery_loop 触发 swap（mock runtime.state）

**commit**: "feat(recovery): swappable httpx client + transport recovery loop"

---

### Step 7：gateway.py + \_\_init\_\_.py

**参考**：spec §3-4，sft-label pipeline 里 acquire→call→observe 模式

**实现 `LLMGateway`**：
- `__init__(config)` — 存配置
- `__aenter__` — 构建 httpx limits → SwappableAsyncClient → AdaptiveLLMRuntime → FatalFailureMonitor → RequestStats →（非 adaptive 模式）独立 `AdaptiveRateLimiter`（可带 warmup）→ 启动 recovery loop
- `__aexit__` — 取消 recovery loop → 关闭 client
- `call(messages, model, *, temperature, max_tokens)` → 截断 → acquire（限流）→ async_llm_call → observe → 记 stats → 释放 permit → 返回
- `runtime_snapshot` / `http_stats` / `should_abort` / `abort_reason` 属性

**⚠️ 限流统一决策（review 发现的矛盾，此处拍板）**：
限流**两种模式都在 gateway 层做一次**，transport 永不持有 rate_limiter：
- **adaptive 模式**：`runtime.acquire()` 同时扣信号量 + 令牌桶。
- **非 adaptive 模式**：gateway 持有一个独立 `AdaptiveRateLimiter`（可选 warmup），`call()` 里在发请求前 `await limiter.acquire()`；并发用一个固定 `asyncio.Semaphore(concurrency)`。
- 这样 transport 保持纯粹（只管一次 HTTP + attempt 级重试），不会双重扣 token。
- 已同步修正 spec §4.2。

**⚠️ RequestStats 调用位置（review 发现）**：
`stats.record(status_code)` / `stats.record_timeout()` 在 `call()` 里 transport 返回后、observe 之后调用（不在 transport 内部）。因为 transport 不再持有 rate_limiter，stats 归 gateway 管。

**⚠️ warmup（review 发现）**：
sft-label 的 warmup 只在非 adaptive 的 `AsyncRateLimiter` 里。本方案的 `AdaptiveRateLimiter` 需补一个可选 warmup 支持（从 1 rps 线性爬到目标，`rps_warmup` 秒），供非 adaptive 模式使用；adaptive 模式不需要（断路器的 probing 已起到渐进作用）。

**`__init__.py`** 导出：
```python
from llm_gateway.gateway import LLMGateway
from llm_gateway.config import GatewayConfig
from llm_gateway.outcomes import OutcomeClass, RequestOutcome
__all__ = ["LLMGateway", "GatewayConfig", "OutcomeClass", "RequestOutcome"]
```

**测试**：`tests/test_gateway.py` — 完整集成测试（mock HTTP），验证：
- 正常调用返回 text + usage
- 截断生效（长消息不被 400）
- 断路器状态转换（连续 429 → degraded → open）
- should_abort 在 AUTH_ERROR streak 后变 True
- 非 adaptive 模式也能工作

**commit**: "feat(gateway): LLMGateway facade + public API exports"

---

## 验证

### 全量测试
```bash
cd /Users/deng/开发/UXFlow
pip install -e ".[dev]"
pytest tests/ -v
```

### import 验证
```python
from llm_gateway import LLMGateway, GatewayConfig
config = GatewayConfig()
print(config.concurrency)  # 200
```

### 端到端集成（需要 LiteLLM 运行）
```python
import asyncio
from llm_gateway import LLMGateway, GatewayConfig

async def smoke():
    config = GatewayConfig(litellm_base="http://localhost:4000/v1", litellm_key="sk-...")
    async with LLMGateway(config) as gw:
        text, usage = await gw.call(
            [{"role": "user", "content": "say hello"}],
            "gpt-4o-mini",
        )
        print(f"text={text[:100]}, usage={usage}")
        print(f"runtime={gw.runtime_snapshot}")

asyncio.run(smoke())
```

### 断路器集成（mock 429 风暴）
test_gateway.py 中构造连续 429 响应 → 验证 runtime 状态从 healthy → degraded → open，cooldown 后 → probing → 恢复。

---

## Checkpoint 与 Commit 节奏

| Step | 通过的测试 | Commit message |
|------|-----------|----------------|
| 0 | — | chore: init project with pyproject.toml |
| 1 | test_outcomes | feat(outcomes): OutcomeClass enum + classify functions |
| 2 | + test_config | feat(config): GatewayConfig + connection limits |
| 3 | + test_truncation | feat(truncation): message budget truncation |
| 4 | + test_runtime | feat(runtime): circuit breaker + gate + rate limiter |
| 5 | + test_transport | feat(transport): async HTTP with adaptive retry |
| 6 | + test_recovery | feat(recovery): swappable client + recovery loop |
| 7 | ALL PASS | feat(gateway): LLMGateway facade + public exports |
