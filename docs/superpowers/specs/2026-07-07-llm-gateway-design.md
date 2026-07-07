# LLM Gateway 设计 Spec

> 通用 LLM 调用网关，服务模块 0（轻量交互式调用）和模块 1（百万级离线批量调用）。
> 按模块 1 的强度设计，模块 0 视为其轻量调用方。

---

## 1. 设计目标与约束

### 1.1 要解决的三个拦截问题

| # | 拦截类型 | 表现 | 对策层 |
|---|---------|------|--------|
| 1 | 单请求过长 | input token > ~10k 被 WAF/代理拦截（403 或静默丢弃） | 发送前截断 |
| 2 | 滑动窗口限速 | 短时间请求数过多 → 429/503，有冷却期 | 令牌桶 + 断路器 |
| 3 | 并发过大 | 同时在途请求过多 → 连接被拒/超时 | 动态信号量 + 断路器 |

### 1.2 上游形态

自建 LiteLLM 代理（OpenAI 兼容接口），支持多 key 路由。

### 1.3 设计原则

- **对齐 sft-label**：架构、组件拆分、参数命名、默认值尽量一致，降低认知切换成本。
- **门面极简**：调用方只看到 `gateway.call(messages, model, ...) → (text, usage)`，零感知限流细节。
- **全局共享**：单一 gateway 实例服务所有模块，实现全局背压。
- **调用方自己解析**：gateway 只返回原始文本，JSON 解析/schema 校验由调用方负责。

---

## 2. 目录结构

```
llm_gateway/
  ├── __init__.py       # 导出 LLMGateway, GatewayConfig
  ├── gateway.py        # LLMGateway 门面：编排四组件 + call() 入口
  ├── config.py         # GatewayConfig dataclass
  ├── transport.py      # async_llm_call：单次 HTTP + 状态码处理 + 逐次退避
  ├── runtime.py        # AdaptiveRuntime：断路器状态机 + DynamicConcurrencyGate + AdaptiveRateLimiter
  ├── truncation.py     # 发送前长度预算截断
  ├── recovery.py       # SwappableAsyncClient + transport_recovery_loop（僵尸连接热替换）
  └── outcomes.py       # OutcomeClass 枚举 + RequestOutcome + classify_http_result
```

---

## 3. 门面 API

### 3.1 LLMGateway

```python
class LLMGateway:
    """全局 LLM 调用网关，内部持有 httpx client + 断路器 + 令牌桶 + 信号量。"""

    def __init__(self, config: GatewayConfig):
        ...

    async def __aenter__(self) -> "LLMGateway":
        """初始化 httpx client、启动 recovery loop（如配置）。"""
        ...

    async def __aexit__(self, *exc) -> None:
        """关闭 httpx client、停止 recovery loop。"""
        ...

    async def call(
        self,
        messages: list[dict],
        model: str,
        *,
        temperature: float = 0.1,
        max_tokens: int = 1000,
    ) -> tuple[str | None, dict]:
        """
        发起一次 LLM 调用。

        返回:
            (content, usage_dict)
            - content: 模型返回的原始文本，失败时为 None
            - usage_dict: 包含 prompt_tokens, completion_tokens, status_code,
                          以及可能的 error, non_retryable 等诊断字段

        调用方自行 json.loads(content) 做解析。
        """
        ...

    @property
    def runtime_snapshot(self) -> dict:
        """当前断路器状态快照（用于监控/日志）。"""
        ...

    @property
    def http_stats(self) -> dict:
        """HTTP 请求统计摘要。"""
        ...

    @property
    def should_abort(self) -> bool:
        """致命失败监控是否已触发 abort。调用方在批量循环中轮询此属性。"""
        ...

    @property
    def abort_reason(self) -> str:
        """abort 原因描述（should_abort=False 时为空字符串）。"""
        ...
```

### 3.2 调用方使用示例

```python
from llm_gateway import LLMGateway, GatewayConfig

# 模块 0：用默认配置（concurrency=200 对 4 次调用无约束）
config = GatewayConfig(litellm_base="http://...", litellm_key="sk-...")
async with LLMGateway(config) as gw:
    text, usage = await gw.call(messages, "gpt-4o-mini", max_tokens=2000)
    if text:
        result = json.loads(text)

# 模块 1：覆盖并发/RPS
config = GatewayConfig(
    litellm_base="http://...",
    litellm_key="sk-...",
    concurrency=500,
    rps_limit=30.0,
)
async with LLMGateway(config) as gw:
    tasks = [gw.call(msg, model) for msg in batch]
    results = await asyncio.gather(*tasks)
```

---

## 4. 一次 `call()` 的完整生命周期

```
call(messages, model, ...)
  │
  ▼ ① 截断层 (truncation)
  │   检查 messages 总 token 估算 vs max_request_tokens
  │   超预算 → 按比例策略截断
  │   产出：truncated_messages
  │
  ▼ ② 断路器准入 (runtime.acquire)
  │   当前状态：
  │     healthy   → 直接放行
  │     degraded  → 降档后放行（并发/rps 已压低）
  │     open      → 阻塞等冷却期结束
  │     probing   → 仅放少量探针
  │   获得 permit（信号量占位 + 令牌桶扣 token）
  │
  ▼ ③ 传输层 (transport.async_llm_call)
  │   组装 payload → POST /chat/completions
  │   响应处理：
  │     200 → 提取 content
  │     429/5xx → 指数退避重试（adaptive 模式下不本地重试，直接返回）
  │     403(WAF) → 最多重试 1 次
  │     400(content_filter/context_length) → 不重试，标 non_retryable
  │     401/402 → 不重试
  │   返回 (content_text, usage_dict)
  │
  ▼ ④ 观测反馈 (runtime.observe)
  │   结果分类为 OutcomeClass
  │   喂给断路器滑动窗口 → 可能触发状态转换
  │   释放 permit（信号量归还）
  │
  ▼ 返回 (text, usage) 给调用方
```

### 4.1 重试职责划分

| 层 | 谁重试 | 重试目标 | 退避策略 | 最大次数 |
|---|---|---|---|---|
| transport（attempt 级） | gateway 内部 | 429/5xx/timeout | `min(2^attempt × 3 + 2, 60)` 秒 + jitter | `max_retries`（默认 3） |
| sample 级 | **调用方**自己决定 | transport 耗尽重试后仍失败 | 调用方自定 | 不在 gateway 管辖 |

### 4.2 Adaptive 模式下的行为差异

启用 `enable_adaptive_runtime=True`（默认）时：
- transport 遇 429/5xx → **不做本地重试**，立即返回带 `non_retryable=False` 的 usage
- 由外层 observe → 断路器决定降档/open
- 原因：避免在 transport 里闷头重试 3 次，掩盖窗口内真实失败率
- **限流只在 gateway 层做一次**（runtime.acquire 同时获取信号量 + 令牌桶），transport 不持有 rate_limiter，避免双重扣 token

关闭时（退化为简单限流）：
- transport 正常 3 次重试 + jitter
- **限流仍在 gateway 层做**：gateway 持有一个独立 `AdaptiveRateLimiter`（可选 warmup）+ 固定 `asyncio.Semaphore(concurrency)`，`call()` 发请求前限流。transport 始终不持有 rate_limiter。
- 令牌桶/信号量仍生效，但无断路器状态机自适应

---

## 5. 组件详细设计

### 5.1 outcomes.py — 结果分类

```python
class OutcomeClass(str, Enum):
    SUCCESS = "success"
    TIMEOUT = "timeout"
    OVERLOAD = "overload"              # 429
    SERVER_ERROR = "server_error"      # 500/502/503/504
    ABNORMAL_RESPONSE = "abnormal_response"  # 200 但 body 无 choices[0].message.content
    CONTENT_FILTERED = "content_filtered"    # 400 + 内容过滤关键词
    AUTH_ERROR = "auth_error"          # 401/402
    INPUT_ERROR = "input_error"        # 400 + context_length_exceeded
    TRANSIENT_ERROR = "transient_error"     # 其他可重试错误
```

分类逻辑对齐 sft-label `classify_http_result()`：

| status_code | 条件 | 分类 |
|---|---|---|
| 2xx + response body 非空 | — | SUCCESS |
| 2xx + response body 为空或无 choices[0] | — | ABNORMAL_RESPONSE |
| 429 | — | OVERLOAD |
| 500/502/503/504 | — | SERVER_ERROR |
| 400 | 含 content_filter/moderation 关键词 | CONTENT_FILTERED |
| 400 | 含 context_length_exceeded | INPUT_ERROR |
| 400 | 其他 | TRANSIENT_ERROR |
| 401/402 | — | AUTH_ERROR |
| 403 | — | TRANSIENT_ERROR |
| 其他 4xx | — | INPUT_ERROR |
| exception: TimeoutError | — | TIMEOUT |
| exception: 其他 | — | TRANSIENT_ERROR |

### 5.2 runtime.py — 自适应断路器

#### 状态机

```
         ┌─────────────────────────────────────────────────┐
         │                                                 │
         ▼                                                 │
     ┌────────┐  timeout/overload率    ┌──────────┐       │
     │HEALTHY │ ──── ≥ degrade阈值 ──→ │DEGRADED  │       │
     └────────┘                        └──────────┘       │
         ▲                                  │             │
         │ 恢复（连续3窗口健康                │ 率继续恶化    │
         │   + 并发/rps爬回base）            ▼             │
         │                             ┌────────┐         │
         │                             │  OPEN  │←────────┘
         │                             └────────┘
         │                                  │
         │                                  │ 冷却期结束
         │                                  ▼
         │                             ┌─────────┐
         └──── probe连续成功 ←──────── │PROBING  │
                                       └─────────┘
                                            │
                                            │ probe连续失败
                                            ▼
                                        回到 OPEN（冷却期×2）
```

#### 核心子组件

**DynamicConcurrencyGate**：基于 asyncio.Condition 的可动态调限信号量。
- `set_limit(n)` → 运行时收缩/放大，阻塞中的 acquire 自动唤醒
- `pause()` / `resume()` → open 态暂停所有新请求

**AdaptiveRateLimiter**：令牌桶。
- `set_target_rps(rps)` → 运行时调 RPS
- `pause_for(seconds)` → 冷却期直接暂停
- warmup：冷启动时从 1 rps 线性爬到目标

**AdaptiveLLMRuntime**：持有 gate + rate_limiter + 滑动窗口，编排状态转换。

#### 状态转换参数

| 参数 | 默认值 | 含义 |
|---|---|---|
| window_requests | 50 | 滑动窗口容量 |
| window_seconds | 20.0 | 滑动窗口时长 |
| timeout_rate → degraded | ≥5% | 超时率触发降档 |
| overload_rate → degraded | ≥5% | 429 率触发降档 |
| abnormal_rate → degraded | ≥4% | 异常响应率触发降档 |
| timeout_rate → open | ≥20% | 超时率触发 open |
| overload_rate → open | ≥15% | 429 率触发 open |
| abnormal_rate → open | ≥60% | 异常响应率触发 open |
| min_observations_degraded | 3 | 降档最少观测数 |
| min_observations_open | 12 | open 最少观测数 |
| open_base_cooldown | 15s | 冷却基数 |
| open_max_cooldown | 30s | 冷却上限（指数退避 cap） |
| degrade_concurrency_factor | 0.5 | 降档并发 = base × 0.5 |
| degrade_rps_factor | 0.6 | 降档 RPS = base × 0.6 |
| recovery_concurrency_step | +2 | 恢复每步加并发 |
| recovery_rps_step | +1.0 | 恢复每步加 RPS |
| probe_success_required | 3 | probing 连续 3 成功 → degraded |
| probe_failure_tolerance | 3 | probing 连续 3 失败 → 回 open |

#### 恢复逻辑

degraded 态每个健康窗口 +2 并发 / +1.0 rps，连续 3 个健康窗口 + 已爬回 base → 回 healthy。

### 5.3 transport.py — HTTP 传输

对齐 sft-label `async_llm_call()`，关键行为：

| 场景 | 行为 |
|---|---|
| reasoning model (o1/o3/gpt-5*) | 去 temperature，用 max_completion_tokens |
| 403 | 诊断 header 抽取（区分 WAF/proxy/provider），最多重试 1 次 |
| 429 body 含 "No deployments available" | 标 `provider_cooldown=True`，adaptive 下不本地重试 |
| 200 + content 以 \`\`\` 开头 | 自动剥离 markdown 代码块外壳（保留内部文本） |
| 200 + body 无 choices[0].message.content | 标 ABNORMAL_RESPONSE |
| 响应解析 | **不做 json.loads**——只返回 choices[0].message.content 原始字符串（调用方自己解析） |

### 5.4 truncation.py — 发送前截断

对齐 sft-label `preprocessing.py` 的截断策略，但提取为独立函数：

```python
def truncate_messages(
    messages: list[dict],
    *,
    max_tokens: int = 10000,
    head_ratio: float = 0.35,
    last_response_ratio: float = 0.30,
    per_turn_ratio: float = 0.35,
) -> list[dict]:
    """
    按比例预算截断 messages。

    策略：
      0. system message（role=="system"）整段保留不截，不计入比例预算
      1. 估算非 system 部分总 token（chars / 4 粗估，或 tiktoken 精确）
      2. 若未超预算 → 原样返回
      3. 超预算 → 按比例分配剩余预算（总预算 - system 消耗）：
         - 首轮（任务上下文）：剩余预算 × head_ratio
         - 末轮（最近回复）：剩余预算 × last_response_ratio
         - 中间轮：均分剩余，单轮 cap = 剩余预算 × per_turn_ratio
         - 每轮超额部分从中间截断，保留首尾
    """
    ...
```

token 估算策略：默认 `len(text) / 4` 粗估（快且无外部依赖），可选注入 tiktoken encoder 精确计数。

### 5.5 recovery.py — 传输恢复

对齐 sft-label `transport_recovery.py`：

- **SwappableAsyncClient**：httpx.AsyncClient 透明代理，支持运行时热替换底层 client。
- **transport_recovery_loop**：后台协程，监控 runtime 是否长时间卡在 open 态（`transport_stuck_seconds`），超时则：
  1. `swap()` 安装新 httpx client
  2. 取消在途任务
  3. 把 runtime 推入 probing

模块 0（几次调用就结束）不需要这个——只有模块 1 长跑多小时才可能碰到僵尸连接。gateway 根据 `transport_stuck_seconds > 0` 决定是否启动此 loop。

### 5.6 HTTP 连接池上限（对齐 sft-label `http_limits.py`）

并发防线的第二层：在 DynamicConcurrencyGate（应用层信号量）之下，httpx 自身的连接池也需要匹配。

```python
def resolve_httpx_connection_limits(
    *,
    requested_concurrency: int,
    extra_connections: int = 10,
    reserve_fds: int = 64,
) -> tuple[int, int, bool]:
    """
    根据进程 FD 软上限推导安全的 httpx 连接池参数。

    返回: (max_connections, max_keepalive_connections, capped)
    - max_connections = min(concurrency + extra, 可用 FD)
    - max_keepalive = min(concurrency, max_connections)
    - capped: 是否因 FD 不足而被下压
    """
    ...
```

**设计要点**：
- `max_connections` 不能低于 `concurrency`，否则信号量放行了但池里拿不到连接，请求卡在 httpx 内部排队。
- 通过 `resource.getrlimit(RLIMIT_NOFILE)` 取进程 FD 软上限，减去 reserve（日志文件/checkpoint 等非 HTTP FD），得到可用 FD 预算。
- 被 cap 时打 warning，提示用户 `ulimit -n` 不够。
- gateway 初始化时调用此函数，结果传入 `httpx.Limits(max_connections=..., max_keepalive_connections=...)`。

---

## 6. GatewayConfig 完整字段

```python
@dataclass
class GatewayConfig:
    """LLM Gateway 全配置。字段名和默认值对齐 sft-label。"""

    # ─── 连接 ───────────────────────────────────
    litellm_base: str = "http://localhost:4000/v1"
    litellm_key: str = ""

    # ─── 单请求控制 ─────────────────────────────
    max_retries: int = 3
    request_timeout: int = 90
    request_timeout_escalation: list | None = None  # 默认 [60, 90, 120]

    # ─── 限流（令牌桶）──────────────────────────
    rps_limit: float = 20.0
    rps_warmup: float = 30.0

    # ─── 并发（动态信号量）──────────────────────
    concurrency: int = 200

    # ─── 自适应断路器 ─────────────────────────
    enable_adaptive_runtime: bool = True
    adaptive_min_concurrency: int = 4
    adaptive_min_rps: float = 2.0
    adaptive_window_requests: int = 50
    adaptive_window_seconds: float = 20.0
    adaptive_timeout_rate_degraded: float = 0.05
    adaptive_overload_rate_degraded: float = 0.05
    adaptive_abnormal_rate_degraded: float = 0.04
    adaptive_min_observations_degraded: int = 3
    adaptive_min_failures_degraded: int = 2
    adaptive_timeout_rate_open: float = 0.20
    adaptive_overload_rate_open: float = 0.15
    adaptive_abnormal_rate_open: float = 0.60
    adaptive_min_observations_open: int = 12
    adaptive_min_failures_open: int = 4
    adaptive_open_base_cooldown: float = 15.0
    adaptive_open_max_cooldown: float = 30.0
    adaptive_degrade_concurrency_factor: float = 0.5
    adaptive_degrade_rps_factor: float = 0.6
    adaptive_recovery_concurrency_step: int = 2
    adaptive_recovery_rps_step: float = 1.0
    adaptive_probe_success_required: int = 3    # probing 连续 N 次成功 → degraded
    adaptive_probe_failure_tolerance: int = 3   # probing 连续 N 次失败 → 回 open

    # ─── 截断预算 ──────────────────────────────
    max_request_tokens: int = 10000
    truncation_head_ratio: float = 0.35
    truncation_last_response_ratio: float = 0.30
    truncation_per_turn_ratio: float = 0.35

    # ─── 传输恢复 ──────────────────────────────
    transport_stuck_seconds: int = 300

    # ─── 致命失败监控 ──────────────────────────
    fatal_abort_enabled: bool = True
    fatal_abort_streak_limit: int = 5
    fatal_abort_global_rate_limit: float = 0.95
    fatal_abort_global_rate_min_obs: int = 200
```

---

## 7. 致命失败监控（FatalFailureMonitor）

对齐 sft-label，双信号 abort：

| 信号 | 条件 | 触发时机 |
|---|---|---|
| Fatal streak | 连续 N 次 AUTH_ERROR（默认 5） | 秒级发现"billing 耗尽/key 失效" |
| Global failure rate | 失败率 ≥ 95%（最少 200 次观测后） | 分钟级发现持续性降级 |

**触发方式：外部轮询（对齐 sft-label）**，`call()` 不抛异常。gateway 暴露 `should_abort` / `abort_reason` 属性，由调用方在批量循环里主动检查、自行决定中断粒度和收尾（checkpoint、写日志）：

```python
async with LLMGateway(config) as gw:
    for chunk in chunks:
        if gw.should_abort:
            log(f"aborting: {gw.abort_reason}")
            break  # 调用方自己做 checkpoint / 收尾
        results = await asyncio.gather(*[gw.call(m, model) for m in chunk])
```

- gateway 内部每次 `observe()` 后更新 monitor 计数，只翻转 `should_abort` 标志，绝不主动打断在途请求。
- 模块 0（4 次调用）可忽略此机制或 `fatal_abort_enabled=False`；模块 1 长跑批量循环每个 chunk 前查一次。

---

## 8. 监控与可观测性

### 8.1 RequestStats

每个 gateway 实例累计统计：
- success / errors(by status_code) / timeouts
- `summary_line()` → 单行摘要如 `http(✓1200 429×3 timeout×1 99.7%)`

### 8.2 runtime_snapshot

暴露断路器实时状态：
```json
{
  "state": "degraded",
  "effective_concurrency": 100,
  "effective_rps": 12.0,
  "in_flight": 87,
  "cooldown_remaining_seconds": 0,
  "rates": {"timeout_rate": 0.02, "overload_rate": 0.04, ...}
}
```

调用方可按需打日志 / 接入 metrics 系统。

---

## 9. 模块 0 vs 模块 1 的使用差异

| | 模块 0 | 模块 1 |
|---|---|---|
| 调用次数 | 2–4 次 / 问题清单 | 百万级 |
| 并发 | 默认 200（无实际约束） | 覆盖为 500+ |
| 截断 | 用户原始描述短，通常不触发 | 轨迹超长，重度依赖截断 |
| transport_recovery | 不启动（几秒跑完） | 启动（多小时长跑） |
| fatal_abort | 可关闭（4 次失败不构成统计意义） | 开启 |
| sample 级重试 | 调用方按 spec 重跑 Call 2 | 调用方管恢复队列 |

---

## 10. 未来增强项（TODO）

- **TODO：多 endpoint 路由** — LiteLLM 本身支持多 deployment，但 gateway 侧可能需要 provider-aware 的降档（某个 provider 挂了只降它的流量）。
- **TODO：精确 token 计数** — 当前粗估 chars/4，后续接入 tiktoken 或模型自带 tokenizer 做精确预算。
- **TODO：metrics 上报** — Prometheus / OpenTelemetry 集成，export 断路器状态 + 延迟分位数。
- **TODO：请求优先级队列** — 模块 0 交互式调用可插队高优，模块 1 批量走低优通道。
