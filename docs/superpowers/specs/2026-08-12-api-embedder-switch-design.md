# 切换 Embedding 到 API 后端

日期：2026-08-12
分支：`feat/api-embedder-switch`
状态：设计已确认，待实现

## 1. 背景与问题

本地 `LocalEmbedder`（Qwen3-Embedding-0.6B，CPU）速率无法支撑真实数据处理需求。

实测量化（语料为本库 `slice_sources` 还原的真实切片文本）：

| 后端 | 吞吐 | 1113 条切片 |
|---|---|---|
| 本地 CPU（`device="cpu"`，1024 维） | **0.57 texts/s** | **~33 分钟** |
| **出货配置**（batch 32 / 3072 维 / base64） | **5.7 texts/s** | **~3.3 分钟** |
| 参考：batch 512 / 1024 维 / JSON | 21.5 texts/s | ~52 秒 |

出货配置相对本地 CPU 约 **10 倍**改善。

注意它**不是** 38 倍——那是 batch 512 / 1024 维下的数字。出货配置为迁就网关做了两处让步：
批次 512→32（请求数 ×16）、维度 1024→3072（传输量 ×3）。这是自觉的取舍：
被网关拦截等于 0 texts/s，10 倍已足够跨过可用性门槛。

> 关于 MPS：本开发机 MPS 可用，MPS 下本地可达 16–20 texts/s，反而高于出货配置。
> 但**目标用户机器无 MPS**，因此 CPU 才是有效基线；MPS 数字不能用于论证或否证
> 本次迁移，记录在此仅为避免后续有人在带 MPS 的机器上复测时得出矛盾结论。

项目设计之初已预留 API 调用 seam，本次是**接通并调优已有 seam**，不是新建功能：

| 已存在的组件 | 位置 |
|---|---|
| `Embedder` Protocol | `src/uxflow_embed/protocol.py:16` |
| `ApiEmbedder`（OpenAI 兼容 `/v1/embeddings`） | `src/uxflow_embed/api.py` |
| `make_embedder()` 已认 `UXFLOW_EMBED_BACKEND=api` | `src/uxflow_runtime.py:61` |
| 6 个 ApiEmbedder 测试 | `tests/uxflow_embed/test_api.py` |
| `httpx>=0.27` 已在主依赖 | `pyproject.toml:12` |

因此改动集中在**参数调优、可靠性加固、批次策略**，协议保持向后兼容。

## 2. 根因分析：为什么大量超时

排查过程（针对 `http://1.95.77.23:3000/v1`）：

**分层隔离** —— 排除网络与代理整体故障：

| 层 | 结果 |
|---|---|
| 原始 TCP 连接 | 10/10 通过，中位 0.04s |
| `GET /models` | 10/10 通过，中位 0.13s |
| 单条 embedding | 10/10 通过，中位 1.22s |
| 批量 embedding | ~30% 超时 |

**排除批次大小因素** —— 超时率与批次无关：

| 批次 | 成功率 |
|---|---|
| n=32 | 2/3 |
| n=64 | 2/3 |
| n=128 | 2/3 |
| n=256 | 2/3 |
| n=512 | 3/3 |

**决定性证据** —— 超时放宽到 90s 后 8/8 全部成功，耗时散布 2.2s–86.8s。

**结论：请求从未真正失败或挂死，而是在代理队列中长时间排队。** 代理并发处理能力有限，请求随机落入慢队列。

> 修正记录：排查中期曾误判为「要么 10 秒返回，要么永不返回」。该判断错误——中间地带不空，是 25–30s 的客户端超时人为截断了长尾。所谓「精确 25.04 秒的失败」是客户端准时超时，非服务端沉默。

**策略验证** —— 短超时快重发 vs 耐心等待（n=64，各 8 轮）：

| 策略 | 成功率 | 平均耗时 |
|---|---|---|
| 12s 超时 + 立即重试（≤4 次） | 8/8 | **8.51s** |
| 90s 耐心等待，单次 | 8/8 | 29.39s |

两者成功率相同，短超时重试快 3.5 倍。机制：请求一旦排入慢队列会持续慢，放弃后重发通常落到空闲 worker，数秒返回。

**因此重试用固定短间隔而非指数退避**——根因是排队而非过载，退避只是白等。

## 3. 向量维度决策

用库中 **1113 条真实切片文本**（`slice_sources` 表，经 `_json_to_slice()` + `build_embedding_text()` 还原）实测，非合成数据。语料 400 条、查询 100 条，以 3072 全维为基准比较 top-20 重叠率：

> 数字说明：`signatures` 表有 1132 行，`slice_sources` 还原出 1113 条非空文本，二者差异来自空文本切片。下文提到 1132 时指存量向量数，1113 指可嵌入文本数。

| 维度 | 召回重叠率 | 最差查询 | 存储(1113条) | 召回计算 |
|---|---|---|---|---|
| 256 | 79.2% | 45% | 1.09 MB | — |
| 512 | 87.0% | 55% | 2.17 MB | — |
| 1024 | 91.9% | 75% | 4.35 MB | 0.102 ms |
| 1536 | 95.5% | 80% | 6.52 MB | 0.148 ms |
| **3072** | **100%** | — | **13.04 MB** | **0.325 ms** |

**生成速度与维度无关**——3072 维那轮（55s）反而快于 1024 维（124s），耗时完全被代理排队噪声主导。

**决策：使用 3072 原生全维，不降维。**

降维代价（约 8% 召回准确度，最差查询丢 25% 结果）远大于收益：存储 13MB 相对现有 54MB 库可忽略；召回计算 0.325ms 被后续 LLM judge 的秒级调用完全淹没。

> 修正记录：初版方案选 1024，理由是「与现有 schema 和标定基线对齐，不引入第二个变量」。该理由不成立——既然要新建库、阈值本就要重标，「对齐旧基线」无实际收益，等于为不存在的收益付出 8% 召回质量。

## 4. 设计

### 4.1 模型与端点

- 模型：`text-embedding-3-large`
- 维度：3072（原生，**不传 `dimensions` 参数**）
- 端点：复用现有 LiteLLM proxy

选型依据（N=512，dimensions=1024，交替 6 轮，避免代理负载漂移偏袒单方）：

| 模型 | 中位吞吐 | 成功率 |
|---|---|---|
| `text-embedding-3-small` | 52.5 t/s | 4/6 |
| `text-embedding-3-large` | 57.5 t/s | 5/6 |

large 至少不劣于 small。差距（9%）在 6 轮采样噪声范围内，**不能声称 large 显著更优**，但可确定：没有理由为速度选 small，因而 large 的质量优势是净收益。

> 修正记录：初版基于单条延迟（small 1.78s vs large 4.14s）推断 large 慢 2.3 倍。该推断错误——单条延迟主要反映往返开销，批次上到 512 后瓶颈在代理排队，模型计算差异被淹没。

### 4.2 认证：复用 LiteLLM 凭证

`make_embedder()` 的 api 分支按以下顺序解析：

1. `LITELLM_KEY` / `LITELLM_BASE`（优先）
2. `OPENAI_API_KEY` / `UXFLOW_EMBED_API_BASE`（回退）

用户 `.env` 无需新增任何字段，仅需 `UXFLOW_EMBED_BACKEND=api`。

### 4.3 传输编码：base64

请求体固定带 `encoding_format: "base64"`。实测（真实切片文本，3072 维）：

| 批次 | JSON 浮点数组 | base64 | 缩减 | 耗时对比 |
|---|---|---|---|---|
| 32 | 1.85 MB | **0.50 MB** | 3.7× | 4.22s → 3.03s |
| 128 | 7.39 MB | **2.01 MB** | 3.7× | 7.23s → 5.38s |

解码后 norm=0.9998，数值无损。体积和耗时同时下降，无取舍，无条件采用。

`ApiEmbedder` 需处理两种响应形态：`data[].embedding` 为 str 时按 base64 解 float32，为 list 时按原路径处理（保持对不支持该参数的端点的兼容）。

### 4.4 批次：`preferred_batch_size` 可选协议属性

`protocol.py` 增加可选属性声明；各实现取值：

| 实现 | preferred_batch_size |
|---|---|
| `ApiEmbedder` | 32（可经 `UXFLOW_EMBED_BATCH` 覆盖） |
| `LocalEmbedder` | 32 |
| `FakeEmbedder` | 128 |

`module1/pipeline.py` 改为：

```python
chunk_size = getattr(emb_model, "preferred_batch_size", _EMBED_CHUNK)
```

`getattr` 带默认值确保：任何第三方实现无需修改即可运行，`runtime_checkable` 行为不受影响，`_EMBED_CHUNK = 32` 保留为兜底默认。

**默认 32 的依据**：目标用户在公司受限网络内使用，**网关响应体上限约 1 MB**（用户确认）。

响应体大小可精确推算，与文本内容无关——base64 编码的 float32 长度仅由维度与条数决定：

```
3072 维 × 4 B = 12,288 B  →  base64 = 16,384 B ≈ 16 KB/条
```

| 批次 | 响应体 | 占 1 MB 额度 |
|---|---|---|
| 16 | 0.26 MB | 26% |
| **32（默认）** | **0.51 MB** | **51%** |
| 64 | 1.03 MB | **103% — 超限** |

**64 恰好撞线，故取 32**，留约 2× 安全余量。推算值与实测 0.50 MB 吻合。

吞吐上 32 并非最优（1024 维实测 32→20 t/s、256→40 t/s），但**可用性优先于吞吐**——被网关拦截则完全不可用。

若维度或网关额度变更，安全批次上限为 `floor(gateway_limit_bytes / (dimension * 4 / 3 * 1.02))`，建议再除以 2 作为余量。

### 4.5 重试

`ApiEmbedder` 内置：

| 参数 | 值 | 环境变量 |
|---|---|---|
| 单次超时 | 25s | `UXFLOW_EMBED_TIMEOUT` |
| 最大尝试次数 | 5 | `UXFLOW_EMBED_MAX_TRIES` |
| 重试间隔 | 固定 1s（非指数退避） | — |

**仅对超时、5xx、429 重试。4xx 立即抛出**——认证失败或请求格式错重试 5 次仍是同样的错，只会把秒级可诊断的问题拖成分钟级。

端到端验证（batch 512、1024 维、上述参数、6 轮）：6/6 成功，平均 23.85s/批，**21.5 texts/s**（含重试开销的真实吞吐）。

### 4.6 并发：客户端侧恒为 1

实测重试**不产生并发连接**。排除环境中 2 条无关连接后：

| 场景 | 峰值同时 ESTABLISHED | 累计开启套接字 |
|---|---|---|
| 强制重试（3s 超时 × 4 次） | **1** | 4 |
| 正常单次（60s 超时） | **1** | 1 |

httpx 超时时关闭旧 socket 再开新的，4 次重试是 4 条**先后**连接，任一时刻仅 1 条。滞留的是代理服务端的计算任务，不经过企业网关，不占网关连接数。

因此对受限网络而言，**风险点只有单请求体积（已由 base64 + batch 32 解决），并发不是风险**。设计不引入线程池，串行执行即天然满足并发为 1。

> 修正记录一：初版第 6 节称重试「会提高瞬时并发请求数」。该结论未经测量，实测为错。
>
> 修正记录二：排查中期曾报「峰值 3 条连接」。该数据无效——当时 `lsof` 解析有 bug，抓取的是进程打开的 .so 文件列表而非网络连接。


### 4.7 存量数据

新建独立数据库运行 API 后端；`~/.local/share/uxflow/uxflow.db`（1132 条 Qwen 向量，1024 维）原封保留。

必要性：向量空间与维度均不兼容，混用会使召回排序完全错乱。`sqlite_store.py:128` 的 `_check_dim` 会在维度不一致时报错，但**语义不一致（同为 1024 维的 Qwen vs OpenAI 向量）无法被自动检测**——这是必须换库而非仅依赖校验的原因。

### 4.8 阈值

本轮**不修改**默认值（`mount_threshold=0.50`、`UXFLOW_QUESTION_DEDUP_THRESHOLD=0.90`）。

实测 cosine 分布随维度下移：

| 维度 | mean | p50 | p90 | p99 |
|---|---|---|---|---|
| 1024 | 0.349 | 0.343 | 0.488 | 0.690 |
| 1536 | 0.339 | 0.332 | 0.475 | 0.682 |
| 3072 | 0.321 | 0.314 | 0.455 | 0.668 |

现有阈值标定于真实 Qwen 向量（见 `scripts/calibrate_mount_threshold.py`），换模型后失效。本轮补充脚本量取 API 向量下的分布并记录数据，**阈值调优另开一轮**，不阻塞本次合入。

#### API 向量下的实测分布（本轮采集，未据此改阈值）

用 `scripts/calibrate_thresholds_api.py`（`make_embedder()` 装配的当前后端，覆盖 `calibrate_mount_threshold.py` 的同一批 `PAIRS`，并额外交叉出 unrelated pairs 作为噪声基线）分别跑 `fake` 与 `api` 两个后端。

`fake` 后端（冒烟对照，预期无区分度）：

```
backend: FakeEmbedder  dimension=1024

-- true child→parent pairs --
cos=-0.021  child='修复运行时抛出的异常，如 TypeError/K' parent='从错误中恢复并修复问题'
cos=0.004  child='修复导入模块失败的问题' parent='从错误中恢复并修复问题'
cos=-0.033  child='为函数补充单元测试' parent='验证代码正确性'

-- unrelated child→wrong-parent pairs (noise floor) --
cos=0.022  child='修复运行时抛出的异常，如 TypeError/K' wrong_parent='验证代码正确性'
cos=0.008  child='修复导入模块失败的问题' wrong_parent='验证代码正确性'
cos=0.030  child='为函数补充单元测试' wrong_parent='从错误中恢复并修复问题'

true:   min=-0.033 max=0.004 mean=-0.017
noise:  min=0.008 max=0.030 mean=0.020

separation gap (true_min - noise_max) = -0.063
→ NO clean separation: at least one noise pair scores at or above the lowest true pair. There is no midpoint here that would mean anything as a threshold; more data (or a different measure) is needed before retuning.
```

哈希向量近正交，true/noise 无法区分——这是 fake 后端下的预期结果，不是 bug。

`api` 后端（`text-embedding-3-large`，3072 维，走真实端点）：

```
backend: ApiEmbedder  dimension=3072

-- true child→parent pairs --
cos=0.531  child='修复运行时抛出的异常，如 TypeError/K' parent='从错误中恢复并修复问题'
cos=0.537  child='修复导入模块失败的问题' parent='从错误中恢复并修复问题'
cos=0.409  child='为函数补充单元测试' parent='验证代码正确性'

-- unrelated child→wrong-parent pairs (noise floor) --
cos=0.233  child='修复运行时抛出的异常，如 TypeError/K' wrong_parent='验证代码正确性'
cos=0.206  child='修复导入模块失败的问题' wrong_parent='验证代码正确性'
cos=0.251  child='为函数补充单元测试' wrong_parent='从错误中恢复并修复问题'

true:   min=0.409 max=0.537 mean=0.492
noise:  min=0.206 max=0.251 mean=0.230

separation gap (true_min - noise_max) = 0.158
→ clean separation: every true pair (0.409) scores above every noise pair (0.251).
→ default mount_threshold=0.5 falls OUTSIDE the gap [0.251, 0.409] — does NOT reliably separate true pairs from noise under this backend.
```

**结论**：API 向量下 true/noise 有清晰分离（gap = [0.251, 0.409]），但当前默认 `mount_threshold=0.50` 落在这个区间**之外**（高于 `true_min=0.409`）——按当前默认值，本轮测得的最弱一条真实 child→parent 关系（`为函数补充单元测试` → `验证代码正确性`，cos=0.409）会被误判为不挂载。0.50 是否需要下调、下调到多少，留给阈值调优轮次决定；本节只记录数据，不改 `mount_threshold`。

## 5. 改动范围

| 文件 | 改动 |
|---|---|
| `src/uxflow_embed/protocol.py` | 增加可选 `preferred_batch_size` 声明 |
| `src/uxflow_embed/api.py` | 重试逻辑；base64 编解码；`preferred_batch_size=32`；默认 model/dimension |
| `src/uxflow_embed/local.py` | `preferred_batch_size=32` |
| `src/uxflow_embed/fake.py` | `preferred_batch_size=128` |
| `src/uxflow_runtime.py` | api 分支优先读 LITELLM 凭证；默认 large/3072；读取 batch/timeout/tries 环境变量 |
| `src/module1/pipeline.py` | `getattr` 读 chunk size |
| `.env.example` | 更新 api 后端说明；新增受限网络调优指引 |
| `README.md` / `README.zh-CN.md` | 嵌入后端章节 |
| `tests/uxflow_embed/test_api.py` | 重试行为、4xx 不重试、base64 解码、batch size |
| `tests/test_runtime_wiring.py` | 凭证回退顺序 |

`local` 后端行为完全不变（`preferred_batch_size=32` 与现 `_EMBED_CHUNK=32` 等价），该路径零回归。

## 6. 已知代价

1. **进度条粒度**：32 条一跳。与现有 `_EMBED_CHUNK=32` 一致，无变化。
2. **吞吐让位于可用性**：默认批次 32 而非吞吐最优的 256，是为受限网络的可用性做的让步。网关宽松的用户可经 `UXFLOW_EMBED_BATCH` 上调。
3. **重试增加代理服务端负载**：被放弃的请求仍在代理侧继续计算。该负载不经过企业网关（见 §4.6），但会消耗代理资源。

## 7. 未验证项（诚实标注）

- **出货配置吞吐已复测（5.7 texts/s，见 §1）**，替代了原先「21.5 texts/s 待复测」的标注。
  但该数字为单批次采样，代理负载会浮动；T14 的完整 ingest 会给出更可靠的均值。
- **base64 仅在 batch 32 与 128 下验证**：其他批次未单独测试。响应体大小可精确推算（§4.4），但延迟特性未逐一实测。
- **网关阈值为用户口述的约 1 MB**：未实地压测确认精确值，也未确认该限制作用于响应体、请求体还是两者。batch 32 留有 2× 余量，可容纳一定误差。
- **代理负载敏感**：所有吞吐数字测于特定时段，代理负载变化时会浮动。

## 8. 验收标准

1. `UXFLOW_EMBED_BACKEND=api` 且仅配置 `LITELLM_BASE`/`LITELLM_KEY` 时可正常启动
2. `ApiEmbedder` 返回 3072 维 L2 归一化向量
3. base64 与 float 两种响应形态均能正确解析（单元测试覆盖）
4. 超时/5xx/429 触发重试；4xx 立即抛出（单元测试覆盖）
5. `UXFLOW_EMBED_BATCH` / `UXFLOW_EMBED_TIMEOUT` / `UXFLOW_EMBED_MAX_TRIES` 生效
6. `local` / `fake` 后端行为与改动前逐字节一致
7. 全量测试通过
8. 用真实轨迹数据完成一次端到端 ingest，记录实际吞吐与响应体大小
