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

**方法**：`scripts/calibrate_thresholds_api.py`，后端由 `make_embedder()` 按 `UXFLOW_EMBED_BACKEND` 选择。真值取自 `fixtures/taxonomy_v0.json`（5 个 root、11 个 child，均为仓库内已确认的 child→parent 关系）。每个 label 以 `f"{label}: {description}"` 编码——root 的 description 极短，单独用它信息量不足。正样本 = 每个 child 对其声明的 parent（11 条）；负样本 = 每个 child 对其余 4 个非父 root（44 条）。全部 16 个 label 用**一次** `embed_batch` 取向量（端点约 30% 请求超时，16 次串行单调几乎必踩）。

`api` 后端（`text-embedding-3-large`，3072 维，真实端点）实测：

```
TRUE  n=11  min=0.288 p50=0.383 max=0.514
FALSE n=44  min=0.186 p50=0.317 max=0.467

overlap: true_min=0.288 vs false_max=0.467 -> OVERLAPPING
```

**两个分布重叠**，不是"阈值偏了一点"，而是这个相似度本身无法线性分开两类。具体反例：

| pair | cos | 判定 |
|---|---|---|
| `valid_syntax_in_toolcall → tool_use` | 0.467 | **错**（真父是 `code_generation`） |
| `requirement_analysis_before_coding → code_generation` | 0.466 | **错**（真父是 `planning`） |
| `self_verification → error_recovery` | 0.460 | **错**（真父是 `execution_control`） |
| `file_localization_and_edit → code_generation` | 0.288 | **对**，却是全场最低分 |

即：有 child 与错误 root 的相似度（0.467）显著高于另一些 child 与真父的相似度（0.288）。

阈值扫描（balanced accuracy = TPR 与 TNR 的均值；负样本是正样本的 4 倍，用普通 accuracy 会让"全部拒绝"也拿到 80%，故不用）：

```
 thresh  miss_true  wrong_mount  balanced_acc
  0.275          0           31         0.648
  0.350          3           11         0.739  <- 最优
  0.375          5            5         0.716
  0.500          9            0         0.591  <- 当前默认
```

**另测 argmax 上限**：`evolution.py:61-68` 实际是先对全部 root 取 argmax、再用阈值判断"是否挂载"——即阈值只决定挂不挂，挑哪个父由排序决定。所以上表 `wrong_mount` 高估了真实错挂率（负样本过线还须同时压过真父才会被真的挂错），而与决策真正相关的是排序准确率：

```
argmax-over-roots (ignoring any threshold): 6/11 children rank their true parent #1
  wrong: correct_shell_embedding: picked execution_control, true code_generation
  wrong: file_localization_and_edit: picked tool_use, true code_generation
  wrong: requirement_analysis_before_coding: picked code_generation, true planning
  wrong: reproduce_before_fix: picked error_recovery, true execution_control
  wrong: self_verification: picked error_recovery, true execution_control
```

**关键结论**：

1. 当前默认 `mount_threshold=0.50` 会漏掉 **9/11** 条真实 child→parent 关系（只有 `valid_syntax_in_toolcall`、`effective_error_fix`、`wellformed_tool_call` 三条接近或过线）。它并非"略高"，而是几乎让挂载失效。
2. balanced accuracy 最优点在 **0.35（0.739）**，但仍漏 **3/11** 真样本、错挂 **11/44** 负样本。它是重叠区里的最小损失折中，**不是解**。
3. **排序本身也不够**：即使完全去掉阈值、只取 argmax，也只有 **6/11** 命中真父。所以"改用相对排序替代绝对阈值"**不足以修好这个问题**——它把上限从"漏 9/11"提到"错 5/11"，仍有近半错误。这条要点比早前设想的更严重：瓶颈在向量本身对这批标签的区分力，不只在判据形式。
4. 因此这是**设计信号，不是调参问题**。可能的方向（均超出"更换 embedding 后端"这一轮范围，且第一条已知不足以单独奏效）：
   - 给 root 更**丰富的 embedding 文本**（现有 root description 只有"代码生成相关能力"这类 6~8 字，信息量过低；`file_localization_and_edit` 被判给 `tool_use`、`self_verification` 被判给 `error_recovery` 都是这种信息不足的典型表现）；
   - 用 **LLM 裁决**做最终判定，embedding 仅作 prefilter（给定 6/11 的排序上限，这条看起来是必需项而非可选项）；
   - 相对排序（argmax / top-k）可作为组合项之一，但由上第 3 点，**不能单独依赖**。
5. 本轮**不改** `mount_threshold`，`src/module0_5/evolution.py` 未改动。本节只记录证据，调优/改判据另开一轮。

数值有轻微跑动（同一批文本重跑，个别 cosine 在 ±0.002 内浮动，如 `requirement_completeness→planning` 0.383/0.387、`0.375` 行 `wrong_mount` 4/5 互换），不影响上述任何结论的方向。

`fake` 后端为冒烟对照：哈希向量近正交，TRUE `min=-0.084 p50=-0.009 max=0.044`、FALSE `min=-0.052 p50=0.002 max=0.082`，全部 cosine 贴近 0，任何阈值下 balanced_acc 恒为 0.500。这是 fake 的**预期结果，不是 bug**——它只验证脚本能跑通。

> **勘误**：本小节的早期草稿曾用 3 条手写 pair 测出 `separation gap = 0.158` 并称 "clean separation"，结论是 0.50 只需下移。该数字是**样本过易造成的假象**（手写 pair 的正负样本语义距离被人为拉大），已被上面基于 `taxonomy_v0.json` 全部 11 条真值的测量推翻。git 历史中仍可见那个 0.158，**不可采信**。

### 4.9 空文本切片

真实 ingest 中，`dataset/swe-chat-reasoning-sample-50.jsonl` 产出的 1099 条切片里有 **19 条 embedding 文本为空串**。`build_embedding_text()`（`src/module1/signature.py`）对 steps 中既无 `tool_call_args`、又无 assistant `content`、也无 `tool_result` 的切片，`" ".join([])` 得到 `""`。

**这是既有问题，非本次迁移引入**：`git show origin/main:src/module1/signature.py` 中 `_build_summary_for_embedding` 完全相同。它一直没暴露，是因为本地 Qwen 模型会照单全收地嵌入空串（现有库 1132 条 signature 无一条 embedding 为 NULL）。API 后端则直接拒绝：

```
400 Bad Request: Invalid 'input[15]': input cannot be an empty string.
```

而 **400 是刻意不重试的**（见 §4.5：认证失败或请求格式错重试 5 次仍是同样的错），于是单条空切片就会终止整场 ingest。

**处理**：在 `ApiEmbedder.embed_batch` 内解决，而非改调用方——持有 API 契约的是 embedder，不该让每个调用方都知道"这个后端不收空串"。请求前剔除空/纯空白文本，只把非空的发出去；若整批皆空则完全跳过 HTTP 调用。返回时按原始下标重组，**每条输入恰好对应一个输出**，空的那些填零向量。

零向量是正确的哨兵值：`recall_core.vector_score`（`src/module1/recall_core.py:110`）本就以 `if sig.embedding and any(v != 0.0 for v in sig.embedding)` 过滤候选，零向量在下游已被当作"无可用 embedding"跳过。零向量也**不做归一化**——`_ensure_dimension` 要除以模长，模长为 0 会得到全 NaN 并污染所有余弦比较（该函数以 `if norm > 0` 守卫）。

> **已知缺口，未修复**：更深的问题是 `build_embedding_text` 对某些切片压根产不出文本，这些切片无论有没有零向量兜底，**向量检索都召不回它们**。零向量只是让 ingest 不再崩，不等于这些切片被正确索引了。修 `build_embedding_text`（例如回退到 `tool_call_name`、user content 或切片元信息）会改变被嵌入的文本、进而改变每一条向量，是独立的一轮决策，本轮不做。

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
