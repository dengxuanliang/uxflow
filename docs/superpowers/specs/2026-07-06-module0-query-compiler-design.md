# 模块 0：问题清单编译（Query Compiler）—— 设计 Spec

> 后续在此文件上逐步审核与增量更新。

---

## 1. 项目目标与模块职责

### 1.1 项目整体目标

从海量回流的 SWE 多轮对话轨迹中，筛出能作为 SFT 数据的高价值样本，使模型 A 训练后能解决用户反馈的问题清单。

全局约束条件：

- 输入只有自然语言描述的问题清单，无伴随的失败轨迹样本
- 无 test case，不投入维护程序化检测器
- 轨迹超长（几万~十几万 token），格式为消息列表 + 其他结构
- 允许截取轨迹中的一段子轨迹作为最终 SFT 样本
- 轨迹中存在环境返回的执行结果（stdout/stderr/traceback/exit code），可作为"自报信号"免费使用

### 1.2 模块 0 的职责定位

管线第一步"查询理解层"：把一句模糊的自然语言抱怨，编译成结构化、可检索、可验证的 Problem Spec，直接驱动下游召回 + 质量过滤。

关键认知——**一次翻转**：查询描述的是"失败"，要检索的是"正确能力"，中间需要翻转。每个子问题编译为「失败签名（挖反例/建验证集）」+「能力签名（检索正例）」一对。

### 1.3 未来增强项（本 spec 暂不展开，留 TODO）

以下模块 0 相关缺口在本 spec 中不设计，仅登记待后续审核：

- **TODO：反例集 / 验证集组装** —— `trajectory_signal` 与切片侧失败信号目前生成后无消费者，缺一个把失败信号组装成带标签反例集的模块。
- **TODO：打标质量 / 一致性评估** —— 缺 gold set 与打标器准确率/跨 run 一致性回归。
- **TODO：盲区 / 覆盖登记** —— 前端视觉类因无硬信号成批 drop，缺显式登记与方向性决策。
- **TODO：词表治理量化触发** —— 模块 0.5 定期清理缺触发指标（命中频次、稀疏阈值、近重复率）。

---

## 2. 各 Part 设计

模块 0 内部分为 6 个 Part + 1 个条件触发的 Part 6.5。Part 1 由 Call 1 独立完成；Part 2–6 合并进 Call 2 一次批量调用；Part 6.5 仅在存在歧义型 drop 时触发。

### Part 1：子问题分解（Sub-problem Decomposition）

| 项目 | 说明 |
|------|------|
| **目标** | 将一句可能包含多个子问题的自然语言，拆成互不重叠的原子子问题 |
| **输入** | 用户原始描述 |
| **输出** | 子问题列表，每条含 `id`、`raw_text`（原始片段）、`failure_summary`（一句话标准化描述） |
| **是否调用 LLM** | ✅ 是，单次调用（Call 1），structured output |
| **分解规则** | 按逗号/分号/语义转折切分；合并重复；每条必须是单一失败模式 |
| **边界处理** | 整句只描述一个问题则输出单条；无法确定是否该拆时保守不拆，标 `needs_review: true` |

### Part 2：能力标签映射（Capability Taxonomy Mapping）

| 项目 | 说明 |
|------|------|
| **目标** | 为每个子问题打上能力标签——"问题"与"轨迹签名"对齐的唯一桥梁 |
| **输入** | Part 1 的 `failure_summary` |
| **输出** | `target_capability`：1–3 个能力标签 |
| **是否调用 LLM** | ✅ 是，合并进 Call 2 |

**冷启动决策：从空词表开始（不预建 taxonomy）。** 第一版不预建任何 taxonomy，初始词表为空，标签由 LLM 在第一批问题上自由提议，跑完人工整理为 v0 taxonomy，后续按模块 0.5 机制演化。

Part 2 行为随词表状态动态切换：

| 词表状态 | LLM 行为 | prompt 注入 |
|---------|---------|------------|
| 空（冷启动） | 完全自由提议标签（附 description + parent 建议） | 不注入词表 |
| 有词表（N > 0） | 优先从已有词表选取；确无匹配才提议新标签 | 注入当前词表树 |

**标签命名规则**：小写英文 + 下划线，动宾结构，描述"正确做法"而非"错误现象"（如 `correct_shell_embedding`）；每个新标签附一句话中文 `description` + `parent` 建议（顶层则 `parent: null`）；新标签标 `taxonomy_extension: true` 自动入库。

**一致性保障**：轨迹离线签名阶段用**完全相同**的词表 + 相同映射 prompt，确保两端对齐。

### Part 3：轨迹自报信号描述（Trajectory Self-Report Signal）

| 项目 | 说明 |
|------|------|
| **目标** | 描述"该失败在真实轨迹中会留下什么可观察痕迹"，用于下游召回辅助匹配 |
| **输入** | `failure_summary` + `target_capability` |
| **输出** | `trajectory_signal`：一段自然语言规则描述，描述在轨迹 JSON 中应匹配什么模式 |
| **是否调用 LLM** | ✅ 是，合并进 Call 2 |
| **设计原则** | 不写代码检测器，而是描述"grep/匹配什么"，利用环境已返回的真实执行结果 |
| **双向用途** | 匹配该信号的轨迹 = 反例候选；不匹配该信号且能力标签命中 = 正例候选 |

### Part 4：HyDE 正例生成（Hypothetical Positive Example）

| 项目 | 说明 |
|------|------|
| **目标** | 生成"正确做法长什么样"的假设性轨迹片段，作为向量召回查询锚（解决"embedding 抱怨会召回同样出错的轨迹"） |
| **输入** | `failure_summary` + `target_capability` |
| **输出** | `hyde_positive`：2–3 段 200–500 token 假设性轨迹片段变体 |
| **是否调用 LLM** | ✅ 是，合并进 Call 2 |
| **生成要求** | 1) 与真实轨迹格式一致（tool_call → observation → reasoning）；2) 体现 target_capability 的正确行为；3) 含成功信号（exit code 0） |

### Part 5：关键词与结构化过滤条件（Keywords & Structured Filters）

| 项目 | 说明 |
|------|------|
| **目标** | 为 BM25 通道和元数据字段过滤提供精确匹配条件 |
| **输入** | `failure_summary` + `target_capability` |
| **输出** | `keywords`（字符串列表）+ `structured_filters`（字段条件 dict） |
| **是否调用 LLM** | ✅ 是，合并进 Call 2 |
| **keywords 内容** | 报错关键词（`SyntaxError`、`EOF`）、工具名、API/库名、语言名 |
| **structured_filters 内容** | 对轨迹签名字段的过滤条件，如 `languages`、`tools_used`、`min_turns`、`has_verification_step`（字段名与枚举值以接口契约为准） |

### Part 6：置信度与分流（Confidence & Routing）

| 项目 | 说明 |
|------|------|
| **目标** | 判定每个子问题编译结果是否可信，决定通过/筛除，并对筛除者标注原因 |
| **输入** | 每个子问题 Part 1–5 的全部字段 |
| **输出** | `confidence`（0–1）+ `route`（pass / drop）+ `drop_reason`（仅 route==drop 时填写） |
| **是否调用 LLM** | ✅ 是，合并进 Call 2（自评维度全是 Call 2 的直接产出，无需回看） |

**置信度评估维度**：子问题是否有歧义（高权重）、能力标签是否唯一命中（中）、trajectory_signal 是否可操作（中）、HyDE 变体是否一致（低）。

**分流规则**：

| confidence | route | 行为 |
|------------|-------|------|
| ≥ 0.8 | `pass` | 通过，进入 Problem Spec |
| < 0.8 | `drop` | 筛除，并按下表标注 `drop_reason` |

**`drop_reason` 枚举**：

| drop_reason | 含义 | 是否触发 Part 6.5 澄清 |
|---|---|---|
| `ambiguous` | 子问题语义有多种解释未区分（如"中途停止"= 被截断/主动收尾/遇错放弃） | ✅ 触发 |
| `not_applicable` | 本质不适用（前端视觉无硬信号、环境/基础设施非模型能力） | ❌ 直接丢弃 |
| `label_diverged` | 命中多个不相关标签、问题未聚焦 | ❌ 直接丢弃 |
| `other` | 其他低分原因 | ❌ 直接丢弃 |

### Part 6.5：歧义澄清与 drop 救回（条件触发）

> **核心约束（务必遵守）**：本 Part 仅对 `drop_reason == "ambiguous"` 的子问题触发，且**至多执行一次额外澄清调用，绝不递归**。澄清后产出的子问题无论 confidence 高低，都不会再次进入本 Part。

| 项目 | 说明 |
|------|------|
| **目标** | 把因语义歧义被 drop、但本可救回的子问题，拆成多个被消歧的清晰查询，而非直接丢弃 |
| **触发闸门** | 仅 `route == "drop"` 且 `drop_reason == "ambiguous"` 的子问题进入；其余 drop 一律丢弃 |
| **是否调用 LLM** | ✅ 是。条件触发；对本批所有 ambiguous 子问题一次 batch 调用（Call 3），随后一次重评（Call 2'） |
| **输入** | 每条 ambiguous 子问题的 `raw_text` + `failure_summary` + Call 2 已生成字段 |
| **动作（Call 3）** | 让 LLM 列出该子问题所有合理解释（互斥消歧），每条解释产出一条独立的被澄清新子问题（新 id，如 p12 → p12a/p12b/p12c） |
| **消歧后处理（Call 2'）** | 每条被澄清子问题重走 Call 2 全套字段生成（打标 + 自报信号 + HyDE + keywords + 置信度自评）；这次不再输出 `drop_reason`，也不再触发 Part 6.5 |
| **回收规则** | 重评后 confidence ≥ 0.8 的解释 → 进入 Problem Spec；仍 < 0.8 → 最终丢弃 |
| **防膨胀** | 单条子问题最多拆 K 条解释（默认 K=4）；澄清仅做一轮，不递归 |
| **可追溯** | 救回的子问题带 `origin: "clarified"` 和 `parent_id`（如 p12a 的 parent_id=p12） |

**p12 走查示例**：`p12 "多步任务未完成即终止"` confidence=0.70, drop_reason=ambiguous → Call 3 拆为 p12a（被 max_turns 截断）/ p12b（主动收尾误判为完成）/ p12c（遇错放弃）→ 各自重走 Call 2' → p12a、p12c 抬到 ≥0.8 进 Spec，p12b 若仍模糊则丢弃。三条均不再触发第二次澄清。

---

## 3. LLM 调用汇总与执行流程

### 3.1 调用汇总

| 调用 | 触发条件 | 覆盖 Parts | 输入 | 输出 |
|------|---------|-----------|------|------|
| **Call 1** | 必定 | Part 1 | 用户原始描述 | 子问题列表（id, raw_text, failure_summary） |
| **Call 2** | 必定 | Part 2–6 | 所有子问题 batch | 每条的 target_capability, trajectory_signal, hyde_positive, keywords, structured_filters, confidence, route, drop_reason |
| **Call 3** | 仅当存在 `drop_reason=="ambiguous"` | Part 6.5 拆解 | 所有 ambiguous 子问题 batch | 每条拆为 ≤K 条被消歧的新子问题（含 failure_summary、parent_id） |
| **Call 2'** | 紧跟 Call 3 | Part 2–6（重评） | Call 3 产出的新子问题 batch | 同 Call 2 的字段，但**不输出 drop_reason，且不再触发 Call 3** |

**关键约束**：Call 3 + Call 2' 是原子对，**至多执行一次，不递归**。

**典型场景调用次数**：

- 无歧义 drop：**2 次**（Call 1 + Call 2）
- 有歧义 drop：**4 次**（Call 1 + Call 2 + Call 3 + Call 2'）
- 上限恒为 4 次

### 3.2 执行流程

```
用户原始描述
    │
    ▼
[Call 1] 子问题分解
    │
    ▼ (N 条子问题)
[Call 2] 批量打标 + 置信度自评 + drop_reason
    │
    ├── route==pass ──────────────────────► Problem Spec
    ├── route==drop, reason≠ambiguous ───► 丢弃（终态）
    └── route==drop, reason==ambiguous ──► 进入 Part 6.5
                │
                ▼
        [Call 3] 歧义澄清（一次，不递归）
                │ 每条拆 ≤K 条消歧子问题
                ▼
        [Call 2'] 消歧子问题重评（同 Call 2 格式，不再输出 drop_reason）
                │
                ├── confidence ≥ 0.8 ──► Problem Spec
                └── confidence < 0.8 ──► 丢弃（终态，不再触发澄清）
```

---

## 4. 产出格式与验证方式

### 4.1 Problem Spec 产出格式

Problem Spec 结构与字段类型以**接口契约**为准（`raw_input` / `domain` / `sub_problems[]`）。字段约定：

- 每条子问题必带 `origin`：`"original"`（Call 2 直接通过）或 `"clarified"`（经 Part 6.5 澄清救回）
- `origin == "clarified"` 的子问题必带 `parent_id`，指向被拆解的原始子问题 id；`origin == "original"` 时 `parent_id` 为 null
- Call 2 阶段被 drop 的子问题带 `drop_reason`（不进入 `sub_problems[]`，单独存储供审计）

```json
{
  "raw_input": "...解题过程中途停止...",
  "domain": "agentic_swe",
  "sub_problems": [
    {
      "id": "p6",
      "origin": "original",
      "parent_id": null,
      "raw_text": "写入py文件有语法错误",
      "failure_summary": "写入 py 文件语法错误",
      "target_capability": ["valid_syntax_in_toolcall"],
      "trajectory_signal": "observation 含 SyntaxError 且 tool_call 参数含 python 代码",
      "hyde_positive": ["<假设正例: 工具调用中正确编写 python，observation 正常返回>"],
      "keywords": ["SyntaxError", "import", "python"],
      "structured_filters": {"languages": ["python"]},
      "confidence": 0.90,
      "route": "pass"
    },
    {
      "id": "p12a",
      "origin": "clarified",
      "parent_id": "p12",
      "raw_text": "解题过程中途停止",
      "failure_summary": "多步任务因命中 max_turns 被截断而未完成",
      "target_capability": ["persist_through_truncation"],
      "trajectory_signal": "末轮命中 max_turns / 最后一步为未返回的 tool_call",
      "hyde_positive": ["<假设正例: 检测到接近 max_turns 时主动总结进度并完成收尾>"],
      "keywords": ["max_turns", "truncated", "incomplete"],
      "structured_filters": {"min_turns": 5},
      "confidence": 0.84,
      "route": "pass"
    }
  ]
}
```

> 上例含两条 pass 子问题：p6 是 Call 2 直接通过（`origin: "original"`, `parent_id: null`）；p12a 是经澄清救回（`origin: "clarified"`, `parent_id: "p12"`）。被 drop 的子问题（如 p5，`drop_reason: "not_applicable"`）不出现在 `sub_problems[]` 中。

### 4.2 验证方式

1. 拿 3–5 条不同模糊度的真实问题，跑完整流程，检查 Problem Spec 各字段完整、可操作
2. HyDE 正例 embedding 召回 vs 直接 embedding 抱怨的召回，对比相关性
3. 故意给模糊输入，验证是否被正确 drop 且 `drop_reason` 分类准确
4. 验证 ambiguous 型 drop 能触发澄清、消歧后 confidence 抬升、且澄清**严格只跑一次**（用 p12"中途停止"做基准 case）

---

## 5. 标签系统演化（Taxonomy Evolution）

### 5.1 问题定位

能力 taxonomy 是"问题"与"轨迹签名"对齐的唯一桥梁，但它会随新问题持续生长。核心矛盾：新标签出现后历史轨迹没有该标签会漏召回；全量重打标签成本不可接受（百万切片 × LLM）。

这是"本体演化（Ontology Evolution）"问题。

### 5.2 方案：增量追加 + 延迟回填 + 层级继承

新标签审批策略：**自动入库 + 定期合并清理**（非人工 gate）。入库前做 embedding 去重（cosine > 0.85 视为重复，映射到已有标签）。

#### 新标签挂载规则

| 场景 | 处理方式 |
|------|---------|
| 能挂到已有父节点（cosine ≥ 0.6） | 挂到相似度最高的已有父节点下，作为新叶子 |
| 挂不到任何已有父节点（所有父节点 cosine < 0.6） | 创建**新顶层父节点** + 新叶子标签一起入库（taxonomy 是森林，不是单棵树） |

新顶层节点的创建频率很低（大部分 SWE 能力可归入已有大类）。定期清理时再检查是否可合并到已有树。

#### 继承降权机制

**解决的问题**：新标签刚入库还没回填时，不能让查询返回空结果。

**机制**：查询某个子标签时，系统**同时将标了其父节点的切片也纳入候选**，但排序时权重打折。

| 切片标签情况 | 处理 | relevance_score 乘数 |
|------------|------|---------------------|
| 精确标了新子标签（回填后才有） | 精确匹配 | ×1.0 |
| 只标了父标签（通过继承进入候选） | 弱匹配 | ×0.3 |
| 标了无关标签 | 不进入候选 | — |

**降权而非等权的原因**：标了 `code_generation` 的切片可能是正则相关（有教学价值），也可能是 API 调用相关（无关）。继承只保证"不断档"，不保证精确——必须降权，让向量/BM25 召回分数主导排序。回填完成后，确实相关的切片会被补上精确标签 → ×1.0。**继承是回填完成前的过渡，不是最终状态。**

#### 查询三路融合

一次针对某能力标签的查询，系统同时从三个来源捞候选切片，合并排序：

| 路 | 怎么捞 | 靠什么字段 | 加权 | 何时失效 |
|---|--------|-----------|------|---------|
| 第一路：精确标签 | 找 `capability_labels` 精确标了该叶子的切片 | 标签字段（回填后才有） | ×1.0 | 回填未完成时为空 |
| 第二路：父标签继承 | 找标了父节点的切片当弱相关 | 标签字段（父标签） | ×0.3 | 父节点是全新顶层、无切片标过时为空 |
| 第三路：向量 fallback | 拿查询的 HyDE embedding 对所有切片 embedding 做 ANN | 切片免费层 `embedding`，**完全不看标签** | 按相似度 | 几乎不失效 |

**向量 fallback 的兜底作用**：前两路失效（如新顶层标签空窗期），第三路保证结果不为空——降级但非开天窗。回填完成后第一路精确命中接管。

**ANN 性能说明**：系统用 HNSW（对数级图搜索），非暴力 KNN（线性级）。1000 万切片上单次查询仍是毫秒级。再叠加 filtered search（先 ES 结构化过滤缩域到几十万再 ANN），查询不在性能瓶颈。耗时的是离线建索引和内存（1536 维 × 1000 万 ≈ 60GB 常驻），不是查询本身。

#### 完整演化生命周期

| 环节 | 做法 |
|------|------|
| 新标签入库 | 指定 `parent`（已有节点）或标 `new_root: true`（创建新顶层）。入库即生效，无需等回填 |
| 即时覆盖（继承） | 已标父标签的切片，对新子标签查询自动成为候选（×0.3 降权），零重打 |
| 异步回填 | 用新标签 `description_embedding` + `keywords` 做粗召回（全量 1-5% 候选），只对候选集 LLM batch 精判，命中者追加精确标签（×1.0） |
| 回填优先级 | 按新标签关联的问题数量排序，高频优先 |
| 历史 Problem Spec | 不需重跑。老问题下次查询时其 HyDE + keywords 在混合召回中自然命中新标签轨迹 |
| 定期清理 | 合并近义标签、降级过稀疏标签为父标签 keyword、检查新顶层是否可合并到已有树 |

**一句话**：新标签即时挂树继承（不断档）→ 后台只对小范围候选重打标签（成本可控）→ 定期合并清理（不膨胀）。全量重刷永远不发生。

### 5.3 异步回填（Targeted Backfill）详细流程

> **关键澄清**：回填的候选池来自切片的**免费层签名**（embedding + bm25_tokens），**不依赖切片是否已打过任何能力标签**。因此"新顶层 + 新叶子"这个 case 对回填流程零影响。

```
触发条件：新标签入库
         │
         ▼
Step 1：构造回填查询（零 LLM）
         │
         ▼
Step 2：粗筛候选（用已有索引，零 LLM）
         │
         ▼
Step 3：LLM 精判（只对粗筛候选，batch 调用）
         │
         ▼
Step 4：写回 capability_labels（更新 ES）
```

**Step 1：构造回填查询（零 LLM）**

每个标签入库时自带元信息（模块 0 Call 2 生成时一起产出）：

```json
{
  "label": "correct_regex_escaping",
  "parent": "code_generation",
  "description": "在工具调用中正确编写正则表达式，转义符使用正确",
  "keywords": ["regex", "re.compile", "re.search", "\\\\", "escape", "backslash"],
  "description_embedding": [/* 1536-d */]
}
```

从中组装查询条件（纯规则）：BM25 用 `keywords`；向量用 `description_embedding`；结构化过滤从标签语义推断（如 `languages=python`、`min_turns` 等）。

**Step 2：粗筛候选（零 LLM）**

| 通道 | 动作 | 缩小幅度 |
|------|------|---------|
| ES 结构化过滤 | `languages` + `tools_used` + `min_turns` 等 | 1000 万 → ~200 万 |
| Qdrant 向量召回 | `description_embedding` 在过滤子集内 ANN，top-30000 | 200 万 → 3 万 |
| ES BM25 召回 | `keywords` 在过滤子集内检索，top-30000 | 200 万 → 3 万 |
| RRF 融合去重 | 两路合并 | → 约 3-5 万候选 |

零 LLM，纯走已有索引，秒级完成。

**Step 3：LLM 精判（成本集中在此）**

对 3-5 万候选，batch 调用 LLM 判"该切片是否演示了该能力"。每次 20-50 条切片，structured output 布尔数组。

成本估算（3 万候选）：~600 次调用、~200 万 token、用快速便宜模型。只在新标签入库时跑一次，非每次查询。

**Step 4：写回**

判 true 的切片在 ES 追加 `capability_labels`。Qdrant 不需更新（标签存 ES）。

#### 时间线

| 时间点 | 发生什么 | 查询体验 |
|--------|---------|---------|
| T=0 | 新标签入库 | 继承降权（×0.3）+ 向量 fallback 兜底 |
| T+数秒 | 粗筛完成 | 无变化（后台） |
| T+数小时 | 精判 + 写回完成 | 精确命中（×1.0），体验质变 |

回填全程不阻塞查询。

### 5.4 新顶层标签的回填走查（correct_markdown 示例）

**场景**：用户反馈"markdown 格式经常破损"，词表无匹配叶子、无匹配父节点（cosine 均 < 0.6）→ 创建新顶层 `output_formatting` + 新叶子 `correct_markdown`。

**继承在此 case 的表现**：`output_formatting` 是全新父节点，无切片标过 → 继承路径为空，查询仅靠向量 fallback（降级但非空）。

**回填流程（与挂老父节点完全一致）**：

| 步骤 | 用到的字段 | 是否碰 capability_labels |
|------|-----------|------------------------|
| Step 1 构造查询 | 标签的 `keywords`、`description_embedding` | ❌ |
| Step 2 粗筛 | 切片免费层 `embedding`、`bm25_tokens`、结构化字段 | ❌ |
| Step 3 LLM 精判 | 切片 `context_header` + `bm25_tokens` | ❌ |
| Step 4 写回 | — | ✅ 此时才写入 |

**对比**：

| | 挂老父节点 | 新顶层 |
|---|---|---|
| 空窗期兜底 | 继承降权（×0.3）| 向量 fallback |
| 回填流程 | 完全相同 | 完全相同 |
| 回填候选池来源 | 免费层签名 | 免费层签名 |

**结论**：回填不依赖标签体系，依赖切片的免费层签名——新顶层和老父节点走完全一样的路。
