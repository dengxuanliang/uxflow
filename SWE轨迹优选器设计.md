# SWE 轨迹 SFT 数据优选器 —— 整体设计

## Context

设计一个多轮对话 SWE 轨迹数据的优选器，目的是从海量回流轨迹中筛选出能作为 SFT 数据的高价值样本，使得模型 A 训练后能解决用户反馈的问题清单。

全局约束条件：
- 输入只有自然语言描述的问题清单，无伴随的失败轨迹样本
- 无 test case，不投入维护程序化检测器
- 轨迹超长（几万~十几万 token），格式为消息列表 + 其他结构
- 保留完整的轨迹，但是允许将轨迹中的一段子轨迹loss mask设置为1，其余置为0进行SFT训练
- 但轨迹中存在环境返回的执行结果（stdout/stderr/traceback/exit code），可作为"自报信号"免费使用

---

## 模块全景与状态

| 序号 | 模块 | 核心问题 |
|------|------|---------|
| 0 | 问题清单编译（Query Compiler） | 查询理解 |
| 0.5 | 标签系统演化（Taxonomy Evolution） | taxonomy 增量更新，不全量重刷 | 
| 1 | 轨迹离线处理（切分 + 切片签名 + 索引） | 超长轨迹怎么切、怎么打签名、怎么建索引 | 
| 2 | 召回+排序+筛选| 混合召回 + 打分 + 过滤，一条管线跑完 | 
| 3 | 集合优选（必须独立） | 去重 + 覆盖度/多样性 + 配比 | 

---

# 模块 0：问题清单编译（Query Compiler Module）

## Context

本模块是管线第一步："查询理解层"——将用户自然语言问题清单编译为结构化、可检索、可验证的 Problem Spec。

关键认知：查询描述的是"失败"，要检索的是"正确能力"，中间需要一次翻转——每个子问题编译为「失败签名（挖反例/建验证集）」+「能力签名（检索正例）」一对。

---


## 模块总览

| 层次 | 职责 | 输入 | 输出 |
|------|------|------|------|
| 整体 | 将一句模糊的自然语言抱怨，编译为结构化的、可直接驱动下游召回+质量过滤的 Problem Spec | 一段用户自然语言描述 | 完整 Problem Spec JSON |

---

## 各部分设计

### Part 1：子问题分解（Sub-problem Decomposition）

| 项目 | 说明 |
|------|------|
| **目标** | 将一句可能包含多个子问题的自然语言，拆成互不重叠的原子子问题 |
| **输入** | 用户原始描述，如"解题过程中途停止，工具调用import中python代码非法，bash command中python代码的闭合有问题" |
| **输出** | 子问题列表，每条包含 `id`、`raw_text`（原始片段）、`failure_summary`（一句话标准化描述） |
| **是否调用 LLM** | ✅ 是。单次 LLM 调用，structured output |
| **Prompt 要点** | system prompt 包含分解规则：按逗号/分号/语义转折切分；合并重复；每条必须是单一失败模式；输出 JSON array |
| **边界处理** | 若整句只描述一个问题，输出单条；若无法确定是否该拆，保守不拆，标 `needs_review: true` |
| **输出示例** | `[{id:"p1", raw:"解题过程中途停止", failure_summary:"模型在多步任务中未完成即终止"}, ...]` |

---

### Part 2：能力标签映射（Capability Taxonomy Mapping）

| 项目 | 说明 |
|------|------|
| **目标** | 为每个子问题打上能力标签——这是"问题"与"轨迹签名"对齐的唯一桥梁 |
| **输入** | Part 1 产出的 `failure_summary` |
| **输出** | `target_capability`：1-3 个能力标签 |
| **是否调用 LLM** | ✅ 是。与 Part 3/4/5/6 合并为一次 Call 2 调用（batch 所有子问题） |

**冷启动决策：从空词表开始（不预建 taxonomy）**

第一版**不预建**任何 taxonomy，初始词表为空。标签由 LLM 在第一批问题上自由提议，第一批跑完人工花 10 分钟归类整理 = v0 taxonomy，后续按模块 0.5 的 B+D 机制自然演化。这比预先设计 30-50 叶子树更快、更贴合真实问题分布。

因此 Part 2 的行为是**动态的**，取决于词表状态：

| 词表状态 | LLM 行为 | prompt 注入 |
|---------|---------|------------|
| 空（冷启动） | 完全自由提议标签（附 description + parent 建议） | 不注入词表 |
| 有词表（N > 0） | 优先从已有词表选取；确无匹配才提议新标签 | 注入当前词表树 |

**标签命名规则**（两种状态通用）：
- 小写英文 + 下划线，动宾结构，描述"正确做法"而非"错误现象"（如 `correct_shell_embedding`、`task_decomposition`）
- 每个新标签附带一句话中文 `description`
- 每个新标签附带 `parent` 建议：从已提议/已有标签中选上层分类；若本身是顶层则 `parent: null`
- 新标签标 `taxonomy_extension: true`，自动入库（见模块 0.5）

演化方向：词表越丰富，LLM 新建标签频率越低，系统从"开放标注"渐进收敛为"分类任务"。

**词表结构示意**（人工整理后的形态，冷启动时为空）：
```
code_generation
  ├── valid_syntax_in_toolcall        # 工具调用中生成合法代码
  ├── correct_shell_embedding         # bash 中正确嵌入其他语言
  └── idiomatic_api_usage             # 使用正确的 API/库
execution_control
  ├── long_horizon_persistence        # 长程任务坚持完成
  ├── error_recovery                  # 遇错后恢复而非放弃
  └── verified_closure                # 验证通过后才收尾
planning
  ├── task_decomposition              # 正确拆分子任务
  ├── tool_selection                  # 选择合适工具
  └── context_tracking                # 跨轮次上下文保持
```

| **一致性保障** | 轨迹离线签名阶段用**完全相同**的词表 + 相同的映射 prompt，确保两端对齐 |

---

### Part 3：轨迹自报信号描述（Trajectory Self-Report Signal）

| 项目 | 说明 |
|------|------|
| **目标** | 描述"该失败在真实轨迹中会留下什么可观察痕迹"，用于下游召回时辅助匹配（第一版不单独建验证集） |
| **输入** | `failure_summary` + `target_capability` |
| **输出** | `trajectory_signal`：一段自然语言规则描述，描述在轨迹 JSON 中应匹配什么模式 |
| **是否调用 LLM** | ✅ 是。与 Part 2/4/5 合并为同一次调用 |
| **设计原则** | 不写代码检测器，而是描述"在轨迹中 grep/匹配什么"——利用环境已返回的真实执行结果 |

| 问题性质 | 信号来源 | 示例描述 |
|----------|---------|------|
| 单步输出错误 | 工具调用后紧跟的 observation 中的 stderr/traceback/exit code | "observation 含 `SyntaxError` 且 tool_call 参数含 `import`" |
| 轨迹形状问题 | 轨迹结构特征：总轮数、最后一步类型、有无验证步骤 | "末轮为 assistant 无 final_answer / 命中 max_turns 截断" |
| 解题策略问题 | 推理过程中的决策模式 | "连续 3 步使用同一工具且均失败，未切换策略" |

| **精度定位** | 对单步错误类精度接近 oracle（是真实执行结果）；对轨迹形状/策略类是强代理信号 |
| **双向用途** | 匹配该信号的轨迹 = 反例（验证集）；**不**匹配该信号且能力标签命中 = 正例候选 |

---

### Part 4：HyDE 正例生成（Hypothetical Positive Example）

| 项目 | 说明 |
|------|------|
| **目标** | 生成"正确做法长什么样"的假设性轨迹片段，作为向量召回的查询锚（解决"embedding 抱怨会召回同样出错的轨迹"的问题） |
| **输入** | `failure_summary` + `target_capability` |
| **输出** | `hyde_positive`：2-3 段 200-500 token 的假设性轨迹片段变体 |
| **是否调用 LLM** | ✅ 是。与 Part 2/3/5 合并为同一次调用 |
| **生成要求** | 1) 和真实轨迹格式一致（tool_call → observation → reasoning 结构）；2) 体现 target_capability 的正确行为；3) 包含成功信号（observation 正常返回、exit code 0） |
| **为什么要多变体** | 不同场景/语言/工具组合的正例，多路召回取并集，提高覆盖率 |
| **向量化方式** | 对每段 hyde_positive 做 embedding，作为正例召回的查询向量 |

---

### Part 5：关键词与结构化过滤条件（Keywords & Structured Filters）

| 项目 | 说明 |
|------|------|
| **目标** | 为 BM25 通道和元数据字段过滤提供精确匹配条件（向量召回的互补） |
| **输入** | `failure_summary` + `target_capability` |
| **输出** | `keywords`（字符串列表）+ `structured_filters`（字段条件 dict） |
| **是否调用 LLM** | ✅ 是。与 Part 2/3/4 合并为同一次调用 |
| **keywords 内容** | 报错关键词（`SyntaxError`、`EOF`）、工具名（`bash`、`python`）、API/库名、语言名 |
| **structured_filters 内容** | 对轨迹签名字段的过滤条件，如 `languages`、`tools_used`、`outcome`、`min_turns` |
| **与向量的关系** | BM25 召回与向量召回并行，用 RRF 融合；structured_filters 作为硬条件叠加在两路之上 |

**输出示例：**
```json
{
  "keywords": ["import", "SyntaxError", "Traceback", "tool_call"],
  "structured_filters": {
    "languages": ["python"],
    "tools_used": ["code_interpreter", "bash"],
    "outcome": "solved",
    "min_turns": 3
  }
}
```

---

### Part 6：置信度与分流（Confidence & Routing）

| 项目 | 说明 |
|------|------|
| **目标** | 判定每个子问题的编译结果是否可信，决定通过还是筛除 |
| **输入** | 每个子问题 Part 1-5 的全部已生成字段 |
| **输出** | `confidence`（0-1）+ `route`（pass / drop） |
| **是否调用 LLM** | ✅ 是，但**合并进 Call 2**（优化①）。置信度评估的维度全是 Call 2 的直接产出属性，LLM 在生成其他字段的同时即可自评，不需要"回看"，无需独立调用 |

| 维度 | 权重 | 说明 |
|------|------|------|
| 子问题本身是否有歧义 | 高 | "停止"可能有 3 种含义 |
| 能力标签是否唯一命中 | 中 | 映射到多个不相关标签说明不清晰 |
| 轨迹自报信号是否可操作 | 中 | 能写出具体匹配模式 vs 只能模糊描述 |
| HyDE 正例变体之间是否一致 | 低 | 变体描述的场景差异过大说明问题没聚焦 |

**分流规则（第一版简化）：**

| confidence | route | 行为 |
|------------|-------|------|
| ≥ 0.8 | `pass` | 通过，进入下游召回 |
| < 0.8 | `drop` | 筛除，不进入下游（第一版不做 expand/clarify，保持管线简单） |

---

## 模块产出：Problem Spec（编译终点）

模块 0 到置信度分流为止结束。通过（confidence ≥ 0.8）的子问题组成最终 Problem Spec，交付下游召回模块。
---

## LLM 调用汇总

| 调用序号 | 覆盖 Parts | 输入 | 输出 | 说明 |
|---------|-----------|------|------|------|
| **Call 1** | Part 1 | 用户原始描述 | 子问题列表（id, raw_text, failure_summary） | 单独调用，因为后续 Parts 依赖其产出 |
| **Call 2** | Part 2 + 3 + 4 + 5 + 6 | 所有子问题的 failure_summary（batch） | 每条子问题的 target_capability, trajectory_signal, hyde_positive, keywords, structured_filters, confidence, route | 合并为一次调用（含置信度自评）；用 structured output 保证格式 |

**总计：2 次 LLM 调用 / 每条问题清单输入。**（优化①：置信度评估维度全是 Call 2 的直接产出属性，无需回看，合并进 Call 2）

---

## Prompt 模板

### Call 1 — 子问题分解

```
System:
你是一个问题分解器。用户会给你一段自然语言描述的模型问题反馈，可能包含多个子问题。

规则：
1. 按逗号、分号、句号、语义转折切分为原子子问题
2. 每条必须是单一失败模式，不能混合多个问题
3. 如果两个描述是同一个问题的不同表述，合并为一条
4. 如果整句只描述一个问题，输出单条
5. 如果无法确定是否该拆，保守不拆，标 needs_review: true
6. failure_summary 用一句话标准化描述失败模式，去除具体场景细节

输出 JSON array，每条包含 id, raw_text, failure_summary。

User:
{用户原始描述}
```

### Call 2 — 批量生成 + 置信度自评

`{词表注入}` 是动态部分，随词表状态切换（见 Part 2 冷启动决策）。

```
System:
你是一个 SWE 轨迹数据优选器的查询编译器。对于每个子问题，你需要生成以下字段：

1. target_capability: 为该子问题打 1-3 个能力标签。
   - 标签命名：小写英文 + 下划线，动宾结构，描述"正确做法"而非"错误现象"
     （示例：correct_shell_embedding, task_decomposition, error_recovery）
   - 新标签附带一句话中文 description
   - 新标签附带 parent 建议：从已提议/已有标签中选上层分类；若本身是顶层则 parent: null
   - 新标签标 taxonomy_extension: true
   {词表注入}
2. trajectory_signal: 描述该失败在真实 agentic 轨迹中会留下什么可观察痕迹（可被 grep/匹配的模式）。
3. hyde_positive: 生成 2-3 段假设性轨迹片段（200-500 token），模拟"正确处理该问题"的 assistant 行为。
   格式需和真实轨迹一致（tool_call → observation → reasoning），并包含成功信号（exit code 0、observation 正常返回）。
4. keywords: 用于 BM25 召回的关键词列表（报错关键词、工具名、API 名、语言名等）。
5. structured_filters: 对轨迹签名字段的过滤条件（languages, tools_used, outcome_transition, min_turns 等）。
6. confidence: 0-1 置信度。评估维度：子问题是否有歧义、标签是否唯一命中、
   trajectory_signal 是否可操作、hyde_positive 变体是否一致。
7. route: confidence >= 0.8 为 "pass"，否则为 "drop"。

输出 structured JSON array。

User:
子问题列表：
{Call 1 的输出}
```

**`{词表注入}` 的两种取值：**

```
# 空词表（冷启动）时注入：
   - 当前词表为空，请自由提议标签。后续会有人工整理归并。

# 有词表（N > 0）时注入：
   - 当前能力词表如下，优先从中选取最匹配的标签（直接写 label 即可）。
     仅在确实无匹配时才提议新标签（需附带 description 和 parent）。
     {词表树形结构}
```

---

## 模块执行流程

```
用户原始描述
    │
    ▼
[Call 1] 子问题分解 ─── LLM 调用 #1
    │
    ▼ (N 条子问题)
    │
[Call 2] 批量生成 + 置信度自评 ─── LLM 调用 #2（一次性产出以下所有字段）
    │  ├─► 能力标签 (target_capability)
    │  ├─► 轨迹自报信号 (trajectory_signal)
    │  ├─► HyDE 正例 (hyde_positive × 2-3 变体)
    │  ├─► 关键词 + 结构化过滤 (keywords, structured_filters)
    │  └─► 置信度 + 分流 (confidence, route)
    │
    ▼ 按 route 分流（第一版简化）
    │
    ├─── confidence ≥ 0.8 ──► pass → 进入下游召回
    └─── confidence < 0.8 ──► drop → 筛除
              │
              ▼
输出 Problem Spec（仅含 pass 的子问题）→ 交付下游召回模块
```

---

## 完整产出格式（Problem Spec）

```json
{
  "raw_input": "解题过程中途停止，工具调用import中python代码非法，bash command中python代码的闭合有问题",
  "domain": "agentic_swe",
  "sub_problems": [
    {
      "id": "p1",
      "raw_text": "解题过程中途停止",
      "failure_summary": "模型在多步任务中未完成即终止",
      "target_capability": ["long_horizon_persistence", "verified_closure"],
      "trajectory_signal": "末轮为 assistant 且无 final_answer 标记；或命中 max_turns；或最后一步为未返回的 tool_call；或无验证/确认步骤",
      "hyde_positive": [
        "<假设正例片段1: 遇阻后重试并最终完成>",
        "<假设正例片段2: 验证通过后明确收尾>"
      ],
      "keywords": ["max_turns", "truncated", "incomplete"],
      "structured_filters": {"outcome": "solved", "has_verification_step": true, "min_turns": 5},
      "confidence": 0.55,
      "route": "drop"
    },
    {
      "id": "p2",
      "raw_text": "工具调用import中python代码非法",
      "failure_summary": "工具调用参数中的 python 代码（含 import）存在语法错误",
      "target_capability": ["valid_syntax_in_toolcall"],
      "trajectory_signal": "tool_call 参数含 python 代码（含 import 语句），且紧跟的 observation 含 SyntaxError/ImportError traceback",
      "hyde_positive": [
        "<假设正例片段: 工具调用中正确编写含 import 的 python，observation 返回正常执行结果>"
      ],
      "keywords": ["import", "SyntaxError", "ImportError", "Traceback", "tool_call"],
      "structured_filters": {"languages": ["python"], "tools_used": ["code_interpreter"], "outcome": "solved"},
      "confidence": 0.9,
      "route": "pass"
    },
    {
      "id": "p3",
      "raw_text": "bash command中python代码的闭合有问题",
      "failure_summary": "bash 工具调用中嵌入的 python 代码存在引号/heredoc 闭合错误",
      "target_capability": ["correct_shell_embedding"],
      "trajectory_signal": "bash tool_call 参数含 python -c 或 heredoc，且 observation 含 unexpected EOF / unterminated quote / 非零 exit code",
      "hyde_positive": [
        "<假设正例片段: bash 中用正确转义或 heredoc 嵌入 python 并成功执行>"
      ],
      "keywords": ["python -c", "heredoc", "EOF", "unterminated", "bash", "exit code"],
      "structured_filters": {"languages": ["python"], "tools_used": ["bash"], "outcome": "solved"},
      "confidence": 0.85,
      "route": "pass"
    }
  ]
}
```

> 注：上例中 p1（中途停止）因语义歧义 confidence=0.55 < 0.8 被 `drop`；p2/p3 confidence ≥ 0.8 通过。第一版只保留 pass 的子问题进入下游。

---

## 验证方式

1. 拿 3-5 条真实用户问题描述（不同模糊度），跑一遍 2 次 LLM 调用流程，检查产出 Problem Spec 各字段是否完整、可操作
2. HyDE 正例 embedding 后，在小规模轨迹样本上召回结果 vs 直接 embedding 抱怨的召回结果，对比相关性
3. 置信度分流的合理性：故意给几条模糊输入，验证模糊问题是否被正确 drop（confidence < 0.8）

---

# 模块 0.5：标签系统演化（Taxonomy Evolution）

## Context

能力 taxonomy 是"问题"与"轨迹签名"对齐的唯一桥梁，但它会随新问题持续生长。核心矛盾：新标签出现后历史轨迹没有该标签会漏召回；全量重打标签成本不可接受（百万切片 × LLM）。

定性：这是"本体演化（Ontology Evolution）"问题。

## 推荐方案：（增量追加 + 延迟回填）+ （层级继承）组合

新标签审批策略：**自动入库 + 定期合并清理**（非人工 gate）。入库前做 embedding 去重（cosine > 0.85 视为重复，映射到已有标签）。

### 新标签挂载规则

| 场景 | 处理方式 |
|------|---------|
| 能挂到已有父节点（cosine ≥ 0.6） | 挂到相似度最高的已有父节点下，作为新叶子 |
| 挂不到任何已有父节点（所有父节点 cosine < 0.6） | 创建**新顶层父节点** + 新叶子标签一起入库（taxonomy 是森林，不是单棵树） |

示例——当出现"markdown 格式破损"这种全新能力类型时：
```
code_generation
  └── ...
execution_control
  └── ...
planning
  └── ...
output_formatting          ← 新建顶层父节点（与已有父节点 cosine 均 < 0.6）
  └── correct_markdown     ← 新叶子标签
```

新顶层节点的创建频率很低（大部分 SWE 能力可归入已有大类）。定期清理时再检查：新顶层是否能合并到已有树。

### 继承降权机制详解

**"继承降权"解决的问题**：新标签刚入库还没回填时，不能让查询返回空结果。

**机制**：查询某个子标签时，系统**同时将标了其父节点的切片也纳入候选**，但排序时权重打折。

| 切片标签情况 | 对新标签查询的处理 | relevance_score 乘数 |
|------------|------------------|---------------------|
| 精确标了新子标签（回填后才有） | 精确匹配 | ×1.0 |
| 只标了父标签（通过继承进入候选） | 弱匹配——"可能相关" | ×0.3 |
| 标了无关标签 | 不进入候选 | — |

**为什么降权而不是等权**：标了 `code_generation` 的切片可能是"正则转义"相关（有教学价值），也可能是"API 调用"相关（无关）。继承只保证"不断档"，不保证精确——所以必须降权，让向量/BM25 的召回分数主导排序。

**回填完成后**：原来通过继承进入候选的切片，如果确实相关，会被回填补上精确标签 → 下次查询变为 ×1.0。继承是**回填完成前的过渡**，不是最终状态。

| 环节 | 做法 |
|------|------|
| 新标签入库 | 指定 `parent`（已有节点）或标 `new_root: true`（创建新顶层）。入库即生效，无需等回填 |
| 即时覆盖（D 继承） | 已标父标签的切片，对新子标签查询自动成为候选（×0.3 降权弱匹配），零重打 |
| 异步回填（B） | 用新标签描述 embedding + keywords 做粗召回（全量的 1-5% 候选），只对候选集重跑签名，精细化为强匹配（×1.0） |
| 回填优先级 | 按新标签关联的问题数量排序，高频优先 |
| 查询时融合 | 精确标签切片（×1.0）+ 父标签继承（×0.3）+ 向量 fallback，三路融合排序 |
| 历史 Problem Spec | 不需重跑。老问题下次查询时其 HyDE + keywords 在混合召回中自然命中新标签轨迹 |
| 定期清理 | 每周/月人工或 LLM 辅助 review：合并近义标签、降级过稀疏标签（命中 < 阈值）为父标签的 keyword、检查新顶层节点是否可合并到已有树 |

**一句话总结**：新标签即时挂树继承（不断档）→ 后台只对小范围候选重打标签（成本可控）→ 定期合并清理（不膨胀）。全量重刷永远不发生。

### 异步回填（Targeted Backfill）详细流程

回填要解决的问题：新标签入库后，库里可能已有切片演示了该能力，但其 `capability_labels` 字段缺少这个标签。回填 = **找到这些切片，补上标签**。

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

#### Step 1：构造回填查询（零 LLM）

每个标签入库时自带元信息（来自模块 0 Call 2 生成时一起产出）：

```json
{
  "label": "correct_regex_escaping",
  "parent": "code_generation",
  "description": "在工具调用中正确编写正则表达式，转义符使用正确",
  "keywords": ["regex", "re.compile", "re.search", "\\\\", "escape", "backslash", "pattern"],
  "description_embedding": [/* 1536-d, 对 description 做 embedding */]
}
```

从中组装回填查询条件（纯规则，不调 LLM）：
- BM25 查询：用 `keywords`
- 向量查询：用 `description_embedding`
- 结构化过滤：从标签语义推断（如 `languages` 含 python、`outcome_transition` 含 success）

#### Step 2：粗筛候选（零 LLM，用已有索引）

用 Step 1 的查询条件在 ES + Qdrant 上检索（和正常查询走同样的三件套）：

| 通道 | 动作 | 缩小幅度 |
|------|------|---------|
| ES 结构化过滤 | `outcome_transition` 含 success + `languages` 含 python | 1000 万切片 → ~200 万 |
| Qdrant 向量召回 | 用 `description_embedding` 在过滤后子集内 ANN，取 top-30000 | 200 万 → 3 万 |
| ES BM25 召回 | 用 `keywords` 在过滤后子集内检索，取 top-30000 | 200 万 → 3 万 |
| RRF 融合去重 | 两路合并 | → 约 3-5 万候选 |

这一步**零 LLM 调用**，纯走已有索引，秒级完成。

#### Step 3：LLM 精判（成本集中在这里，batch 调用）

对 Step 2 的 3-5 万候选，用 LLM 判断"这个切片是否演示了该能力"。

**为什么不能跳过**：粗筛只保证"语义或关键词相关"，不能确认"切片确实正确演示了该能力"。

**输入**：每个切片的 `context_header` + `bm25_tokens`（已有字段，不需要读全文）

**batch 方式**：每次 20-50 个切片一次调用，structured output 为布尔数组

```
Prompt 示例：

判断以下切片是否演示了 "correct_regex_escaping" 能力。
标签定义：在工具调用中正确编写正则表达式，转义符使用正确。
对每个切片输出 true/false。

切片列表：
1. context_header: "任务: 解析日志文件..." bm25_tokens: ["re.compile", "\\\\d+", "bash"]
2. context_header: "任务: 替换字符串..." bm25_tokens: ["regex", "re.sub", "python"]
...

输出：[true, false, ...]
```

**成本估算**（以 3 万候选为例）：
- 每 batch 50 个切片 → 600 次 LLM 调用
- 每次输入约 3000 token，输出约 200 token
- 总计约 200 万 token，用快速便宜模型即可
- 只在新标签入库时跑一次，非每次查询

#### Step 4：写回 capability_labels（更新 ES）

Step 3 判定为 true 的切片，在 ES 中追加 `capability_labels`：

```
对每个 judged_true 的 slice_id：
    ES update: capability_labels 字段追加 "correct_regex_escaping"
```

Qdrant 不需要更新（`capability_labels` 存在 ES 中，不在向量库里）。

#### 整体时间线

| 时间点 | 发生什么 | 查询体验 |
|--------|---------|---------|
| T=0 | 新标签入库，触发回填任务 | 查询走父标签继承（×0.3 降权），有结果但不精确 |
| T ~ T+数秒 | Step 1+2 粗筛完成 | 无变化（后台） |
| T+数分钟 ~ T+数小时 | Step 3 LLM 精判完成 | 无变化（后台，不阻塞查询） |
| T+数小时 | Step 4 写回完成 | 查询精确命中（×1.0），体验质变 |

回填全程**不阻塞任何查询**——回填前走继承降权兜底，回填后自动升级为精确匹配。

### 完整生命周期示例（以 correct_regex_escaping 为例）

1. **新问题进来**：用户反馈"正则转义经常错"，Part 2 在词表找不到精确叶子标签 → LLM 提议 `correct_regex_escaping`
2. **去重检查**：拿 `correct_regex_escaping` embedding 和已有叶子标签比较，cosine 均 < 0.85 → 不是重复，允许创建
3. **挂载判定**：和已有父节点比较，与 `code_generation` 的 cosine = 0.72（≥ 0.6）→ 挂到 `code_generation` 下
4. **自动入库（D 生效）**：继承立刻生效，所有标为 `code_generation` 的切片对 `correct_regex_escaping` 查询自动成为候选（×0.3 降权）
5. **Targeted Backfill（B）**：keywords `["regex", "re.compile", "\\\\", "escape"]` + 标签描述 embedding 粗筛（100万→3万）→ 加结构化过滤 `outcome=solved, languages=python`（→8000）→ 只对 8000 条重跑 LLM 签名（只判一个布尔"是否演示了正确正则转义"）→ 命中的（如 1200 条）追加精确标签
6. **回填后**：精确命中 1200 条排序 ×1.0，父标签继承的降权 ×0.3 排后
7. **老问题受益**：老 Problem Spec 不需更新，其 HyDE + keywords 在混合召回中自然命中新标签轨迹
8. **定期清理**：合并近义标签（如后来自动入库的 `regex_backslash_handling` → 合并到 `correct_regex_escaping`），降级过稀疏标签为 keyword


### end of module 0.5


### 冷启动真实样例（第一批 12 条问题清单）

以下是空词表冷启动的真实跑通样例：输入 12 条分号分隔的用户问题，经 Call 1 分解为 18 个原子子问题，Call 2 打标（空词表，标签全部为 LLM 新提议，批内复用去重）。

**输入问题清单：**
```
webui前端对齐有问题，按钮没有突出核心按钮，5分钟的休息机制没有按要求实现；
前端按钮点击后没有反应；上传bug显示代理阻止；
写入py文件有语法错误，逐行写入但文件中缩进错误；
工具解析有错误，tool_call的标签在thinking里面；
python实现五子棋，玩家胜利后没有提示，人工提示修正后仍然没有实现；
复杂问题理解不全面就开始实现，导致实现逻辑错误，需要人为提示才能发现问题；
解题过程中途停止，import后代码非法，bash command中python代码引导闭合有问题；
未经复现直接修改；连续重复相同操作；文件定位与编辑失败；多轮修复效果不佳
```

**Call 1 分解结果：18 个原子子问题**（其中问题 4 拆为 语法错/缩进错 2 条；问题 6 拆为 缺功能/修复无效 2 条；问题 8 拆为 中途停止/import 语法/bash 闭合 3 条）

**Call 2 打标 + 分流结果：**

| id | failure_summary | target_capability | parent | confidence | route |
|----|-----------------|-------------------|--------|-----------|-------|
| p1 | 前端 UI 元素对齐不正确 | correct_ui_alignment | frontend_development | 0.62 | drop |
| p2 | 未突出主操作按钮 | emphasize_primary_action | frontend_development | 0.55 | drop |
| p3 | 需求明确功能点未实现 | requirement_completeness | planning | 0.82 | pass |
| p4 | 前端交互事件未生效 | functional_ui_interaction | frontend_development | 0.75 | drop |
| p5 | 上传被代理拦截 | resolve_upload_restriction | environment_handling | 0.40 | drop |
| p6 | 写入 py 文件语法错误 | valid_syntax_in_toolcall | code_generation | 0.90 | pass |
| p7 | 逐行写入缩进错误 | correct_indentation | code_generation | 0.85 | pass |
| p8 | tool_call 错位到 thinking | wellformed_tool_call | tool_use | 0.88 | pass |
| p9 | 需求明确功能点未实现 | requirement_completeness | planning | 0.82 | pass |
| p10 | 用户指出后修复未生效 | effective_error_fix | error_recovery | 0.85 | pass |
| p11 | 未理解需求就编码且无法自主发现 | requirement_analysis_before_coding, self_verification | planning / execution_control | 0.85 | pass |
| p12 | 多步任务未完成即终止 | long_horizon_persistence | execution_control | 0.70 | drop |
| p13 | import 相关 python 语法错误 | valid_syntax_in_toolcall | code_generation | 0.90 | pass |
| p14 | bash 嵌入 python 闭合错误 | correct_shell_embedding | code_generation | 0.85 | pass |
| p15 | 未复现就直接修改 | reproduce_before_fix | execution_control | 0.82 | pass |
| p16 | 陷入重复动作循环 | avoid_redundant_repetition | execution_control | 0.80 | pass |
| p17 | 无法定位文件或编辑失败 | file_localization_and_edit | code_generation | 0.82 | pass |
| p18 | 多轮修复仍未解决 | effective_error_fix | error_recovery | 0.80 | pass |

**批内标签复用（去重合并）：** `requirement_completeness`（p3+p9）、`valid_syntax_in_toolcall`（p6+p13）、`effective_error_fix`（p10+p18）。

**被 drop 的 4 条及原因：**
- p1/p2（对齐/视觉突出）：视觉主观问题，SWE 轨迹无可靠硬信号，难找正例
- p4（按钮无反应）：借线 0.75，前端功能 bug 在纯 SWE 轨迹库覆盖度不确定
- p5（上传被代理阻止）：环境/基础设施问题，非模型能力，最低 0.40
- p12（中途停止）：歧义——被截断/主动收尾/遇错放弃三种含义未区分

**人工归类后的 v0 taxonomy 草稿**（⚠️ = drop 的标签，仅记录不入正式词表）：

```
frontend_development                          ⚠️ 整个分支暂缓（视觉/前端硬信号弱）
  ├── correct_ui_alignment                    ⚠️ drop
  ├── emphasize_primary_action                ⚠️ drop
  └── functional_ui_interaction               ⚠️ drop
environment_handling                          ⚠️ 非模型能力
  └── resolve_upload_restriction              ⚠️ drop
code_generation
  ├── valid_syntax_in_toolcall                # p6, p13
  ├── correct_indentation                     # p7
  ├── correct_shell_embedding                 # p14
  └── file_localization_and_edit              # p17
tool_use
  └── wellformed_tool_call                    # p8
planning
  ├── requirement_completeness                # p3, p9
  └── requirement_analysis_before_coding      # p11
execution_control
  ├── long_horizon_persistence                ⚠️ p12 drop（歧义待澄清）
  ├── reproduce_before_fix                    # p15
  ├── avoid_redundant_repetition              # p16
  └── self_verification                       # p11
error_recovery
  └── effective_error_fix                     # p10, p18
```

**进入下游召回的正式 v0 词表（仅 pass，5 顶层 + 12 叶子）：**

```
code_generation
  ├── valid_syntax_in_toolcall
  ├── correct_indentation
  ├── correct_shell_embedding
  └── file_localization_and_edit
tool_use
  └── wellformed_tool_call
planning
  ├── requirement_completeness
  └── requirement_analysis_before_coding
execution_control
  ├── reproduce_before_fix
  ├── avoid_redundant_repetition
  └── self_verification
error_recovery
  └── effective_error_fix
```

**两个机制层面的观察：**
1. **前端类问题（p1/p2/p4）成批 drop** 揭示机制边界——"纯 SWE 轨迹 + 执行自报信号"对前端/视觉类问题天然覆盖不足（不产生 traceback/exit code 硬信号）。是否纳入前端是方向性决策，非打标错误。
2. **p12（中途停止）因歧义反复被 drop**——值得单独做一次澄清，把它从 drop 救回成可用查询。

---

# 模块 1：轨迹离线处理（切分 + 切片签名 + 索引）

## Context

超长轨迹（几万~十几万 token）需切分。切片本身就是 SFT 样本候选单位（允许截取子轨迹）。切分不用 LLM（开销大、遵从行不准）。

## 三层表示

| 层 | 粒度 | 用途 |
|---|---|---|
| 轨迹签名 | 整条 1 个 | 结构化过滤、能力标签粗匹配、BM25 粗筛 |
| 子轨迹切片 | 每条 N 个（3-20） | 向量召回单位、最终 SFT 样本候选单位 |
| 切片签名 | 每切片 1 个 | 切片级标签 + 质量信号 + embedding |

查询时：切片级向量召回 → 命中切片 → 回溯母轨迹 → 决定最终 SFT 样本边界（切片本身或扩展到相邻切片）。

## Part 1.1：切分方法 —— 结构感知规则切分（零 LLM）

不用固定轮次（会切断"工具调用→观察→推理"闭环，轮次长度极不均匀）。用 agentic 轨迹天然的免费结构边界：

| 边界信号 | 来源 | 含义 |
|---------|------|------|
| 执行结果状态翻转 | observation 的 exit code / traceback | 失败→成功 = 修复闭环结束 |
| 工具类型切换 | tool_call 的 tool 名 | 搜索/读取 → 编辑/执行 = 阶段转换 |
| 新子任务信号 | assistant 规划性语言（正则/关键词，非 LLM） | "现在处理下一个..." |
| 长 reasoning 块 | 连续无 tool_call 的长文本 | 阶段总结/重新规划点 |

算法：解析事件序列 → 在候选边界标记 → 切（约束：每片 token ∈ [500, 8000]（优化③：上限从 4000 放宽到 8000，避免强行切断完整修复闭环，现代 SFT context ≥ 8k 可训；如实测长切片过多致训练效率下降再收回）；太小向后合并，太大在次级边界再切；不跨越未闭合 tool_call）。

## Part 1.2：上下文处理 —— 离线摘要 + 在线扩展

| 时刻 | 做什么 | 机制 |
|------|--------|------|
| 离线建索引 | 切片头拼 50-150 token 压缩摘要（由前序切片签名字段拼接，非 LLM），让切片自带语境 | 方案 A |
| 生成 SFT 样本 | 以命中切片为核心按切片类型做规则化向前/向后扩展 | 方案 C（受控版） |

扩展规则（纯规则）：错误恢复类 → 必须向前纳入引发错误的切片；收尾类/独立子任务 → 不扩展；兜底不超过 max token。边界决策放最后而非离线定死，因为同一切片对不同问题最佳边界不同。

## Part 1.3：切片签名 —— 两层成本结构

惰性打标签（已确认）：免费层全量离线建好；LLM 层只在切片被召回/回填时现打并缓存。LLM 调用从百万级降到每查询几千级。

### 免费层（结构签名）—— 所有切片，零 LLM

| 组 | 字段 | 来源 / 用途 |
|----|------|------|
| A 身份定位 | slice_id, parent_trajectory_id, index, prev_id, next_id, start_turn, end_turn, token_count, turn_count | 主键、回溯、边界扩展找相邻、过滤 |
| B 行为特征 | tools_used, languages（pygments/正则）, has_code_in_toolcall | 结构化过滤 |
| C 结果/质量自报信号（金矿） | exit_codes, error_types（正则抽 SyntaxError 等）, outcome_transition（failed→success / success_only / failed_only / no_execution）, num_retries, ends_with_unreturned_toolcall | 成败判定、精确匹配失败模式（接近 oracle）、premature_termination 信号 |
| D 切片类型（由 C 推导，驱动扩展） | slice_type ∈ {planning, attempt, error_recovery, verification, cleanup_summary, ambiguous} | 边界扩展规则；ambiguous 留给惰性 LLM 层 |
| E 检索表示 | embedding（[context_header + 正文]）, bm25_tokens（含 API/报错/工具名标识符）, context_header | 向量召回、BM25 召回、自带语境 |

### LLM 层（语义签名）—— 惰性，只打召回候选集

| 字段 | 何时打 | 说明 |
|------|--------|------|
| capability_labels | 切片被召回 / taxonomy 回填时 | 映射统一能力词表（与 Problem Spec 同词表），精确匹配的桥；打完缓存回写 |
| slice_type（仅 ambiguous） | 同上 | 补规则判不准的类型 |

> **优化④**：第一版不做实时逐条打标签，改为 **batch 调用**——一次查询的千级候选收集后，每 batch 含 20-50 个切片的摘要（context_header + bm25_tokens），LLM 批量判定 capability_labels，structured output 为数组。将"千次串行"变为"几十次 batch"，查询延迟大幅降低。打完缓存回写，下次同切片免打。
>
> **优化⑥**：第一版不打 `demo_quality`（"是否干净演示正确行为"的软质量分）。因为第一版去掉了质量门，没有消费者。减少 LLM prompt 长度和 token 消耗。第二版加回质量门时恢复。

### 能力标签：切片级 vs 轨迹级

| | 轨迹级 | 切片级 |
|---|---|---|
| 粒度 | 粗 | 细 |
| 何时打 | 离线（轨迹数远少于切片数，成本可接受） | 惰性，召回后 |
| 用途 | BM25 粗筛、结构化过滤（粗定位） | 精排精确能力匹配（精定位） |

### 切片签名结构示意

```json
{
  "slice_id": "traj_8821#s3",
  "parent_trajectory_id": "traj_8821",
  "index": 3, "prev_id": "traj_8821#s2", "next_id": "traj_8821#s4",
  "start_turn": 14, "end_turn": 22,
  "token_count": 2100, "turn_count": 8,
  "tools_used": ["bash", "code_interpreter"],
  "languages": ["python"],
  "has_code_in_toolcall": true,
  "exit_codes": [1, 1, 0],
  "error_types": ["SyntaxError"],
  "outcome_transition": "failed→success",
  "num_retries": 2,
  "ends_with_unreturned_toolcall": false,
  "slice_type": "error_recovery",
  "context_header": "任务: 修复 CSV 解析器。前序: 首次修改触发 SyntaxError。本片: 分析并修复。",
  "embedding": [/* 1536-d */],
  "bm25_tokens": ["SyntaxError", "import", "csv", "bash"],
  "capability_labels": null,
  "demo_quality": null
}
```

## Part 1.4：向量索引与混合召回（已定稿）

### 决策 1：索引架构 —— 分库（已确认）

切片 embedding 和轨迹签名分库。理由：结构化过滤条件是**轨迹级**的，检索单位是**切片级**的，粒度错位；分库让结构化存储负责"先缩小候选池"，向量库职责单一只做 ANN。

### 决策 2：向量索引选型 —— HNSW（已确认）

规模估算：100 万轨迹 × 平均 10 切片 = 1000 万切片向量。1536 维 float32 ≈ 6KB/条 ≈ 60GB 裸向量。HNSW 精度高（recall@10 > 95%）、增量插入友好（配合惰性打标签），内存够即可。规模再涨 10 倍或内存紧张时切 IVF-PQ（压缩 4-16x）。使用向量库的 filtered search（给定候选 ID 集合内搜索）。

### 决策 3：parent-doc 回溯 —— docstore 模式（业界最常用）

small-to-big 范式（LangChain ParentDocumentRetriever / LlamaIndex RecursiveRetriever）。三件套存储：

| 层 | 选择 | 存什么 |
|----|------|--------|
| 向量库（子块索引） | Qdrant / Milvus（HNSW） | 切片 embedding + 最小 payload（slice_id, parent_trajectory_id, prev_id, next_id） |
| 结构化存储（过滤 + BM25） | Elasticsearch | 轨迹级签名全字段（bool query 过滤）+ 切片 bm25_tokens（关键词召回） |
| docstore（原文回溯） | 对象存储（S3/OSS）或 PostgreSQL | 切片原文、母轨迹原文，按 key 直接取 |

关键：**ES 一身二用**——既做结构化过滤层（决策 1 候选缩小），又做 BM25 召回引擎，省一个组件。回溯 = 一次 key lookup（命中 slice_id → docstore 取原文；扩展 → prev_id/next_id 再取）。

### 模块 1 完整查询时序

```
Problem Spec
    │
    ▼
① ES 结构化过滤（structured_filters）
   → 候选轨迹 ID 集合（万级）→ 展开为候选切片 ID 集合（十万级）
    │
    ├──────────────────────┬──────────────────────┐
    ▼                      ▼                       │
② Qdrant 向量召回         ③ ES BM25 召回           │
   HyDE embedding          keywords                │
   filtered search        （限定候选切片内）        │
   （候选切片 ID 内）       top-k                    │
   top-k                                            │
    │                      │                        │
    └──────────┬───────────┘                        │
               ▼                                     │
        ④ RRF 融合 → 召回切片集（千级）               │
               │                                     │
               ▼                                     │
        ⑤ docstore 回溯（S3/PG）← prev_id/next_id ───┘
           取命中切片原文 + 相邻切片（为边界扩展备料）
               │
               ▼
        交付模块 2（召回+排序+筛选）
```

## 待续（模块 1 剩余）

- 切分边界 case（轨迹中途格式异常、超长单轮）—— 工程细节，实现时处理

---

# 模块 2：召回 + 排序 + 筛选（第一版合并模块）

## Context

模块 1 交付的是一批召回切片（千级）+ 原文 + 相邻切片。本模块把千级候选排序、过滤到最终能进 SFT 的几十~几百条。

关键背景：
- **轨迹来自更强模型的 SWE 数据，和模型 A 无关**——不需要 rejection sampling，质量基线高
- **回流数据已过上游质量筛选**（格式正确、内容完整有保证）

## 第一版决策：模块 2 = 纯相关性精排（单闸门）

> **为什么去掉质量门（闸门 2）和教学价值门（闸门 3）**：
> - 质量门：上游已保证格式正确+内容完整，来源又是更强模型，LLM-judge 的过滤力极低（只能区分"正确但拖沓"vs"正确且干净"，属偏好优选非质量过滤）。第一版不引入，避免增加变量、干扰因果归因。
> - 教学价值门（learnability）：未经验证的先验代理；第一版若用它筛"验证筛选器的数据"会混淆"筛选器好"与"代理好"。挂起为第二版增强项。
>
> **结果**：模块 0（查询编译）+ 模块 1（索引召回）+ 模块 2（相关性过滤）整体 = 一个干净的"语义检索系统"。质量由上游保证，教学价值由下游 SFT 实验 + 验证集事后实测验证。因果归因清晰：**SFT 若有效，说明"相关性检索"核心假设成立**。

```
召回切片集（千级）
    │
    ▼
闸门 1：相关性
   ① 触发惰性能力标签打标（LLM batch，仅千级候选，可缓存）
   ② 标签匹配：软加分而非硬过滤（优化⑤）
      - 有标签交集：relevance_score 维持原分
      - 无标签交集：relevance_score × 0.3（衰减，不直接砍掉）
   ③ 按 relevance_score 排序，取 top-N 进模块 3
    │
    ▼
最终候选（百级）→ 交付模块 3 集合优选

final_score = relevance_score = RRF 融合分 ×（标签命中则 ×1，未命中则 ×0.3）
```

> **优化⑤理由**：标签交集作为唯一闸门做硬二值过滤，风险在于 LLM 打标签有噪声（不可能 100% 准确），会误杀"实际相关但恰好标签没打对"的切片。改为软加分后，向量/BM25 召回本身已保证语义相关性，标签只是精修信号——即使标签打错，切片仍有机会凭召回分数留下来。

## 实验将指示下一步加什么（缺口定位）

| SFT 实验结果 | 诊断 | 下一版动作 |
|-------------|------|-----------|
| 效果好 | 检索假设成立 | 可加 learnability 提效降本 |
| 相关但无训练增益 | A 可能已会（缺 learnability） | 加闸门 3 |
| 相关但质量参差 | 上游质量假设不成立 | 加闸门 2 |

---

## 附录：闸门 2 / 闸门 3 设计（第二版增强项，暂不实现）

### 闸门 2：质量优选

免费硬信号（outcome_transition 含成功、ends_with_unreturned_toolcall=false、结尾无未解决报错）先过；LLM-judge 按 rubric（正确性/可模仿性/干净度，各 0-3）打 quality_score 归一到 [0,1]。

### 闸门 3：教学价值（Learnability）

倒 U 形可学性：模型 A 在切片 assistant token 上的 loss 既不能太低（已会）也不能太高（噪声）。推荐 RHO-loss 思路用参考模型校准：

```
raw = L_A − L_ref                              # 参考模型推荐用产出轨迹的更强模型
learnability_score = sigmoid((raw − μ)/σ)      # 平滑归一，μ/σ 由校准批次估计
```

- 噪声切片：L_A、L_ref 都高 → 差值小 → 低分（自动排除噪声，无需手设上阈值）
- A 弱点切片：L_A 高、L_ref 低（强模型会而 A 不会）→ 差值大 → 高分
- 正确性不在此重复（由闸门 2 / 上游保证）
- 工程注意：需同 tokenizer 逐 token 对齐；tokenizer 不一致时退用序列级 perplexity 比值

引入时机：模块 4 用闭环实测数据校准 learnability 与真实提升的相关性后再启用。

完整综合分（第二版）：`final_score = w1·relevance + w2·quality + w3·learnability`

---

# 模块 3：集合优选（Set Selection）

## Context

前面所有模块都在给"单条"打分。但 SFT 吃的是"集合"——直接取 top-N 会踩三个坑：冗余（同类切片堆积、过拟合）、覆盖不均（只覆盖最热子问题）、配比失衡（全简单/全某语言）。本模块从高分单条中选出"作为整体最优"的子集。

预算约束：**固定总预算 N**（由训练预算/显存上限决定），submodular 贪心选满 N 条停止。

## 三步流程

```
模块 2 交付（百~千级带分切片，跨多个子问题）
    │
    ▼
① 语义去重
    │
    ▼
② 覆盖度优选（submodular 贪心，核心）
    │
    ▼
③ 配比与防遗忘
    │
    ▼
最终 SFT 数据集（N 条）
```

## 步骤 1：语义去重

| 项目 | 说明 |
|------|------|
| **目标** | 去掉语义近重复切片，避免训练集里同一模式过度堆积 |
| **方法** | 两层：1) MinHash/SimHash 抓近乎逐字重复（快，token 级）；2) embedding cosine > 0.95 抓语义重复（准，向量级） |
| **保留策略** | 一组重复里保留 final_score 最高的那条 |
| **是否 LLM** | ❌ 否。纯向量/哈希计算 |
| **复用** | 切片 embedding 已在模块 1 免费层算好，零额外成本 |

## 步骤 2：覆盖度优选（submodular 贪心）

经典 **facility location / submodular maximization** 问题：在预算 N 约束下，选一个子集让它对"问题清单所有子问题 + 能力标签"的覆盖最大化。

### 算法

```
初始化：S = {}，剩余候选 = 去重后全集
重复直到 |S| = N：
    对每个候选 x，算边际增益 Δ(x|S) = f(S ∪ {x}) - f(S)
    选 Δ 最大的 x* 加入 S
返回 S
```

### 目标函数 f(S)

```
f(S) = Σ_p  min(cover(p, S), cap_p)    // 每个子问题的覆盖度（有上限 cap_p，防止堆积）
     + λ · diversity(S)                  // embedding 空间多样性项
```

- `cover(p, S)` = 子集 S 中属于子问题 p 的切片数量（× 各自 final_score 加权）
- `cap_p` = 每个子问题的覆盖上限（如 N / 子问题数 × 2，防止单一问题吃光预算）
- `diversity(S)` = S 中切片 embedding 两两距离的总和（鼓励语义分散）
- `λ` 控制覆盖度 vs 多样性的权衡

### 约束

| 约束 | 说明 |
|------|------|
| 总数 = N | 硬约束，贪心终止条件 |
| 每子问题下限 | 可设 min_per_problem（如 max(5, N/子问题数/2)），防止低分子问题被完全饿死 |
| 每子问题上限 | cap_p，防止高分子问题独占 |

### 贪心性质

submodular 函数的递减收益特性保证了：
- 一个子问题已经有足够样本后，再来的同类切片边际增益骤降，自动让位给覆盖不足的子问题
- 天然同时解决"覆盖均匀"和"多样性"

贪心解保证 ≥ (1-1/e) ≈ 63% 最优解。对此场景实际效果远好于此（因为候选池本身已是高质量的）。

| **是否 LLM** | ❌ 否。纯算法（O(N × 候选数)，百~千级候选下毫秒级） |

## 步骤 3：配比与防遗忘

| 项目 | 说明 |
|------|------|
| **难度配比** | 用切片免费信号（num_retries、turn_count、token_count）做难度代理。在步骤 2 的 diversity 项中隐式处理（不同难度的切片 embedding 自然不同），或后置检查：如果难度分布严重偏斜，swap 少量切片 |
| **掺通用数据** | SFT 时混入固定比例（如 20-30%）的通用 SWE 数据（非本次筛选产出），防窄问题过拟合和灾难性遗忘。通用数据不参与 submodular 选择，直接按比例拼入最终集 |
| **最终产出** | N 条定向切片（70-80%）+ 通用 SWE 数据（20-30%）= 完整 SFT 数据集 |
| **是否 LLM** | ❌ 否。配比规则 |

## 模块 3 整体特征

- **零 LLM 调用**：全程纯算法（哈希/向量/贪心/配比规则）
- **输入**：模块 2 交付的带分切片集 + Problem Spec 的子问题结构
- **输出**：最终 SFT 数据集（N 条定向 + 通用掺入）
- **关键参数**：N（总预算）、cap_p（单问题上限）、min_per_problem（单问题下限）、通用数据比例、λ（覆盖 vs 多样性权衡）

---

# 模块 4：闭环验证（第一版不实现，第二版增强项）

> 第一版聚焦快速出可落地方案，闭环验证由人工在 SFT 实验后手动对比验证集结果完成，不做系统化。第二版再系统化为模块。

---

# 全系统端到端流程总结（第一版）

```
用户问题清单（自然语言）
    │
    ▼
[模块 0] 问题编译（2 次 LLM 调用）
    ├── Call 1: 子问题分解
    └── Call 2: 能力标签 + 自报信号 + HyDE 正例 + keywords + 置信度自评
    │
    ├─► confidence ≥ 0.8 的子问题组成 Problem Spec
    └─► confidence < 0.8 的子问题筛除
    │
    ▼
[模块 1] 离线处理（已完成）
    ├── 切分（结构感知规则，零 LLM，token ∈ [500, 8000]）
    ├── 切片签名（免费层全量 + LLM 层惰性 batch）
    └── 索引（ES + Qdrant + docstore）
    │
    ▼
[模块 1→2] 混合召回
    ├── ES 结构化过滤 → 候选轨迹 → 展开为候选切片
    ├── Qdrant 向量召回（HyDE embedding，filtered search）
    ├── ES BM25 召回（keywords，限定候选内）
    └── RRF 融合 → 召回切片集（千级）
    │
    ▼
[模块 2] 相关性精排（单闸门）
    ├── 惰性 batch 打能力标签
    ├── 标签匹配 → 软加分（命中 ×1 / 未命中 ×0.3）
    └── relevance_score 排序 → 候选（百级）
    │
    ▼
[模块 3] 集合优选（零 LLM）
    ├── 语义去重（MinHash + embedding cosine）
    ├── submodular 覆盖度贪心（预算 N）
    ├── 配比 + 掺通用数据（20-30%）
    └── 最终 SFT 数据集（N 条）
    │
    ▼
SFT 训练 → 人工对比实验结果 → 诊断缺口 → 迭代
```

---

# 第一版优化汇总

| # | 优化 | 模块 | 效果 |
|---|------|------|------|
| ① | Call 2 + Call 3 合并为一次调用 | 模块 0 | 2 次 LLM/输入（原 3 次），延迟减 30% |
| ② | 标签冷启动从第一批问题自动生成 | 模块 0.5 | 省去前期设计 taxonomy 时间 |
| ③ | 切片 token 上限 4000→8000 | 模块 1 | 保留完整修复闭环作为训练样本 |
| ④ | 惰性打标签改 batch 调用 | 模块 1/2 | 查询延迟从千次串行→几十次 batch |
| ⑤ | 标签交集从硬过滤→软加分（×0.3 衰减） | 模块 2 | 降低标签噪声导致的误杀 |
| ⑥ | 第一版不打 demo_quality | 模块 1 | 减少 LLM token 消耗 |

---

# 第二版增强项索引

| 项 | 位置 | 触发条件 |
|----|------|---------|
| 闸门 2（质量优选 LLM-judge） | 模块 2 附录 | 实验显示"筛出数据质量参差" |
| 闸门 3（learnability RHO-loss） | 模块 2 附录 | 实验显示"相关但无训练增益"，且闭环数据校准了 learnability |
| 模块 4 系统化闭环验证 | 单独模块 | 积累 ≥3 轮实验数据 |
| demo_quality 字段恢复 | 模块 1 LLM 层 | 闸门 2 启用时 |
| 切分边界 case（格式异常、超长单轮） | 模块 1 | 工程实现时处理 |
