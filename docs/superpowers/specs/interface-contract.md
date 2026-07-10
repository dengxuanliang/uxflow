# 接口契约：模块 0 ↔ 模块 1 输入端

> 本文件是模块 0（问题清单编译）与模块 1（轨迹离线处理）之间的**唯一耦合点**定义。两个模块各自独立设计/实现/测试，但必须共同遵守此契约。
>
> **变更规则**：修改本文件视为影响两个模块的跨模块变更，需两方确认。

---

## 1. Problem Spec 完整 Schema

模块 0 的最终产出，模块 1 的查询输入。

### 1.1 顶层结构

```jsonc
{
  "raw_input": string,          // 必填。用户原始自然语言描述
  "domain": string,             // 必填。固定 "agentic_swe"（预留扩展）
  "sub_problems": SubProblem[]  // 必填。通过置信度分流的子问题列表（仅 route=="pass"）
}
```

### 1.2 SubProblem Schema

```jsonc
// SubProblem
{
  // ─── 身份 ───
  "id": string,                         // 必填。格式 "p{N}" 或 "p{N}{a-z}"（澄清产出）
  "origin": "original" | "clarified",   // 必填。"original"=Call 2 直接通过；"clarified"=经 Part 6.5 澄清救回
  "parent_id": string | null,           // origin=="clarified" 时必填，指向被拆解的原始子问题 id；否则 null

  // ─── 问题描述 ───
  "raw_text": string,                   // 必填。从用户原始描述中切出的原始片段
  "failure_summary": string,            // 必填。一句话标准化失败描述

  // ─── 能力标签 ───
  "target_capability": string[],        // 必填。1–3 个能力标签（label），来自共享 taxonomy
  // 每个 label 必须满足：小写英文 + 下划线，动宾结构，描述"正确做法"

  // ─── 轨迹自报信号 ───
  "trajectory_signal": string,          // 必填。描述该失败在轨迹中的可观察痕迹（grep/匹配模式）

  // ─── HyDE 正例 ───
  "hyde_positive": string[],            // 必填。2–3 段假设性正例轨迹片段（200–500 token/段）

  // ─── 检索条件 ───
  "keywords": string[],                 // 必填。BM25 关键词列表
  "structured_filters": StructuredFilters, // 必填。结构化过滤条件

  // ─── 置信度 ───
  "confidence": number,                 // 必填。0–1，≥0.8 才出现在此列表中
  "route": "pass"                       // 必填。本列表中恒为 "pass"
}
```

### 1.3 StructuredFilters Schema

```jsonc
// StructuredFilters
{
  "languages": string[] | null,             // 可选。枚举值见 §2.1
  "tools_used": string[] | null,            // 可选。枚举值见 §2.2
  "has_verification_step": boolean | null   // 可选。是否要求含验证步骤
}
```

### 1.4 被 Drop 的子问题（审计用，不进入模块 1）

被 drop 的子问题不出现在 `sub_problems[]` 中，但为审计目的单独存储：

```jsonc
{
  // 所有 SubProblem 字段 +
  "route": "drop",
  "drop_reason": "ambiguous" | "not_applicable" | "label_diverged" | "other"
}
```

---

## 2. 结构化过滤字段对齐（Problem Spec ↔ 切片签名）

模块 0 的 `structured_filters` 中每个字段，必须能落到模块 1 切片签名免费层的具体字段上。下表冻结字段名和枚举值：

### 2.1 languages

| 契约 | 说明 |
|------|------|
| 字段名 | `languages` |
| 类型 | `string[]` |
| 来源（模块 1 侧） | 切片签名免费层 B 组 `languages` 字段（pygments/正则抽取） |
| 枚举值 | `"python"`, `"cpp"`, `"java"`, `"javascript"`, `"go"`, `"bash"`, `"html"`, `"other"` |
| 匹配语义 | 交集非空即命中（OR） |

### 2.2 tools_used

| 契约 | 说明 |
|------|------|
| 字段名 | `tools_used` |
| 类型 | `string[]` |
| 来源（模块 1 侧） | 切片签名免费层 B 组 `tools_used` 字段 |
| 枚举值 | Claude Code 标准内置工具名：`"Bash"`, `"Read"`, `"Write"`, `"Edit"`, `"Glob"`, `"Grep"`, `"WebFetch"`, `"WebSearch"`, `"Task"`, `"TodoWrite"`, `"NotebookEdit"`；未识别工具归入 `"other"` |
| 匹配语义 | 交集非空即命中（OR） |

### 2.3 ~~outcome_transition~~（已移除）

> **已移除**（2026-07-08）。成功/失败判定改由模块 1 Phase 4 LLM 精判承担，不再作为签名层的结构化过滤字段。
> **理由**：轨迹无结构化 exit code，从 tool_result 文本推断成功/失败成本高且不可靠；LLM 精判天然覆盖"是否正向演示能力"的判断，包含成功与否。
> 保留 §2.5 编号不变，避免引用连锁改动。

### 2.5 has_verification_step

| 契约 | 说明 |
|------|------|
| 字段名 | `has_verification_step` |
| 类型 | `boolean` |
| 来源（模块 1 侧） | 切片签名免费层 D 组 `slice_type` 字段，当切片或其相邻切片含 `"verification"` 类型时为 true |
| 匹配语义 | 精确布尔匹配 |

---

## 3. BM25 关键词对齐

| 契约 | 说明 |
|------|------|
| Problem Spec 侧 | `keywords` 字段（string[]） |
| 模块 1 侧 | 切片签名免费层 E 组 `bm25_tokens` 字段（string[]） |
| 内容口径 | 两侧均为：报错关键词（`SyntaxError`、`EOF`、`Traceback`...）、工具名标识符（`bash`、`python`...）、API/库名（`re.compile`、`csv`...）、语言名 |
| 分词规则 | 按标识符级别切分（非 NLP 分词）：保留 `.` 连接（如 `re.compile`）、保留报错类名完整（如 `SyntaxError`）、小写归一化 |
| 匹配语义 | BM25 标准检索（TF-IDF），非精确字符串匹配 |

---

## 4. 向量 Embedding 对齐

| 契约 | 说明 |
|------|------|
| Problem Spec 侧 | 对 `hyde_positive` 每段做 embedding，作为向量召回的查询锚 |
| 模块 1 侧 | 切片签名免费层 E 组 `embedding` 字段（对 context_header + 正文做 embedding） |
| Embedding 模型 | 两端**必须使用同一模型**。第一版：`Qwen3-Embedding`（0.6B 或 4B，待定；两端锁定同一 checkpoint） |
| 维度 | 1024（Qwen3-Embedding-0.6B 原生输出维度） |
| 归一化 | L2 归一化后存储，余弦相似度等价内积 |
| 向量库 | Qdrant，HNSW 索引 |

---

## 5. 共享 Taxonomy Schema

### 5.1 单条标签条目 Schema — `TaxonomyLabel`（稳定核心）

```jsonc
// TaxonomyLabel
{
  "label": string,                  // 必填。唯一标识符，小写英文+下划线，动宾结构
  "parent": string | null,          // 必填。父节点 label；顶层节点为 null
  "new_root": boolean,              // 可选，默认 false。true 表示此标签创建了新顶层父节点
  "description": string,            // 必填。一句话中文描述
  "keywords": string[],             // 必填。用于回填粗筛的 BM25 关键词
  "description_embedding": float[1024], // 必填。对 description 的 embedding（同 §4 模型）
  "taxonomy_extension": boolean,    // 必填。true=LLM 新提议；false=人工预设
  "created_at": string              // 必填。ISO 8601 时间戳
}
```

### 5.2 词表整体结构

```jsonc
{
  "version": string,                // 语义化版本号，如 "0.1.0"
  "updated_at": string,             // ISO 8601
  "labels": TaxonomyLabel[]         // 所有标签条目（扁平列表，通过 parent 字段重建树）
}
```

### 5.3 访问契约

| 规则 | 说明 |
|------|------|
| 单一事实源 | 两端（模块 0 打标 prompt 注入 / 模块 1 切片打标 prompt 注入 / 回填）从同一份词表文件读取 |
| 映射 prompt 一致性 | 模块 0 Part 2 和模块 1 惰性打标使用**相同的映射 prompt 模板**，词表通过 `{词表注入}` 占位符注入 |
| 新标签入库 | 新标签追加到 `labels[]`，version patch 号 +1，不影响已有条目 |
| 去重检查 | 入库前 `description_embedding` 与已有标签 cosine > 0.85 视为重复，映射到已有标签 |
| 挂载判定 | 与已有父节点 cosine ≥ 0.6 → 挂其下；均 < 0.6 → 创建新顶层（`new_root: true`） |

---

## 6. Embedding 配置汇总

集中管理所有 embedding 相关配置，避免散落各处：

| 用途 | 输入文本 | 模型 | 维度 | 归一化 |
|------|---------|------|------|--------|
| HyDE 正例（查询锚） | `hyde_positive` 各段 | Qwen3-Embedding (0.6B/4B) | 1024 | L2 |
| 切片 embedding | context_header + 切片正文 | Qwen3-Embedding (0.6B/4B) | 1024 | L2 |
| 标签 description embedding | `description` 字段 | Qwen3-Embedding (0.6B/4B) | 1024 | L2 |

**一致性约束**：三者必须使用同一模型同一维度，否则向量空间不对齐，召回/回填/去重全部失效。模型切换时需全量重算（一次性成本）。

---

## 附录 A：v0 Taxonomy 快照（fixture 基准，会随模块 0.5 演化）

> ⚠️ 以下为冷启动实验产出的初始词表，仅供 fixture 打桩和首版实现使用。随新问题进入和模块 0.5 演化机制运行，内容会增长。**词表内容变化不构成契约变更**（仅 §5.1 的 schema 结构属于契约）。

```
code_generation                        (parent: null)
  ├── valid_syntax_in_toolcall         工具调用中生成合法代码（含 import）
  ├── correct_indentation              写入文件时缩进正确
  ├── correct_shell_embedding          bash 中正确嵌入其他语言
  └── file_localization_and_edit       正确定位目标文件并成功编辑

tool_use                               (parent: null)
  └── wellformed_tool_call             tool_call 结构正确，不错位到 thinking 块

planning                               (parent: null)
  ├── requirement_completeness         覆盖需求中所有明确功能点
  └── requirement_analysis_before_coding  复杂任务先分析需求再编码

execution_control                      (parent: null)
  ├── reproduce_before_fix             修改前先复现问题
  ├── avoid_redundant_repetition       避免重复动作循环，及时切换策略
  └── self_verification                实现后自主验证，不依赖用户提示才发现错误

error_recovery                         (parent: null)
  └── effective_error_fix              用户指出后修复真正生效
```

5 个顶层父节点 + 12 个叶子标签。

---

## 附录 B：Golden Fixture 构造指南

用于模块 0 独立测试时，模拟模块 1 侧的签名数据。

### 构造规则

1. **每条 fixture 切片签名**至少包含以下被引用字段：
   - `languages`: string[]（从 §2.1 枚举选）
   - `tools_used`: string[]（从 §2.2 枚举选）
   - `turn_count`: integer
   - `slice_type`: string
   - `bm25_tokens`: string[]（按 §3 口径）
   - `embedding`: float[1024]（用 §4 指定模型对 fixture 正文做 embedding）
   - `capability_labels`: string[] | null（回填前为 null）

2. **每个 v0 词表标签至少对应 3 条 fixture 切片**（1 条强正例 + 1 条弱相关 + 1 条不相关），用于验证：
   - HyDE embedding 召回能命中强正例
   - structured_filters 能正确过滤
   - BM25 keywords 能命中 bm25_tokens

3. **fixture 数据量**：v0 12 个叶子 × 3 条 = 36 条最小集；建议扩展到 ~100 条覆盖边界 case。

### fixture 存储

```
fixtures/
  ├── slices.jsonl          # 每行一条切片签名 JSON
  ├── taxonomy_v0.json      # 附录 A 的 JSON 格式版本
  └── problem_specs/        # 几条测试用 Problem Spec
       ├── test_01.json
       └── test_02.json
```
