# 模块 1：轨迹处理 Pipeline —— 设计 Spec

> 从回流的成功轨迹中，筛选出命中用户问题清单的正向能力示范片段，并输出 loss mask span 供 SFT 训练使用。

---

## 1. 目标与约束

### 1.1 核心目标

给定模块 0 产出的 Problem Spec（目标能力 + 检索条件），从回流轨迹库中：
1. **召回**可能演示了该能力的轨迹/切片
2. **精判**切片是否正向演示了该能力（而非复现了失败）
3. **标注** loss mask span（哪几步是正例演示，模型应该学的部分）

### 1.2 输入

- 回流轨迹 JSONL（每行一条轨迹，OpenAI messages 格式：system/user/assistant + tool_call/tool_result）
- 轨迹质量前提：回流数据本身是**成功的**轨迹（格式无问题），从中捞正例
- 轨迹长度：当前数据 ≤20 steps / ≤10 turns；系统需兼容 >100 steps

### 1.3 输出

每条命中的轨迹产出：
```jsonc
{
  "trajectory_id": string,
  "matched_capability": string[],     // 命中的能力标签
  "problem_spec_id": string,          // 来自哪个 Problem Spec
  "loss_mask_spans": [                // 模型应学的步骤范围
    {"start_step": int, "end_step": int}   // 包含两端，0-indexed
  ],
  "confidence": float                 // 精判置信度
}
```

### 1.4 关键约束

| 约束 | 说明 |
|------|------|
| 无 test cases | 不能用程序化检测器验证，依赖 LLM + 轨迹信号 |
| 网关拦截 | 单请求 ≤10k token，精判时必须压缩/摘要 |
| 数据量 | 当前几千条（秒级完成），设计需支撑百万级（分钟级完成） |
| 方向 | 问题驱动（Problem Spec → 召回切片），非切片驱动 |
| 正例导向 | 找"正确做了该能力"的片段，不是找"复现失败"的 |

---

## 2. 切片策略

### 2.1 双模式切分

| 轨迹长度 | 策略 | 切片数 |
|----------|------|--------|
| ≤10 steps | 整条 = 1 个切片（不物理切分） | 1 |
| >10 steps | 按语义边界切分 | 2-N（目标每片 5-10 步） |

**阈值 10 steps 的理由**：10 步的轨迹约 3.5-12k token，作为完整上下文送精判可能超网关限制。故精判统一走摘要模式（不管长短），阈值只决定"是否物理切分成多个独立切片"。

### 2.2 语义边界信号（>10 steps 时的切分依据）

不用 LLM，纯规则（成本为零）：

| 信号 | 检测方式 | 权重 |
|------|---------|------|
| 子任务转折 | assistant 文本含"现在""接下来""然后来做""Let me now" 等转折词 + 后续动作类型变化 | 高 |
| 工具类型切换 | 从探索类（Read/Grep/Glob）→ 修改类（Write/Edit）；从修改类→验证类（Bash 跑测试） | 高 |
| 验证点 | tool_call 是运行/测试/检查类操作（bash 含 test/check/pytest/run） | 中 |
| tool_result 状态翻转 | 前序 result 含错误特征 → 后续 result 无错误（"修复完成"边界） | 中 |
| 用户介入 | 出现 user role 消息（用户新指令 = 天然子任务边界） | 最高 |

**切分算法**：
1. 标注所有边界候选点（按信号打分）
2. 贪心选择：从头开始，每 5-10 步取分数最高的边界点切开
3. 产出 `[{start_step, end_step}]` 切片列表

### 2.3 切片作为匹配单元

- 切片是检索和精判的基本单元
- 短轨迹：整条做签名 + 整条送精判
- 长轨迹：每片独立做签名 + 独立送精判
- loss mask span 可以跨切片（精判后向外扩展到语义完整边界）

---

## 3. 处理 Pipeline（四阶段）

### Phase 1：轨迹签名（免费信号抽取，无 LLM）

对每条轨迹/每个切片，抽取结构化签名。纯规则 + 正则，零 LLM 成本。

```jsonc
// TrajectorySignature（对齐接口契约 §2 的切片侧字段）
{
  "trajectory_id": string,
  "slice_index": int,             // 0=整条或第几个切片
  "step_range": [int, int],       // [start_step, end_step]

  // ── A 组：基础结构 ──
  "step_count": int,
  "turn_count": int,              // user messages 数量

  // ── B 组：工具与语言 ──
  "languages": string[],          // 从 tool_call 内容用 pygments/正则抽取
  "tools_used": string[],         // tool_call.name 去重列表

  // ── C 组：结果状态（辅助信号，非结构化过滤字段）──
  "has_error_pattern": boolean,   // tool_result 含 Traceback/Error/SyntaxError
  "has_success_pattern": boolean, // tool_result 含 "file created"/无错误/"PASSED"

  // ── D 组：验证信号 ──
  "has_verification_step": boolean, // 含测试/检查类 tool_call

  // ── E 组：检索字段 ──
  "bm25_tokens": string[],        // 报错关键词 + 工具名 + 库名 + 语言名
  "embedding": float[1024]        // 对摘要做 Qwen3-Embedding（本地）
}
```

### Phase 2：索引建设（ES + Qdrant）

| 存储 | 内容 | 用途 |
|------|------|------|
| Elasticsearch | 全部签名字段（结构化过滤 + BM25） | 粗筛 |
| Qdrant | embedding（1024-d HNSW） | 向量召回 |

索引设计对齐接口契约 §2（Problem Spec 的 structured_filters 能直接查 ES 的对应字段）。

### Phase 3：召回（问题驱动，零 LLM）

输入一个 Problem Spec 的子问题，执行多路召回：

```
structured_filters → ES 过滤（languages + tools_used + min_turns + has_verification_step）
    ↓ 缩小候选集
keywords → ES BM25（bm25_tokens 字段）
hyde_positive embedding → Qdrant ANN（top-K）
    ↓ 两路融合
RRF 排序 → top-N 候选切片
```

对当前几千条数据，Phase 2/3 基本是全表扫描也够快。索引是为百万级扩展准备的。

### Phase 4：LLM 精判 + Span 标注

对 Phase 3 的 top-N 候选切片，LLM 判定：
1. 该切片是否正向演示了目标能力？（match: true/false）
2. 如果 match，具体哪几步是正例演示？（spans: [{start_step, end_step}]）

**精判 prompt 输入（摘要模式，控制在 10k 以内）**：

不塞原文，而是每步压缩为一行摘要：
```
Step 0: [assistant] 分析问题... → call Bash("grep -r 'error' src/")
Step 1: [tool_result] 找到3处匹配: src/main.py:12, src/util.py:45...
Step 2: [assistant] 定位到问题... → call Edit("src/main.py", ...)
Step 3: [tool_result] file edited successfully
Step 4: [assistant] 验证修复... → call Bash("python -m pytest tests/")
Step 5: [tool_result] 5 passed, 0 failed
```

加上目标能力描述 + trajectory_signal，让 LLM 判断"哪几步正确演示了该能力"。

**精判输出格式**：
```json
{
  "match": true,
  "confidence": 0.88,
  "spans": [{"start_step": 2, "end_step": 5}],
  "reasoning": "步骤 2-5 正确定位了错误、修改了代码、并通过测试验证"
}
```

**Batch 策略**：每次请求塞 3 条切片摘要（每条 1.5-2.5k token，加 prompt 共 7-9.5k，安全在 10k 以内），structured output。

---

## 4. 模块边界与接口

### 4.1 输入接口（从模块 0）

直接消费 Problem Spec（接口契约 §1）：
- `target_capability` → 精判 prompt
- `structured_filters` → Phase 3 ES 过滤
- `keywords` → Phase 3 BM25
- `hyde_positive` → Phase 3 向量召回（embedding 已在模块 0 计算）
- `trajectory_signal` → 精判 prompt（辅助 LLM 理解"什么信号代表正确能力"）

### 4.2 输出接口（送 SFT 训练）

```jsonc
// SFTCandidate — 模块 1 最终输出
{
  "trajectory_id": string,
  "trajectory_path": string,          // 原始轨迹文件路径（整条保留训练用）
  "matched_problems": [
    {
      "problem_spec_id": string,
      "sub_problem_id": string,
      "capability": string[],
      "confidence": float,
      "loss_mask_spans": [
        {"start_step": int, "end_step": int}
      ]
    }
  ]
}
```

SFT 训练消费这个输出：加载 `trajectory_path` 的完整轨迹，按 `loss_mask_spans` 构建 loss mask。

### 4.3 与 LLM Gateway 的关系

Phase 4 精判通过 `llm_gateway.LLMGateway.call()` 发起。Batch 3 条/请求，利用 gateway 的并发控制 + 断路器。

---

## 5. 精判摘要生成

把一个切片（可能 5-20 步）压缩成 ≤2.5k token 的摘要格式，送精判。

**原则：保留足够的操作内容让 LLM 能判断"能力是否正确演示"。** 光知道"调了 Write"不够，需要看到写了什么；光知道"调了 Bash"不够，需要看到执行了什么命令。

```python
def summarize_slice(messages, step_range):
    """每步压缩为一行，保留操作内容摘要。

    保留量：
      - assistant 推理：前 150 字符（意图说明）
      - tool_call：完整 name + 参数前 200 字符（含文件路径、代码片段、命令）
      - tool_result：前 300 字符（含报错或成功信息）
    """
    lines = []
    for i, step in enumerate(steps_in_range):
        if step.role == "assistant":
            reasoning = step.content[:150].replace("\n", " ")
            if step.tool_call:
                args_summary = step.tool_call.args[:200].replace("\n", "\\n")
                tool = f'call {step.tool_call.name}("{args_summary}")'
                lines.append(f"Step {i}: [assistant] {reasoning} → {tool}")
            else:
                lines.append(f"Step {i}: [assistant] {reasoning}")
        elif step.role == "tool_result":
            result = step.content[:300].replace("\n", "\\n")
            lines.append(f"Step {i}: [result] {result}")
    return "\n".join(lines)
```

**Token 估算**：
- 每步 ≈ 150-250 token（assistant 150 + tool_call 200 + result 300 ≈ 650 字符 ≈ 160 token）
- 10 步切片 ≈ 1.5-2.5k token
- Batch 3 条切片 ≈ 5-7.5k + prompt 2k ≈ 7-9.5k，安全在 10k 以内

**示例输出**：
```
Step 0: [assistant] 分析报错信息，定位到 src/main.py 第12行有语法错误... → call Read("src/main.py")
Step 1: [result] 1: package main\n2: import "fmt"\n3: func main() {\n4:   if err { ...
Step 2: [assistant] 发现第4行缺少条件判断，修复为 if err != nil → call Edit("src/main.py", old="if err {", new="if err != nil {")
Step 3: [result] file edited successfully
Step 4: [assistant] 验证修复是否生效 → call Bash("go build ./...")
Step 5: [result] Build successful. No errors.
```

---

## 6. 扩展性设计（百万级）

当前几千条全表扫描即可，但架构为百万级预留：

| 量级 | Phase 1 签名 | Phase 2 索引 | Phase 3 召回 | Phase 4 精判 |
|------|-------------|-------------|-------------|-------------|
| 几千条 | 秒级（纯规则） | 可跳过（内存扫描） | 可跳过（全量过滤） | 几百次调用（分钟级） |
| 百万级 | 分钟级（并行） | ES+Qdrant 必需 | 毫秒级（索引查询） | 只对 top-N 候选（可控） |

**第一版实现策略**：
- Phase 1：必做（签名是一切的基础）
- Phase 2：用内存 dict 替代 ES/Qdrant（几千条不需要真索引）
- Phase 3：内存全量过滤 + numpy 向量运算
- Phase 4：真实 LLM 调用（通过 gateway）

百万级时再切换到真 ES/Qdrant，Phase 1-3 接口不变。

---

## 7. 第一版实现范围

| 组件 | 第一版 | 百万级扩展 |
|------|--------|----------|
| 切片策略 | ≤10 steps 整条 + >10 steps 语义切分 | 不变 |
| 签名抽取 | 完整实现（规则 + embedding） | 不变 |
| 索引 | **内存 dict + numpy**（几千条够用） | 换 ES + Qdrant |
| 召回 | 内存过滤 + numpy cosine | 换 ES BM25 + Qdrant ANN |
| 精判 | LLM via gateway（batch 3） | 不变（并发加大） |
| 输出 | SFTCandidate JSONL | 不变 |

---

## 8. 与接口契约的对齐

| 契约字段 | 模块 1 侧实现 |
|---------|-------------|
| `languages` (§2.1) | Phase 1 从 tool_call 内容提取 |
| `tools_used` (§2.2) | Phase 1 从 tool_call.name 提取 |
| `min_turns` ↔ `turn_count` (§2.4) | Phase 1 统计 user messages |
| `has_verification_step` (§2.5) | Phase 1 检测测试/检查类 tool_call |
| `bm25_tokens` (§3) | Phase 1 提取报错关键词 + 工具名 + 库名 |
| `embedding` (§4) | Phase 1 对摘要做 Qwen3-Embedding |

枚举值与契约冻结的一致（languages 8 值、tools_used 12 值）。

---

## 9. TODO / 未来增强

- **主动召回**：不等 Problem Spec 输入，定期对新入库轨迹主动匹配全部能力标签（回填方向）
- **负例标注**：同时标出"复现了失败"的 span，供对比学习
- **跨切片 span**：精判结果的 span 允许跨越切片边界（当前版不做，span 限制在切片内）
- **增量签名**：新轨迹入库时增量计算签名，不重跑全量
