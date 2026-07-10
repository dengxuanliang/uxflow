# SWE 轨迹优选器 — 架构 UML

> 视图：模块划分 + 模块间数据传递 + 依赖关系。
> 状态标注：实线框 = 已实现（模块 0/1/2/3 + LLM Gateway）；虚线框 = 仅设计未实现（模块 0.5）。
> 数据流方向：Problem Spec → 召回打分 → 精排 → 集合优选 → SFT 数据集。

## 1. 组件 + 数据流视图（主图）

```mermaid
flowchart TB
    user([用户自然语言问题描述])

    %% ============ 共享基础设施 ============
    subgraph GW["🔌 LLM Gateway (src/llm_gateway)"]
        direction LR
        gwcore["LLMGateway<br/>transport · runtime<br/>recovery · truncation · outcomes"]
    end

    %% ============ 模块 0 ============
    subgraph M0["📥 模块 0 · 问题清单编译 (src/module0)"]
        direction TB
        compiler["QueryCompiler.compile()"]
        emb0["EmbeddingModel<br/>(Qwen3-Embedding 1024d)"]
        prompts0["prompts / parsing"]
        compiler --> prompts0
        compiler --> emb0
    end

    %% ============ 共享词表 ============
    subgraph TAX["🏷️ 共享 Taxonomy (src/module0/taxonomy)"]
        taxonomy["Taxonomy (只读视图)<br/>TaxonomyLabel[]"]
    end

    %% ============ 模块 0.5（设计未实现）============
    subgraph M05["🧬 模块 0.5 · 标签系统演化 (src/module0_5) — 设计未实现"]
        direction TB
        evo["① 提议捕获 + ② 去重挂载<br/>evolution.py"]
        inh["③ 继承降权查询<br/>inheritance.py (×0.3)"]
        bf["④ 异步回填 backfill.py<br/>粗筛→LLM精判→写回"]
        taxstore["TaxonomyStore (可变写回)"]
        evo --> taxstore
    end

    %% ============ 模块 1 ============
    subgraph M1["🗂️ 模块 1 · 轨迹离线处理 (src/module1)"]
        direction TB
        loader["loader 解析"] --> slicer["slicer 切片"]
        slicer --> sig["signature 抽签名<br/>+ summarizer"]
        sig --> store["MemoryIndex ⟵impl⟶ SliceStore<br/>(add_batch/recall/get_slice/update_labels)"]
        judge["Judge.judge_batch()<br/>match / confidence / loss_mask_spans"]
        pipe1["TrajectoryPipeline.run_scored()"]
        pipe1 --> store
        pipe1 --> judge
    end

    %% ============ 模块 2 ============
    subgraph M2["⚖️ 模块 2 · 相关性精排 (src/module2)"]
        rerank["rerank() 软加分<br/>relevance = rrf ×(match?1.0:0.3)"]
    end

    %% ============ 模块 3 ============
    subgraph M3["🎯 模块 3 · 集合优选 (src/module3)"]
        direction TB
        dedup["dedup 语义去重<br/>MinHash + cosine>0.95"]
        sel["selection submodular<br/>facility location 贪心"]
        comp["compose 配比 + 通用数据钩子"]
        dedup --> sel --> comp
    end

    sft([最终 SFT 数据集 · N 条])

    %% ============ 跨模块数据流 ============
    user -->|raw_input| compiler
    compiler -->|"ProblemSpec<br/>(SubProblem[] + StructuredFilters)"| pipe1
    compiler -.->|hyde_embeddings 查询锚| pipe1
    compiler -.->|"label_proposals (side-channel)"| evo

    taxonomy -->|"to_prompt_text() 注入"| prompts0
    taxonomy -->|标签树注入| judge
    taxstore -.->|version+1 写回| taxonomy

    store -->|"RecallHit (sig + rrf_score)"| pipe1
    pipe1 -->|"recalled_hits + judge_results"| rerank
    rerank -->|"ScoredCandidate[] (切片级 top-N)"| dedup
    comp -->|"targeted 70-80% + 通用 20-30%"| sft

    %% 模块 0.5 与语料/查询的交互
    inh -.->|"query-time 融合 (父标签×0.3)"| store
    bf -.->|"store.update_labels() 回写<br/>capability_labels"| store

    %% 对 Gateway 的依赖
    compiler -.->|LLM 调用| gwcore
    judge -.->|LLM 调用| gwcore
    bf -.->|LLM 精判| gwcore

    classDef done fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px;
    classDef todo fill:#fff3e0,stroke:#ef6c00,stroke-width:2px,stroke-dasharray:5 4;
    classDef infra fill:#e3f2fd,stroke:#1565c0,stroke-width:2px;
    classDef shared fill:#f3e5f5,stroke:#6a1b9a,stroke-width:2px;
    class M0,M1,M2,M3 done;
    class M05 todo;
    class GW infra;
    class TAX shared;
```

## 2. 关键数据模型（跨模块契约对象）

```mermaid
classDiagram
    class ProblemSpec {
        +raw_input: str
        +domain: str
        +sub_problems: SubProblem[]
    }
    class SubProblem {
        +id, origin, parent_id
        +target_capability: str[]
        +trajectory_signal: str
        +hyde_positive: str[]
        +keywords: str[]
        +structured_filters
        +confidence, route
    }
    class StructuredFilters {
        +languages, tools_used
        +min_turns
        +has_verification_step
    }
    class TrajectorySignature {
        +trajectory_id, slice_index
        +languages, tools_used
        +turn_count, has_verification_step
        +embedding: float[1024]
        +capability_labels: str[] | None
    }
    class RecallHit {
        +signature: TrajectorySignature
        +rrf_score: float
    }
    class JudgeResult {
        +match: bool
        +confidence: float
        +loss_mask_spans
    }
    class ScoredCandidate {
        +trajectory_id, slice_index
        +sub_problem_id, capability
        +relevance_score
        +judge_confidence, judge_match
        +embedding
    }
    class TaxonomyLabel {
        +label, parent, new_root
        +description
        +description_embedding: float[1024]
    }
    class LabelProposal {
        +label, description, parent
        +description_embedding
        (模块0.5 输入 · 未实现)
    }

    ProblemSpec "1" *-- "N" SubProblem
    SubProblem *-- StructuredFilters
    RecallHit --> TrajectorySignature
    ScoredCandidate ..> RecallHit : rrf_score
    ScoredCandidate ..> JudgeResult : match/conf/spans
    TaxonomyLabel <.. LabelProposal : 去重挂载后入库
    SubProblem ..> TaxonomyLabel : target_capability 引用
    TrajectorySignature ..> TaxonomyLabel : capability_labels 引用
```

## 3. 依赖与调用关系速览

| 依赖方 | 被依赖 | 传递物 / 说明 |
|--------|--------|---------------|
| 模块 0 → 模块 1 | `ProblemSpec` + `hyde_embeddings` | 查询输入（契约唯一耦合点） |
| 模块 0 ⇢ 模块 0.5 | `label_proposals` (side-channel) | 新标签提议（0.5 ② 消费）*未实现* |
| 模块 1 内部 | `MemoryIndex.recall()` → `RecallHit` | 带回 RRF 融合分（关键改动） |
| 模块 1 → 模块 2 | `recalled_hits` + `judge_results` | `rerank()` 软加分 |
| 模块 2 → 模块 3 | `ScoredCandidate[]`（切片级 top-N） | 去重 → submodular → 配比 |
| 模块 0.5 ⇢ 模块 1 | `SliceStore.update_labels()` | 异步回填写回 `capability_labels` *未实现* |
| 模块 0 / 1 / 0.5 → Gateway | LLM 调用 | `QueryCompiler` / `Judge` / `backfill` |
| 模块 0 / 1 / 0.5 → Taxonomy | 读注入 / 可变写回 | 单一事实源，0.5 经 `TaxonomyStore` 演化 |
```
