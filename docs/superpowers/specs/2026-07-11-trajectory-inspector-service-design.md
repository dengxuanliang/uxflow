# 轨迹 Inspector 服务与前端 设计

> **状态**：设计定稿，待评审。
> **范围**：新增 `src/service/`（只读编排 + 聚合 + 轻量异步 HTTP/SSE 服务）+ 一个原生单页前端（`src/service/web/`），把现有模块 0/1/2/3/0.5 流水线包成"上传清单+轨迹 → 运行 → 可视化检分命中轨迹"的产品。**现有 4 个模块零改动。**
> **目标**：交付上次 brainstorm 定的轨迹 Inspector UI 的可运行实现，并在前后端之间冻结一个稳定契约（`InspectorView` + SSE 进度），使前端与后端各自独立演化。

---

## 1. 背景与动机

上次 brainstorm 用 4 个 HTML mockup（`.superpowers/brainstorm/.../content/*.html`）定了一个"人工检分 SWE 轨迹"的 Inspector UI，导航层次为 **问题清单 → 正向能力 → 命中轨迹 → 轨迹详情（step 级高亮 loss_mask 命中片段）**。但仓库主体是纯 Python 后端流水线，没有任何前端实现，也没有把 4 个模块的末端产物 join 成前端可消费视图的聚合层。

本设计补齐三块：
1. **编排层**：把 `scripts/e2e_smoke.py` 的 `main()` 编排逻辑抽成可被服务调用、能 emit 进度的 `run_pipeline()`。
2. **聚合层**：把末端 `ScoredCandidate`（含 `loss_mask_spans`）与 `Trajectory`（含 `steps` 全文）join 成 `InspectorView` 契约。
3. **服务 + 前端**：轻量异步 HTTP/SSE 服务 + 原生单页 UI（上次 mockup 的实现）。

## 2. 已确认决策

| # | 决策点 | 选定 | 备注 |
|---|--------|------|------|
| 1 | 上传口1（用户清单） | **原始抱怨清单**（每行一条自然语言抱怨） | 每条走模块0 LLM 编译成 ProblemSpec |
| 2 | 上传口2（候选回流） | 轨迹 jsonl（`{"id","messages":[...]}`，每行一条） | 直接喂模块1，格式同 `fixtures/trajectories/*.jsonl` |
| 3 | 运行时长处理 | **异步 job + SSE 进度** | V1 内存态；避免同步 HTTP 撞网关/浏览器超时 |
| 4 | 高亮粒度 | **step 级**（`{start_step, end_step}`） | 与后端 `loss_mask_spans` 现状 1:1，前端给整个 step 加背景色 |
| 5 | 能力高亮颜色 | **后端给稳定色** | 聚合层按 taxonomy label 分配稳定色，写进契约 |
| 6 | 前端技术栈 | **原生单页（无构建）** | 不为内部审阅工具引入 node 工具链；mockup 本就是原生 HTML |
| 7 | 服务层位置 | **新建 `src/service/`** | 与 module0/1/2/3 平级；它是编排层，不属于任何单模块 |
| 8 | 第一版展示范围 | **含模块3选集** | 右侧标记哪些命中片段真进了最终 SFT 选集 + manifest |
| 9 | 存储后端 | **V1 内存 `RunStore`，留 seam** | `RunStore` 协议，以后换 Redis/DB 上层零改动（对齐既有 `SliceStore` 模式）|
| 10 | 结果页布局 | **方案A 三栏**（问题+能力 / 命中轨迹列表 / 轨迹详情）| layout.html 方案A |
| 11 | 非聚焦片段 | **淡显 + 小色标签** | 保留"点它可切过去"线索（focus-highlight 增强项1）|
| 12 | 全能力开关 | **要**（总开关回到多色同时高亮）| focus-highlight 增强项2 |

## 3. 数据溯源（Inspector 每块数据的确切后端出处）

| Inspector 视图元素 | 后端来源 | 结构 |
|---|---|---|
| 问题清单 | 模块0 `ProblemSpec.sub_problems[]` | `id / failure_summary / target_capability[] / confidence` |
| 正向能力 | `SubProblem.target_capability[]` + taxonomy | label 为共享词表叶子，有 parent 树 |
| 命中轨迹列表 | 模块2 `ScoredCandidate[]` 按 `sub_problem_id` 聚合 | `trajectory_id / slice_index / relevance_score` |
| 轨迹详情（逐 step） | 模块1 `Trajectory.steps[]` | `Step{index, role, content, tool_call_name, tool_call_args, tool_result}` |
| 高亮命中片段 | `ScoredCandidate.loss_mask_spans` | `[{start_step, end_step}]`（loss_mask=1 锚点）|
| 片段↔能力归属 | `ScoredCandidate.capability[] + sub_problem_id` | 决定 focus-highlight "这段是哪个能力" |
| 分数/置信 | `relevance_score / judge_confidence / judge_match` | |
| 最终选集标记 | 模块3 `select_final_dataset()` → `targeted[]` | candidate 在 `targeted` 里 = `selected: true` |
| manifest | 模块3 `{targeted_count, general_count, general_ratio}` | 右上角/底部展示 |

**关键实现事实**（已核对源码）：
- `TrajectoryPipeline.run_scored(*, trajectory_paths, problem_specs)` → `list[ScoredCandidate]`
- `select_final_dataset(candidates, *, sub_problem_ids, selection, general)` → `{targeted, general, manifest}`；`targeted` 是被选中的 `ScoredCandidate` 子集
- `load_trajectories(path)` → `list[Trajectory]`（`steps[]` 即轨迹全文）
- `QueryCompiler.compile(raw_input)` → `ProblemSpec`；`compiler.dropped_records` 为审计用被 drop 子问题

## 4. 架构

```
┌─────────────────────────────────────────────────┐
│ 前端 Inspector (原生单页, src/service/web/)       │
│  上传区: [清单.txt] [轨迹.jsonl] [运行]            │
│  运行中: waiting 式进度 (SSE 阶段流)               │
│  结果:   左=问题清单  右=命中轨迹 + step级高亮     │
│          能力点选聚焦, 后端给稳定色                │
└───────────────┬─────────────────────────────────┘
                │ HTTP + SSE (冻结契约: InspectorView + 进度事件)
┌───────────────▼─────────────────────────────────┐
│ 服务层 src/service/  (FastAPI + uvicorn)          │
│  app.py       POST /runs, SSE /runs/{id}/events,  │
│               GET /runs/{id}/view,                │
│               GET /runs/{id}/trajectory/{tid}     │
│  runstore.py  RunStore 协议 + MemoryRunStore(V1)  │
│  orchestrator.py  run_pipeline() 抽自 e2e_smoke,   │
│                   每阶段 emit 进度事件            │
│  viewmodel.py InspectorView 聚合:                 │
│               ScoredCandidate ⋈ Trajectory,       │
│               + selected 标记 + 能力稳定配色      │
└───────────────┬─────────────────────────────────┘
                │ 直接调用现有编排 (零改动)
┌───────────────▼─────────────────────────────────┐
│ 现有流水线 module0/1/2/3/0.5                       │
└─────────────────────────────────────────────────┘
```

**边界原则**：前端只吃 `InspectorView` + 进度事件两个契约，绝不直接碰 4 个模块的内部 dataclass。模块内部结构演化不影响前端；聚合层是唯一翻译点。

## 5. 契约定义（前后端唯一耦合点）

### 5.1 运行接口

```
POST /runs
  multipart/form-data: manifest(file, .txt) + trajectories(file, .jsonl)
  → 200 { "run_id": string }        # 立即返回, 后台 asyncio task 开跑

GET /runs/{run_id}/events   (text/event-stream, SSE)
  → data: {"stage": "module0", "status": "running", "msg": "编译 p2..."}
  → data: {"stage": "module1", "status": "running", "msg": "召回 20 条..."}
  → data: {"stage": "done",    "status": "ok"}          # 或 {"status":"error","msg":...}

GET /runs/{run_id}/view
  → 200 InspectorView   (见 5.2; 仅 stage=done 后可用, 否则 409)

GET /runs/{run_id}/trajectory/{trajectory_id}
  → 200 { "trajectory_id", "steps": [Step...] }   # lazy: 大轨迹按需拉
```

进度阶段枚举（对齐流水线天然分段）：`module0`（编译）→ `module1`（切片+召回+精判）→ `module2`（软加分）→ `module3`（去重+选集）→ `done`。每阶段一条或多条 `running` 事件 + 一条隐式完成。

### 5.2 InspectorView Schema

```jsonc
InspectorView {
  "run_id": string,
  "problems": [                         // 左侧问题清单 (模块0 产出)
    {
      "id": string,                     // "p1" / "p2a"
      "failure_summary": string,
      "confidence": number,
      "capabilities": [                  // 正向能力 (focus-highlight 可点选项)
        {
          "label": string,              // taxonomy 叶子, 如 "self_verification"
          "parent": string | null,
          "color": string,              // 后端分配的稳定色 (#hex), 决策5
          "hit_count": number,          // 该能力命中的轨迹片段数
          "hit_trajectories": [
            {
              "trajectory_id": string,
              "slice_index": number,
              "relevance_score": number,
              "judge_confidence": number,
              "selected": boolean,       // 是否进了模块3最终选集 (决策8)
              "loss_mask_spans": [       // step级, 决策4
                { "start_step": number, "end_step": number }
              ]
            }
          ]
        }
      ]
    }
  ],
  "manifest": {                          // 模块3 选集摘要
    "targeted_count": number,
    "general_count": number,
    "general_ratio": number
  }
}
```

轨迹全文**不内联**在 view 里（大轨迹会撑爆首屏），通过 `/trajectory/{id}` lazy 拉。view 只带每条命中轨迹的 `trajectory_id` + `loss_mask_spans`，前端点开某条轨迹时再拉全文，用 spans 高亮。

### 5.3 稳定配色规则（决策5）

聚合层为每个出现在结果中的 taxonomy label 分配一个稳定色：按 label 在 taxonomy 顶层父节点下的顺序，从固定调色板（继承 mockup 的 `#f5b800` 黄、`#1c7ed6` 蓝等）取色。同一 label 在同一次 run 内颜色恒定，写进契约的 `capabilities[].color`。前端直接用，不自己分配。

## 6. 聚合层逻辑（viewmodel.py）

输入：`spec: ProblemSpec`、`scored: list[ScoredCandidate]`、`select_result: dict`（模块3产出）、`taxonomy`。
输出：`InspectorView` dict。

核心 join：
1. 建 `selected_keys = {(c.trajectory_id, c.slice_index, c.sub_problem_id) for c in select_result["targeted"]}`。
2. 遍历 `spec.sub_problems` → 每个 sub_problem 一个 `problem`。
3. 该 problem 的 `capabilities` = `sub_problem.target_capability`；每个 capability 分配稳定色。
4. 对每个 `ScoredCandidate`，按 `sub_problem_id` 归到 problem，按 `capability` 归到 capability 的 `hit_trajectories`；`selected = key in selected_keys`。
5. `manifest` 直接取 `select_result["manifest"]`。

轨迹全文单独由 `load_trajectories` 结果按 `trajectory_id` 索引，供 `/trajectory/{id}` 端点返回。

## 7. 编排层（orchestrator.py）

`run_pipeline(manifest_lines, trajectory_path, *, emit, config)` — 抽自 `e2e_smoke.py` 的 `main()`，去掉 print、改为 `emit(stage, msg)` 回调：

1. **module0**：对 manifest 每行 `compiler.compile(line)`；合并所有 `sub_problems`（跨行问题合成一个 `problem_specs` 列表）。emit 每条编译进度。
2. **module1+2**：`pipeline.run_scored(trajectory_paths=[path], problem_specs=specs)` → `scored`。emit 召回/精判进度。
3. **module3**：`select_final_dataset(scored, ...)` → `select_result`。emit 选集进度。
4. **聚合**：`build_inspector_view(...)` → `InspectorView`；连同 `trajectories` 索引存入 `RunStore`。
5. emit `done`。

模块0.5（标签演化回填）**第一版不接入运行主链**——它是离线演化机制，不属于"单次检分"交互。留作后续增强（见 §10）。

失败处理：任一阶段抛异常 → emit `{stage, status:"error", msg}`，run 标记 failed，`/view` 返回 409。对齐既有"LLM 失败不抛穿、进 errors"的风格。

## 8. RunStore seam（决策9）

```python
class RunStore(Protocol):
    def create(self) -> str: ...                       # 返回 run_id
    def append_event(self, run_id: str, event: dict) -> None: ...
    def events(self, run_id: str) -> Iterable[dict]: ...  # SSE 消费, 支持追加中订阅
    def set_view(self, run_id: str, view: dict, trajectories: dict) -> None: ...
    def get_view(self, run_id: str) -> dict | None: ...
    def get_trajectory(self, run_id: str, traj_id: str) -> dict | None: ...
    def status(self, run_id: str) -> str: ...          # running|done|error
```

V1 `MemoryRunStore`：`dict[run_id → {status, events: list, view, trajectories}]` + `asyncio` 事件通知供 SSE 订阅。以后换 Redis/DB 只实现同协议，`app.py` 零改动。对齐模块1 `SliceStore`（`recall`/`update_labels` 分离 seam）的既有做法。

## 9. 前端（src/service/web/，原生单页）

- `index.html` + `app.js` + `style.css`，无构建。继承 mockup 的配色与三层导航。
- 上传区两个 `<input type=file>` + 运行按钮 → `POST /runs`。
- 运行中：订阅 SSE，渲染 waiting.html 式分阶段进度。
- 结果：**布局采纳方案A 三栏**（决策10：左=问题+能力，中=命中轨迹列表，右=轨迹详情）；能力点选聚焦、后端稳定色；非聚焦能力片段**淡显+小色标签**（决策11）；提供**"显示全部能力"总开关**回到多色同时高亮（决策12）。
- 点问题 → 中栏列该 problem 的命中轨迹（每条显示片段数）；点某条轨迹 → `GET /trajectory/{id}` 拉全文到右栏，按 `loss_mask_spans` 给 step 加色；点能力 → 聚焦切换高亮，非聚焦命中片段淡显+小标签。
- `selected: true` 的命中片段额外加"已入选 SFT"标记（决策8）。

> **注**：布局与聚焦交互已定稿（决策10–12）。

## 10. 非目标 / 后续增强

- **模块0.5 回填不接入**：单次检分不触发标签演化；后续可加"发现新能力 → 回填"的异步入口。
- **多用户/持久化**：V1 内存态，重启丢失；`RunStore` seam 已留。
- **字符级高亮**：V1 step 级；若未来要精到字符，`loss_mask_spans` 结构可扩 `char_range` 而不破契约（决策4 已按现状 step 级，未预留 char seam——若要预留需在 plan 确认）。
- **鉴权/并发限流**：内部工具，V1 不做。

## 11. 依赖变更

`pyproject.toml` 新增可选依赖组：
```toml
[project.optional-dependencies]
service = ["fastapi>=0.110", "uvicorn>=0.29", "python-multipart>=0.0.9"]
```
不进主依赖，保持核心流水线纯净（对齐现有 `local-embed`/`dev` 分组惯例）。`src/service` 加入 wheel packages。

## 12. 开放点（实现前需确认）

1. ~~布局 A/B 与聚焦增强开关~~ — **已定稿**（决策10 方案A三栏 / 决策11 淡显+小标签 / 决策12 要总开关）。
2. **manifest 展示位置**——右上角总览 vs 底部条，UI 细节，可实现时定。
3. **多行清单跨行问题的 id 命名**——多条抱怨各自编译，`p1/p2` 会跨行重复；聚合时是否加行前缀（如 `line1.p1`）。倾向加前缀避免碰撞。
