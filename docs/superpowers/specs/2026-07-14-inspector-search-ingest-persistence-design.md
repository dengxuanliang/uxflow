# Inspector 搜索/入库 双功能与持久化 设计

> **状态**：设计定稿，待评审。
> **范围**：把现有 Inspector（`src/service/`）从"每次上传→全量重跑→内存态→重启清零"改造成 **读写分离的双功能前端**：`/search`（Google 式只读检索）+ `/ingest`（问题清单与回流轨迹的持久化入库）。共享一个 `uxflow.db`。新增 3 张持久表 + 3 个 Store，复用已有 `SqliteSliceStore`/`RunStore` seam。**模块 0/1/2/3 内部算法零改动**，只在 pipeline 外围拆分入口。
> **目标**：① 输入新用户问题 → 秒出「能力 / 命中轨迹 / 详情」三栏（老问题零 LLM）；② 问题清单去重后不重复编译；③ 轨迹索引与全文持久化，不重复切片/精判；④ 显式「入库」按钮真实落盘，可只传清单或只传轨迹之一。

---

## 1. 背景与动机

现有 Inspector（见 `2026-07-11-trajectory-inspector-service-design.md`）是**单次检分**模型：`POST /runs` 一次性上传「清单 + 轨迹」，`run_pipeline` 串行跑 `编译 → 建索引 → 召回+精判 → 选集 → view`，结果存内存 `MemoryRunStore`，**重启全丢**。已核对的两个关键实现事实：

- `TrajectoryPipeline.run_scored()` 每次调用都 `_reset_store()` + `_build_index()` —— 每跑一次都重建整个轨迹索引，跑完丢弃。
- 生产线 `scripts/inspector_serve.py` 用默认 `store_factory=MemoryIndex`，全内存。

用户要的是两个**长期、增量**的能力：

1. **搜索（读）**：输入一个新用户问题 → 三栏结果，完全 Google 体感。问题先去重：若与库中已有问题高度重复，展示「与之重复的已有问题 + 其结果」，而非重复编译。
2. **入库（写）**：显式「入库」按钮，上传 *用户清单* 和/或 *回流轨迹* 后真实持久化——轨迹索引与全文、问题清单与编译产物落盘，重启不丢，不重复编译/比对。

现状家底（已核对源码）：

| 数据 | 现状 | 位置 |
|---|---|---|
| 轨迹签名 + 切片源 | ✅ SQLite，`INSERT OR REPLACE` by `(traj_id, slice_index)` 天然幂等 | `module1/sqlite_store.py` `SqliteSliceStore` |
| 分类法标签 | ✅ SQLite | `module0/sqlite_taxonomy.py` |
| 回填任务队列 | ✅ SQLite | `module0_5/queue.py` |
| DB 路径解析 | ✅ 单一 `uxflow.db`，XDG 优先 | `uxflow_paths.py` |
| **问题清单（ProblemSpec）** | ❌ 无表，仅活在单次 run 内存 view | — |
| **轨迹全文（详情栏）** | ❌ 每次 run 从上传文件现载 `build_trajectory_index` | `service/viewmodel.py` |
| **run 状态/SSE/view** | ❌ 内存 `MemoryRunStore`，重启丢 | `service/runstore.py` |

`SqliteSliceStore` 已实现「启动载入内存镜像 + recall 复用 `recall_core`」——**搜索时直接对持久库 recall，无需重建索引**。这是「不重复轨迹比对」的地基，已经就位。

## 2. 已确认决策

| # | 决策点 | 选定 | 备注 |
|---|--------|------|------|
| 1 | 读写边界 | **只读预览，不入库** | `/search` 纯读；搜到"新问题"只展示，**不写** `problems` 表。收藏必须显式走 `/ingest` |
| 2 | "不重复比对"策略 | **成对精判缓存 `JudgeCache`** | 键 `(sub_problem_id, traj_id, slice_index)` → 精判结论。结论与索引里其它轨迹无关，故成对稳定、无需失效逻辑 |
| 3 | 精判缓存填充时机 | **惰性（读时按需）** | 入库快；第一次搜到涉及某 (问题,切片) 对时才精判并回填。避免为永不再搜的对白算 |
| 4 | 轨迹详情跨重启读取 | **新建轨迹全文表 `trajectories`** | 入库时存完整 `steps`，详情栏直接查库；不再依赖单次 run 的内存 index |
| 5 | 问题去重键 | **对原始问题整行做 embedding，cosine ≥ τ_q 判重** | 纯向量、**不走 LLM**；去重发生在 compile **之前**，重复问题免掉最贵的编译 |
| 6 | 轨迹去重 | **复用 `SqliteSliceStore` 的 upsert** | `INSERT OR REPLACE by (traj_id, slice_index)`，重复上传同批轨迹不翻倍 |
| 7 | 入库可分别提交 | **清单、轨迹各自可选，可只传其一** | 两条子通路独立，互不依赖 |
| 8 | 持久后端 | **单一 `uxflow.db`（复用 `resolve_db_path`）** | 对齐 module0.5 SQLite 持久化定案：不引 ES/Qdrant，向量存 float32 BLOB |
| 9 | 前端形态 | **在现有三栏之上加双模式 header**（搜索框 + 入库区），结果三栏渲染零改动 | 原生单页，无构建 |

## 3. 架构

```
                        ┌──────────────  uxflow.db  ──────────────┐
                        │ signatures / slice_sources  [已有]        │  ← 轨迹索引 (upsert)
                        │ tax_labels / tax_meta       [已有]        │
                        │ problems                    [新] §5.1     │  ← 问题清单 + 去重向量
                        │ trajectories                [新] §5.2     │  ← 轨迹全文 (steps)
                        │ judge_cache                 [新] §5.3     │  ← 成对精判结论
                        └───────────────────────────────────────────┘
       ┌──────────── 写路径 /ingest (唯一写入点) ───┐  ┌───── 读路径 /search (纯只读) ─────┐
  轨迹: load→slice→sign→SqliteSliceStore.add_batch    问题 q:
        └────────────────────→ TrajectoryStore.upsert    emb=embed(q); hit=ProblemStore.nearest(emb)
  清单: 每行 embed→ProblemStore 去重(τ_q)→             if cos≥τ_q: spec=hit.spec (零 LLM, 命中已有)
        新问题才 compile→ProblemStore.add               else:       spec=compile(q) (仅编译, 不写库)
        (纯写, 不召回不精判)                            对每 sub_problem:
                                                          hits=SqliteSliceStore.recall(持久库, 不重建)
                                                          verdict=JudgeCache.get(...) or judge()+put()
                                                        score→select→build_inspector_view (复用)
                                                        详情: /trajectory/{id} 查 TrajectoryStore
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│ 现有流水线 module0/1/2/3  —— 内部算法零改动，仅 pipeline 外围拆 ingest/search 两入口 (§7)        │
└─────────────────────────────────────────────────────────────────────────────────────────────┘
```

**边界原则**：`/search` 绝不写库，`/ingest` 是唯一写入点。四个 Store（Slice/Problem/Trajectory/JudgeCache）各是 `class + Protocol` seam，互不耦合，均落 `uxflow.db`。

## 4. 数据溯源（新增数据的确切出处）

| 元素 | 来源 | 结构 |
|---|---|---|
| 问题去重向量 | `Embedder.embed(raw_question)`（`uxflow_embed` 协议，纯向量） | float32 BLOB |
| 问题编译产物 | `QueryCompiler.compile(raw_question) → ProblemSpec` | `spec_json`（orchestrator `_spec_to_dict` 同构，含 sub_problems + hyde_embeddings 溯源）|
| 轨迹索引 | `slice_trajectory` + `extract_signature` | 复用 `SqliteSliceStore` schema |
| 轨迹全文 | `load_trajectories(path) → Trajectory.steps[]` | `steps_json`（`viewmodel._step_to_dict` 同构）|
| 精判结论 | `Judge.judge_batch` 的 `(match, confidence, spans)` | 按 `(sub_problem_id, traj_id, slice_index)` 存 |

## 5. 契约定义

### 5.1 `problems` 表 / ProblemStore

```sql
CREATE TABLE IF NOT EXISTS problems (
    problem_id          TEXT PRIMARY KEY,   -- 稳定 id（raw_question 的 hash 或 uuid）
    raw_question        TEXT NOT NULL,
    question_embedding  BLOB,               -- float32，去重键
    spec_json           TEXT NOT NULL,      -- 编译产物（含 sub_problems）
    created_at          TEXT NOT NULL
);
```

```python
class ProblemStore(Protocol):
    def add(self, raw_question: str, embedding: list[float], spec: dict, *, created_at: str) -> str: ...
    def nearest(self, embedding: list[float]) -> tuple[str, float, dict] | None: ...  # (problem_id, cosine, spec)
    def get(self, problem_id: str) -> dict | None: ...
    def all_ids(self) -> list[str]: ...
    def count(self) -> int: ...
```

- `add` 幂等：`problem_id` 用 `raw_question` 规范化后的 hash，重复 add 走 `INSERT OR REPLACE`。
- `nearest` 返回最高 cosine 及其 spec；调用方比阈值 `τ_q` 判定命中。
- 去重发生在 compile **之前**（决策5）：`/search` 与 `/ingest` 清单子通路都先 `embed → nearest`，命中则跳过 compile。

### 5.2 `trajectories` 表 / TrajectoryStore

```sql
CREATE TABLE IF NOT EXISTS trajectories (
    trajectory_id  TEXT PRIMARY KEY,
    steps_json     TEXT NOT NULL,   -- 完整 steps，viewmodel._step_to_dict 同构
    source_path    TEXT,
    created_at     TEXT NOT NULL
);
```

```python
class TrajectoryStore(Protocol):
    def upsert(self, trajectory_id: str, steps: list[dict], *, source_path: str, created_at: str) -> None: ...
    def get(self, trajectory_id: str) -> dict | None: ...   # {"trajectory_id", "steps": [...]}
    def count(self) -> int: ...
```

`GET /trajectory/{id}` 改查此表（不再依赖内存 index），重启后详情栏正常。

### 5.3 `judge_cache` 表 / JudgeCache

```sql
CREATE TABLE IF NOT EXISTS judge_cache (
    sub_problem_id  TEXT NOT NULL,
    trajectory_id   TEXT NOT NULL,
    slice_index     INTEGER NOT NULL,
    matched         INTEGER NOT NULL,   -- bool
    confidence      REAL NOT NULL,
    spans_json      TEXT NOT NULL,      -- loss_mask_spans
    created_at      TEXT NOT NULL,
    PRIMARY KEY (sub_problem_id, trajectory_id, slice_index)
);
```

```python
class JudgeCache(Protocol):
    def get(self, sub_problem_id: str, trajectory_id: str, slice_index: int) -> dict | None: ...
    def put(self, sub_problem_id: str, trajectory_id: str, slice_index: int, verdict: dict, *, created_at: str) -> None: ...
```

**成对稳定性论证**：「切片 S 是否命中子问题 P 的能力」这一精判判断，只取决于 S 的内容与 P 的 `target_capability/trajectory_signal`，与索引里是否存在其它轨迹无关。故缓存**永不失效**——无版本号、无全量清空。惰性填充（决策3）：`search` 精判前先 `get`，miss 才 `judge` 并 `put`。

> **注意**：`sub_problem_id` 必须**全局稳定且唯一**。现状 orchestrator 用 `L{行号}.{sp.id}` 前缀防同批碰撞（见 §8 开放点1），但跨 run 会重复。持久化后须改用 `problem_id` 派生前缀（如 `{problem_id}.{sp.id}`），否则不同问题的子项撞键、缓存污染。

### 5.4 HTTP 接口

```
POST /search
  json: { "question": string }
  → 200 { "run_id": string }              # 立即返回，后台跑；复用 SSE 进度机制
  # 结果经 GET /runs/{id}/view 取回，附加 dedup 信息见 5.5

POST /ingest
  multipart/form-data: manifest(file, .txt, 可选) + trajectories(file, .jsonl, 可选)
  → 200 { "run_id": string }              # 至少一个文件；两个都缺 → 400
  # 进度阶段: ingest_traj（切片+签名+写库）/ ingest_manifest（去重+编译+写库）→ done

GET /stats
  → 200 { "problems": N, "trajectories": M, "signatures": K }   # header 状态条

# 沿用现有：GET /runs/{id}/events (SSE) · GET /runs/{id}/view · GET /runs/{id}/trajectory/{tid}
```

`/search` 与 `/ingest` 都当成一次「run」，完全复用现有 `RunStore` + SSE + 进度条 + `/view` 机制（决策9），前端渲染零改动。

### 5.5 InspectorView 扩展（向后兼容）

在现有 `InspectorView`（见旧 spec §5.2）顶层**新增可选字段**，老前端忽略即兼容：

```jsonc
{
  "run_id": string,
  "mode": "search" | "ingest",           // 新增
  "dedup": {                              // 新增，仅 search 且命中已有问题时出现
    "matched_problem_id": string,
    "matched_question": string,           // 原始问题原文，前端飘"≈ 与已有问题重复"
    "similarity": number                  // cosine
  } | null,
  "problems": [ ... ],                    // 结构不变
  "manifest": { ... }
}
```

`/ingest` 的 view 可只带一条汇总（写入了多少问题/轨迹/跳过多少重复），不含三栏 problems。

## 6. 阈值 τ_q（问题去重）

- 起步 **0.90**（保守，宁可漏判重也不误判重——误判重会把不同问题的结果张冠李戴）。
- 校准方法对齐 `module0.5 挂载阈值` 的做法：真实 Qwen 下取一批「语义等价问题对」与「相似但不等价问题对」，测 cosine 分布，定在两簇之间。校准脚本仿 `scripts/calibrate_mount_threshold.py`。
- 暴露为配置（`UXFLOW_QUESTION_DEDUP_THRESHOLD` env / CLI），不硬编码。

## 7. Pipeline 改造（薄封装，不碰内部算法）

现状 `run_scored` 把「`_reset_store` + `_build_index` + score 循环」焊死。拆成三段可独立调用：

| 新入口 | 复用现有 | 变化 |
|---|---|---|
| `ingest_trajectories(paths)` | `_build_index` | `store_factory` 注入 `SqliteSliceStore`；**去掉 `_reset_store()`**；额外写 `TrajectoryStore` |
| `search(specs)` | `run_scored` 的 score 循环（`_score_sub_problem`） | **跳过 `_reset_store`/`_build_index`**；judge 前查 `JudgeCache`，miss 才 judge 并 `put` |
| `_build_index` | 原样 | 增写一份轨迹全文到 `TrajectoryStore`（`load_trajectories` 结果已在手） |

`TrajectoryPipeline(store_factory=...)` seam 已存在 —— 注入 `SqliteSliceStore(db_path)` 即持久化，pipeline 内部零改动。这正是既有「V1 设计须留产品级接口余量」原则的兑现：seam 都在，只是从没在生产线上用过。

`_score_sub_problem` 的 judge 段改造（惟一实质改动，详见 D3）：
```
按 (sp_id, traj_id, slice_idx) 查 JudgeCache → 拆 cached / miss 两组
只对 miss 的 slices 调现有 judge_batch → 结果 put 回缓存
cached 项经 dict→JudgeResult 适配器还原，按原 recall 顺序与 miss 结果合并
合并后喂给现有 rerank（rerank 逻辑不变；缓存命中项跳过 update_labels，幂等安全）
```
> 实现细节：先按 cache 命中过滤出 miss 子集，只对 miss 批量 judge，再与命中项**保序**合并——保持批处理效率，同时零重复。`update_labels` 在缓存命中时跳过（首次入缓存时已写，且其 `dict.fromkeys` 合并幂等）。

## 8. 编排层改造（orchestrator.py）

- 新增 `run_search(question, *, deps, emit, run_id)`：`embed → ProblemStore.nearest`；命中取 `spec_json`（emit「命中已有问题」），未命中经 `get_or_compile`（持久 nearest→进程 LRU→compile，emit 编译进度，**不写 ProblemStore**，仅写 LRU，见 D2）→ `pipeline.search(specs)` → `select` → `build_inspector_view`（附 `dedup`）。
- 新增 `run_ingest(manifest_lines?, trajectory_path?, *, deps, emit, run_id)`：
  - 有轨迹：`pipeline.ingest_trajectories([path])`（emit 切片/签名/写库进度）。
  - 有清单：每行 `embed → nearest`，重复则 emit「跳过重复」，新则 `compile → ProblemStore.add`（emit 编译进度）。
  - emit 汇总 done。
- `sub_problem_id` 前缀从 `L{行号}.` 改为 `{problem_id}.`（§5.3 注意项），保证跨 run 稳定唯一。

## 9. 前端（src/service/web/，原生单页）

现有 header（两上传槽 + ▶运行）改为**双模式**：

```
┌ 🔍 [ 输入新用户问题…………………… ]  [ 搜索 ]          → POST /search
├ 📥 [用户清单.txt] [回流轨迹.jsonl]  [ 入库 ]         → POST /ingest（两槽各自可空）
└ 📊 库中 N 问题 · M 轨迹 · K 切片                     → GET /stats
```

- 搜索/入库都走现有 SSE 进度条（复用 stepper，阶段文案切换）。
- 搜索结果：**三栏渲染完全复用**（问题→能力 / 命中轨迹 / 详情）。命中已有问题时，顶部飘一条「≈ 与已有问题重复：`<matched_question>`（相似度 0.93）」（读 `view.dedup`）。
- 详情栏 `/trajectory/{id}` 改由 `TrajectoryStore` 供数据，重启后仍可展开。
- 入库完成：状态条刷新 `/stats`，飘一条「入库 X 问题 / Y 轨迹，跳过 Z 重复」。

## 10. 生产线接线（scripts/inspector_serve.py）

```python
db = resolve_db_path(); ensure_parent(db)
slice_store_factory = lambda: SqliteSliceStore(db)      # 注入持久索引
problem_store = SqliteProblemStore(db)
traj_store    = SqliteTrajectoryStore(db)
judge_cache   = SqliteJudgeCache(db)
pipeline = TrajectoryPipeline(config=cfg, gateway=gw, store_factory=slice_store_factory)
deps = PipelineDeps(compiler=..., pipeline=pipeline, select_fn=..., load_trajectories_fn=...,
                    problem_store=problem_store, trajectory_store=traj_store, judge_cache=judge_cache)
```
所有 Store 共用**同一个 db 文件**、各开一条 `sqlite3.connect(isolation_level=None, WAL)` 连接（对齐现有 `SqliteSliceStore` 写法）。V1 单写者，autocommit 足够。

## 11. 非目标 / 后续增强

- **入库时预补精判（写时增量）**：本设计取惰性（决策3）。若日后用法变成"入库后批量回看所有老问题"，可加一个 `backfill_judge` 离线入口，复用 `JudgeCache.put`——缓存结构不变。
- **多写者并发**：V1 单写者 + WAL；多进程写入需加写锁或迁移到真正的 DB seam（Store Protocol 已留）。
- **问题 spec 版本演化**：taxonomy 演化后老 `spec_json` 可能引用旧标签。V1 不重编译历史问题；后续可加 spec 版本号 + 惰性重编译。**注意**：taxonomy 变更**不影响** `JudgeCache` 成对稳定性（判断的是"切片是否命中该子问题"，与标签树无关），但会影响 `spec_json` 的 `target_capability` 取值——重编译触发新 `sub_problem_id`，自然走新缓存键，不污染旧缓存。
- **软删除/编辑问题清单**：V1 只增不删；`problems` 表可后加 `deleted_at`。
- **鉴权/限流**：内部工具，V1 不做。

## 12. 已定稿决策（原开放点，2026-07-14 拍板）

**D1 · `problem_id` = normalized sha1 前 16 位。**
规范化：`strip + 折叠内部空白 + Unicode NFC`，**不 lowercase**（SWE 问题里 `Bash`/`bash`、大小写标识符可能有意义，lowercase 有误合风险）。理由：hash 与 embedding 模型**无关**，提供一层"模型无关的逐字重复地板"——换 embedding 模型后旧向量与新向量 cosine 失去意义、τ_q 语义去重会失效，此时 hash 仍能拦逐字重复。两层去重互补：**hash 挡逐字重复，τ_q 挡语义重复**。
> 注：同一问题重复 `/ingest` 的幂等性主要由 τ_q `nearest()` 命中后**返回已有 problem_id** 保证（向量层就拦掉，走不到 `add`）；hash 只影响"全新问题首次插入"这一路径。

**D2 · 新问题 spec 加进程内存 LRU（cap 128，重启丢）。**
把所有 compile 收口到一个 `get_or_compile(question)`，顺序**严格**为 `持久 ProblemStore.nearest → 进程 LRU → compile`。**LRU 永不替代持久去重检查**，故去重语义零影响，纯性能层。staleness 不存在：taxonomy 在 `inspector_serve.py` 启动时只加载一次、运行期不变。附带收益：`/search` 把新问题 compile 进 LRU 后，紧接着 `/ingest` 同一问题直接命中、免二次编译。

**D3 · judge 惰性缓存的 miss 拆分走 pipeline 层过滤（不碰 module1 内部）。**
`_score_sub_problem` 内：按 `(sp_id, traj_id, slice_idx)` 查 `JudgeCache` → 拆 `cached / miss` 两组 → 只把 miss 的 slices 传给现有 `judge_batch` → 结果 `put` 回缓存 → cached 项经**适配器 `dict → JudgeResult`** 还原 → 按原 recall 顺序合并 → 喂给现有 `rerank`（rerank 零改动）。
两个连带点（plan 需钉死）：
- **需要 `dict → JudgeResult` 适配器**：`rerank` 吃 JudgeResult 对象且依赖原顺序，合并时必须保序。
- **`update_labels` 副作用**：现状 match 时写 `capability_labels` 到签名。**缓存命中时跳过**——首次精判入缓存时已写过，且该写用 `dict.fromkeys` 合并、本身幂等，跳过安全、省一次写。

**D4 · `/ingest` 部分失败：跨文件不回滚 + 清单逐行错误隔离。**
- 前提：compile 只依赖 taxonomy、**不依赖轨迹**，故轨迹与清单两条子通路无依赖、无需跨文件原子性。
- 跨文件不回滚：轨迹 upsert（按 key 幂等）、问题 add（hash/τ_q 去重）各自幂等，失败重试 = 轨迹 no-op + 已加清单行被去重跳过 + 断点续跑。
- 清单**逐行 try/except**：第 N 行 compile 失败不毁其余行，捕获→emit→继续。对齐 module0.5「失败可读、不抛穿」。
- 结尾 emit 汇总 `{added, skipped_dup, failed}`。
- 事务：每个 Store 各自 autocommit 连接（对齐现有 `SqliteSliceStore`），本无多语句事务，V1 单写者足够。
