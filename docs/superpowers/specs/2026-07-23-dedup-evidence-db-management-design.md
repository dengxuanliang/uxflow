# 入选去重 · 证据折叠 · SQLite 库管理 设计

- 状态：已确认（2026-07-23 拍板），待实现
- 分支：`feat/rubric-capability-card`（沿用当前分支或另开）
- 影响面：`module3/`（去重+优选）、`service/`（viewmodel、app、新增 database 管理）、`service/web/`（前端）、`scripts/inspector_serve.py`（接线）

## 1. 背景与动机

三处独立诉求，合并为一个 spec：

1. **最终入选 SFT 样本未跨子问题去重。** 同一 `(trajectory_id, slice_index)` 被多个子问题各自选中时，`compose_dataset` 的 `targeted_count = len(targeted)` 会把它算作多条。用户预期：两个子问题各选中"同一条"轨迹 slice → 最终入选应为 **1 条**，而非 2 条。
2. **命中卡上"决定性证据"直接内联展开。** `renderHits()` 把 `⭐ 决定性证据: step X · 命中判据: …` 直接渲染在每张命中卡上，信息密度高、遮挡列表浏览。改为默认收起、点"证据"按钮就地展开。
3. **无法在运行时管理 SQLite 库。** DB 路径在 `build_app()` 启动时解析一次，4 个 store 绑死到单一连接。用户需要：列出/切换当前入库的 `.db`、清空当前库。

## 2. 已确认决策

| # | 决策点 | 结论 |
|---|--------|------|
| D1 | 诉求1 去重语义 | **数据层去重 + 合并 loss_mask 掩码**（并集），不是纯显示修复 |
| D2 | 诉求1 合并时机 | **选择前合并**：`N` 等于唯一样本数（先塌缩成唯一 slice 集，再 `select_set`） |
| D3 | 诉求2 展开方式 | **行内 accordion**：命中卡下方就地展开/收起，不弹 modal |
| D4 | 诉求3 范围 | **清除 + 运行时热切换**（不重启服务） |
| D5 | 诉求3 库发现 | **扫描 db 所在目录列出 `*.db`**；新建库=切换到不存在的路径（连接即建空库） |
| D6 | 诉求3 清除语义 | **删文件重建**（连同 `-wal`/`-shm`），前端二次确认，运行中拒绝 |

## 3. 铁律 / 护栏

- **模块依赖方向不变**：诉求1 的改动全部收敛在 `module3/` + `service/viewmodel.py`，不引入新的跨模块依赖。
- **热切换绝不 mid-run**：切换/清除库前抢 `_run_lock`；有任务在跑则拒绝（返回 409），不破坏正在执行的 pipeline 状态。
- **删文件前先释放句柄**：删除 `.db` 前 `close()` 所有相关连接，避免 Windows/NFS 句柄占用与 WAL 残留。
- **删除高风险**：前端"清空当前库"必须二次确认；删文件不可逆。
- **测试注入路径不受影响**：诉求3 通过**原地改写 `deps` 字段**实现热切，不改 `PipelineDeps` dataclass 定义，`create_app` 的直接依赖注入（测试用 fakes）行为不变。

## 4. 关键事实（实现前已核验）

- 一个候选是 `ScoredCandidate`，键为 `(trajectory_id, slice_index, sub_problem_id)`。
- 同一 slice 跨子问题时：`embedding` / `bm25_tokens` **相同**（都来自 `hit.signature`）；`relevance_score` / `loss_mask_spans` / `evidence_step` / `criteria_hit` **各子问题不同**（来自 per-sub_problem 的 judge 结果）。→ 合并必须"并集掩码 + 按子问题保留证据"，不能简单取一条丢其余。
- `deduplicate` 现仅在**同一 sub_problem_id 内**去近重（cosine>0.95 / minhash≥0.9）。
- `select_set` 按子问题选，含 `min_per_problem` 与覆盖度（`cover`/`counts` 按单个 `sub_problem_id` 记账）。
- `app.py` 端点与 `orchestrator` 都在**请求时**读 `deps.problem_store` 等字段（非启动时捕获），因此原地改写 `deps` 字段即可让热切生效。
- `pipeline` 的 slice_store 经 `store_factory=lambda: slice_store` 注入，热切需让该闭包指向新实例。

## 5. 诉求 1：跨子问题去重 + 合并掩码

### 5.1 新增 `MergedCandidate`（module3）

在 `module3` 引入"多归属候选"，把共享 `(trajectory_id, slice_index)` 的 `ScoredCandidate` 塌缩成一个：

| 字段 | 来源 | 说明 |
|------|------|------|
| `trajectory_id` / `slice_index` | 相同 | 合并键 |
| `sub_problem_ids: list[str]` | 各候选的 `sub_problem_id` 并集 | 多归属 |
| `loss_mask_spans` | 各子问题掩码**并集** | D1 |
| `relevance_by_problem: dict[str, float]` | 每个子问题的 `relevance_score` | 供覆盖度记账 |
| `relevance_score` | `max(relevance_by_problem.values())` | 供全局排序/多样性增益 |
| `embedding` / `bm25_tokens` | 任一候选（相同） | 直接取 |
| `capability` | 各候选 capability 并集 | 显示用 |
| `evidence_by_problem: dict[str, {evidence_step, criteria_hit}]` | 按子问题保留 | 诉求2 分组展开 |

### 5.2 module3 流水线改造

`select_final_dataset` 顺序改为：

```
trainable → merge_by_slice → deduplicate → select_set → compose_dataset
```

- **`merge_by_slice(candidates)`**（新函数）：按 `(trajectory_id, slice_index)` 分组塌缩为 `MergedCandidate`。这是 D2"选择前合并"的落点。
- **`selection.py` 覆盖度记账多归属化**：`counts` / `cover` / `min_per_problem` 从"读单个 `candidate.sub_problem_id`"改为"遍历 `candidate.sub_problem_ids`"。一个 slice 被选中即**同时**满足它覆盖的所有子问题的配额与覆盖度。`_diversity_gain` 不变（用单一 `embedding`）。覆盖度增益 `coverage_gain` 用 `relevance_by_problem[sub_id]` 分子问题累加。
- **`dedup.py` 近重塌缩吸收归属**：幸存者吸收被塌缩者的 `sub_problem_ids`（并集）与掩码，避免丢覆盖；原 `sub_problem_id ==` 守卫改为"归属集合有交集"。
- **`compose_dataset`**：`targeted` 现在是 `MergedCandidate` 列表，`targeted_count = len(targeted)` 自然=唯一样本数。
- **预算口径**：orchestrator 默认 `SelectionConfig(n=min(10, max(1, len(scored))))` 里的 `scored` 是合并**前**的数，应改为按合并后的唯一 slice 数计（否则 `n` 可能大于可选样本数，`select_set` 内 `budget=min(n, len(candidates))` 虽会兜底，但 `n` 语义应对齐"唯一样本"）。

### 5.3 viewmodel 改造

- `selected_keys` 从 `(trajectory_id, slice_index, sub_problem_id)` 改为 `(trajectory_id, slice_index)`（`MergedCandidate` 无单一 sub_problem_id）。→ 同一 slice 在它命中的所有子问题卡片下都标"已入选"，且"最终入选 N 条"= 唯一 slice 数。
- 每个 sub_hit 的 `evidence_step` / `criteria_hit` 从 `MergedCandidate.evidence_by_problem[sp_id]` 取，保证证据按子问题正确对应（诉求2 用）。

### 5.4 受影响测试

`tests/module3/test_selection.py`、`test_dedup.py`、`test_pipeline.py`、`test_compose.py`、`tests/service/test_viewmodel.py` 需同步更新（覆盖度/配额记账语义变化、计数口径变化）。新增：多子问题共享 slice 时计数=唯一数、掩码并集、"已入选"跨卡片一致 的用例。

## 6. 诉求 2：证据折叠（行内 accordion）

纯前端，改 `service/web/app.js` + `style.css`，`index.html` 不动。

- `renderHits()`（app.js:531-542）：不再内联渲染证据行。命中卡上放一个 `证据` 小按钮，**仅当** `evidence_step != null` 或 `criteria_hit` 非空时显示。
- 点击按钮 → `e.stopPropagation()`（防止误触 `selectTrajectory`）→ 在该卡片下方就地展开/收起证据面板。
- 面板内容：决定性证据步 + 命中判据；多归属（同一 slice 命中多个子问题）时**按子问题分组**列出，数据来自 5.3 保留的按子问题证据。
- 详情栏（`renderDetail`）里 `⭐ 决定性证据` 步标星逻辑保持不变。

## 7. 诉求 3：SQLite 库管理（热切 + 清除）

### 7.1 新增 `service/database.py` 的 `DatabaseManager`

单一职责：持有"当前库路径 + 4 个 store 实例"，并提供列举/切换/清除。

| 方法 | 行为 |
|------|------|
| `current()` | 返回当前库路径 |
| `list_databases()` | 扫描当前库所在目录下所有 `*.db`，返回 `[{name, path, size, problems, trajectories, is_current}]` |
| `switch(path)` | 抢锁 → `close()` 旧 4 连接 → 建新 store 绑定新库 → **原地更新 `deps` 字段** + 自身 `slice_store`（`store_factory` 闭包随之指向新实例）→ 释放锁。路径不存在则连接时自动建空库（=新建库） |
| `clear_current()` | 抢锁 → `close()` 连接 → 删当前 `.db` 及 `-wal`/`-shm` → 重建空库（连接时自动建表）→ 释放锁 |

- `slice_store` 经 `store_factory = lambda: mgr.slice_store` 喂 pipeline；热切后 factory 自然返回新实例。
- `store_factory` 需从"闭包捕获固定 slice_store"改为"每次读 `mgr.slice_store`"——这是 `inspector_serve.py` 接线处的唯一改动。

### 7.2 新增 HTTP 端点（app.py）

```
GET  /databases          → { current, databases: [...] }
POST /databases/switch   { path }   → 200 切换成功 / 409 有任务运行中
POST /databases/clear                → 200 已清除重建 / 409 有任务运行中
```

- 切换/清除前检查 `_run_lock.locked()`；锁被占用 → 409 + 提示"有任务运行中，无法切换/清除库"。
- 无需单独 create 端点：新建库 = switch 到不存在的路径。
- 均为 `async def`（与既有 store-touching 端点一致，避免 SQLite 跨线程问题）。

### 7.3 前端（header 新增"库"控件）

- header 内（stats-bar 旁）加一块库控件：下拉列出 `GET /databases` 返回的 `.db`（当前库高亮），选中即 `POST /databases/switch`。
- "＋ 新建/自定义"输入框：填新路径 → switch（不存在则建空库）。
- "🗑 清空当前库"按钮 → **二次确认弹窗**（复用现有 modal 风格或 `confirm()`）→ `POST /databases/clear`。
- 切换/清除成功后：刷新 stats-bar（`loadStats()`）、重置工作区。
- 运行中收到 409 → 提示"有任务运行中"。

### 7.4 安全护栏

- 删文件前 `close()` 释放所有句柄。
- 前端删除强制二次确认（删文件不可逆，高风险）。
- 后端运行中拒绝切换/删除。

## 8. 非目标 / 后续增强

- 库的多用户并发隔离、权限控制（V1 单用户本地服务）。
- 库之间的数据迁移/合并。
- `min_turns` 等其它选择配置的调参（沿用现值）。
- 通用数据（general data）加载仍未实现（`compose_dataset` 现状保留）。

## 9. 实现顺序建议

1. 诉求2（纯前端、零风险、独立）。
2. 诉求1（module3 + viewmodel + 测试；数据层核心改动）。
3. 诉求3（新增 database 管理 + 端点 + 前端 + 接线）。

三块彼此独立，可分别成 PR。
