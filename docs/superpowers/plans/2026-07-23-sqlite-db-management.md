# SQLite 库管理（热切 + 清除）Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让用户在运行时列出 / 切换 / 清空当前入库的 SQLite `.db`，无需重启服务；切换/清除在有任务运行时被拒绝，删库不可逆前端二次确认。

**Architecture:** 新增 `service/database.py::DatabaseManager` 持有"当前库路径 + 4 个 Sqlite store + pipeline 引用 + run_lock"，提供 `list_databases/switch/clear_current`。切换通过**原地改写 `deps` 字段 + `pipeline._store`**（端点/orchestrator 请求时才读这些字段，故立即生效）。`app.py` 加三个 `/databases*` 端点（运行中返回 409）。`inspector_serve.py` 构造 manager 并把 `store_factory` 改为 `lambda: mgr.slice_store`，`create_app` 建锁后 `attach_lock`。前端 header 加库控件。`PipelineDeps` dataclass 不改。

**Tech Stack:** FastAPI，sqlite3，pytest（含 ASGI TestClient），原生 JS/CSS。

**参考 spec:** `docs/superpowers/specs/2026-07-23-dedup-evidence-db-management-design.md` §4、§7。

---

## 关键前提（实现前已核验，见 spec §4）

- 4 个 store 均以 `db_path` 构造、均有 `close()`：`SqliteSliceStore`（`module1/sqlite_store.py`）、`SqliteProblemStore`/`SqliteTrajectoryStore`/`SqliteJudgeCache`（`service/stores.py`）。连接时 `executescript(schema)` 自动建表 → 切到不存在路径即新建空库。
- `app.py` 端点与 orchestrator **请求时**读 `deps.problem_store` 等 → 原地改字段即生效。
- `pipeline` 的 `search`/`ingest` 路径**不 reset store**，直接用 `self._store`；`run_scored`/`run` 才调 `store_factory`。→ 切库必须同时改 `pipeline._store`，不能只换 factory。
- `_run_lock`（`app.py:52`）是 `create_app` 内 `asyncio.Semaphore(1)`；`locked()` 可判占用。

---

## File Structure

- **Create** `src/service/database.py` — `DatabaseManager`。
- **Modify** `src/service/app.py` — `create_app` 加 `db_manager=None` 参数、`attach_lock`、三个端点。
- **Modify** `scripts/inspector_serve.py` — 构造 manager、`store_factory` 改闭包、注入。
- **Modify** `src/service/web/index.html` — header 库控件 DOM。
- **Modify** `src/service/web/app.js` — `loadDatabases/switchDatabase/clearDatabase`。
- **Modify** `src/service/web/style.css` — 库控件样式。
- **Test** `tests/service/test_database.py`（新）、`tests/service/test_app_databases.py`（新，端点）。

---

## Task 1: DatabaseManager 核心（open/close/switch/clear）

**Files:**
- Create: `src/service/database.py`
- Test: `tests/service/test_database.py`

- [ ] **Step 1: 写失败测试**

创建 `tests/service/test_database.py`：

```python
# SPDX-License-Identifier: Apache-2.0
import asyncio
from dataclasses import dataclass
from typing import Any

from service.database import DatabaseManager


@dataclass
class FakeDeps:
    problem_store: Any = None
    trajectory_store: Any = None
    judge_cache: Any = None


class FakePipeline:
    def __init__(self):
        self._store = None


def _mgr(tmp_path, name="a.db"):
    deps = FakeDeps()
    pipe = FakePipeline()
    lock = asyncio.Semaphore(1)
    mgr = DatabaseManager(tmp_path / name, deps, pipe, run_lock=lock)
    return mgr, deps, pipe


def test_open_binds_stores_into_deps_and_pipeline(tmp_path):
    mgr, deps, pipe = _mgr(tmp_path)
    assert deps.problem_store is not None
    assert deps.trajectory_store is not None
    assert deps.judge_cache is not None
    assert pipe._store is mgr.slice_store
    assert mgr.current() == tmp_path / "a.db"


def test_switch_rebinds_to_new_db(tmp_path):
    mgr, deps, pipe = _mgr(tmp_path, "a.db")
    old_problem = deps.problem_store
    mgr.switch(tmp_path / "b.db")
    assert mgr.current() == tmp_path / "b.db"
    assert deps.problem_store is not old_problem     # 新连接
    assert pipe._store is mgr.slice_store             # pipeline 同步换


def test_switch_to_missing_path_creates_empty_db(tmp_path):
    mgr, deps, pipe = _mgr(tmp_path, "a.db")
    deps.problem_store.add("q", [1.0, 0.0], {"k": "v"})
    mgr.switch(tmp_path / "fresh.db")
    assert deps.problem_store.count() == 0            # 新空库
    assert (tmp_path / "fresh.db").exists()


def test_clear_current_deletes_and_recreates_empty(tmp_path):
    mgr, deps, pipe = _mgr(tmp_path, "a.db")
    deps.problem_store.add("q", [1.0, 0.0], {"k": "v"})
    assert deps.problem_store.count() == 1
    mgr.clear_current()
    assert deps.problem_store.count() == 0
    assert (tmp_path / "a.db").exists()               # 重建
```

- [ ] **Step 2: 跑确认失败**

Run: `uv run pytest tests/service/test_database.py -v`
Expected: FAIL（`ModuleNotFoundError: No module named 'service.database'`）

- [ ] **Step 3: 实现 `src/service/database.py`**

```python
"""运行时 SQLite 库管理：列举 / 切换 / 清除（spec §7）。

DatabaseManager 持有当前库路径 + 4 个 store + pipeline 引用。切换/清除通过原地
改写 deps 字段与 pipeline._store 生效（端点/orchestrator 请求时才读这些字段）。
"""
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 dengxuanliang

from __future__ import annotations

import pathlib
import sqlite3
from typing import Any

from module1.sqlite_store import SqliteSliceStore
from service.stores import (
    SqliteJudgeCache,
    SqliteProblemStore,
    SqliteTrajectoryStore,
)

__all__ = ["DatabaseManager"]


class DatabaseManager:
    def __init__(self, db_path, deps, pipeline, *, run_lock):
        self._path = pathlib.Path(db_path)
        self._deps = deps
        self._pipeline = pipeline
        self._run_lock = run_lock
        self.slice_store = None
        self._open(self._path)

    # ── 状态 ──
    def current(self) -> pathlib.Path:
        return self._path

    def attach_lock(self, run_lock) -> None:
        """create_app 建锁后回注（见接线方案 B）。"""
        self._run_lock = run_lock

    # ── 私有：装配 / 拆卸 ──
    def _open(self, path) -> None:
        path = pathlib.Path(path)
        self.slice_store = SqliteSliceStore(path)
        self._deps.problem_store = SqliteProblemStore(path)
        self._deps.trajectory_store = SqliteTrajectoryStore(path)
        self._deps.judge_cache = SqliteJudgeCache(path)
        self._pipeline._store = self.slice_store
        self._path = path

    def _close_all(self) -> None:
        for s in (self.slice_store, self._deps.problem_store,
                  self._deps.trajectory_store, self._deps.judge_cache):
            try:
                if s is not None:
                    s.close()
            except Exception:  # noqa: BLE001 — 尽力释放句柄，删文件前不因单个 close 失败中断
                pass

    # ── 操作 ──
    def switch(self, path) -> None:
        self._close_all()
        self._open(path)

    def clear_current(self) -> None:
        p = self._path
        self._close_all()
        for suffix in ("", "-wal", "-shm"):
            pathlib.Path(str(p) + suffix).unlink(missing_ok=True)
        self._open(p)

    def list_databases(self) -> list[dict]:
        results = []
        for f in sorted(self._path.parent.glob("*.db")):
            is_current = (f == self._path)
            problems = trajectories = None
            try:
                if is_current:
                    problems = self._deps.problem_store.count()
                    trajectories = self._deps.trajectory_store.count()
                else:
                    conn = sqlite3.connect(str(f))
                    try:
                        problems = _safe_count(conn, "problems")
                        trajectories = _safe_count(conn, "trajectories")
                    finally:
                        conn.close()
            except Exception:  # noqa: BLE001 — 损坏库不该让整个列举失败
                pass
            results.append({
                "name": f.name,
                "path": str(f),
                "size_bytes": f.stat().st_size if f.exists() else 0,
                "problems": problems,
                "trajectories": trajectories,
                "is_current": is_current,
            })
        return results


def _safe_count(conn, table) -> int | None:
    try:
        return conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    except sqlite3.OperationalError:
        return None   # 表不存在（非 uxflow 库）
```

- [ ] **Step 4: 跑确认通过**

Run: `uv run pytest tests/service/test_database.py -v`
Expected: PASS（4 passed）

- [ ] **Step 5: 加 list_databases 测试并跑**

追加：

```python
def test_list_databases_marks_current(tmp_path):
    mgr, deps, pipe = _mgr(tmp_path, "a.db")
    # 造第二个库文件
    import sqlite3
    sqlite3.connect(str(tmp_path / "b.db")).close()
    listing = mgr.list_databases()
    names = {d["name"]: d for d in listing}
    assert names["a.db"]["is_current"] is True
    assert names["b.db"]["is_current"] is False
    assert names["a.db"]["problems"] == 0
```

Run: `uv run pytest tests/service/test_database.py -v`
Expected: PASS（5 passed）

- [ ] **Step 6: 提交**

```bash
git add src/service/database.py tests/service/test_database.py
git commit -m "feat(service): DatabaseManager — 运行时 SQLite 库列举/切换/清除

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Task 2: app.py 三个端点（运行中 409）

**Files:**
- Modify: `src/service/app.py`（`create_app` 签名 + `attach_lock` + 三端点）
- Test: `tests/service/test_app_databases.py`

- [ ] **Step 1: 写失败测试**

创建 `tests/service/test_app_databases.py`：

```python
# SPDX-License-Identifier: Apache-2.0
import asyncio
from dataclasses import dataclass
from typing import Any

from fastapi.testclient import TestClient

from service import create_app
from service.database import DatabaseManager


@dataclass
class FakeDeps:
    problem_store: Any = None
    trajectory_store: Any = None
    judge_cache: Any = None


class FakePipeline:
    def __init__(self):
        self._store = None


def _app(tmp_path):
    deps = FakeDeps()
    pipe = FakePipeline()
    mgr = DatabaseManager(tmp_path / "a.db", deps, pipe,
                          run_lock=asyncio.Semaphore(1))
    app = create_app(deps=deps, db_manager=mgr)
    return app, mgr


def test_list_databases_endpoint(tmp_path):
    app, mgr = _app(tmp_path)
    client = TestClient(app)
    r = client.get("/databases")
    assert r.status_code == 200
    body = r.json()
    assert body["current"].endswith("a.db")
    assert any(d["name"] == "a.db" for d in body["databases"])


def test_switch_endpoint(tmp_path):
    app, mgr = _app(tmp_path)
    client = TestClient(app)
    r = client.post("/databases/switch", json={"path": str(tmp_path / "b.db")})
    assert r.status_code == 200
    assert mgr.current().name == "b.db"


def test_clear_endpoint(tmp_path):
    app, mgr = _app(tmp_path)
    client = TestClient(app)
    r = client.post("/databases/clear")
    assert r.status_code == 200


def test_switch_rejected_when_running(tmp_path):
    app, mgr = _app(tmp_path)
    # 占满信号量模拟"运行中"。Semaphore(1) 被 acquire 后 value=0，locked() 为 True；
    # asyncio.Semaphore.locked() 与事件循环无关，可在无运行 loop 时安全置位。
    asyncio.run(mgr._run_lock.acquire())
    assert mgr._run_lock.locked() is True
    client = TestClient(app)
    r = client.post("/databases/switch", json={"path": str(tmp_path / "b.db")})
    assert r.status_code == 409


def test_databases_endpoint_absent_without_manager(tmp_path):
    app = create_app()   # 无 db_manager
    client = TestClient(app)
    r = client.get("/databases")
    assert r.status_code in (404, 501)
```

- [ ] **Step 2: 跑确认失败**

Run: `uv run pytest tests/service/test_app_databases.py -v`
Expected: FAIL（`create_app` 无 `db_manager` 参数 / 无 `/databases` 路由）

- [ ] **Step 3: 改 `create_app` 签名 + attach_lock**

`src/service/app.py` 的 `create_app` 增加参数（放在 `tau_q` 后）：

```python
def create_app(
    *,
    store: Any = None,
    run_fn: Callable | None = None,
    search_fn: Callable | None = None,
    ingest_fn: Callable | None = None,
    deps: Any = None,
    tau_q: float = 0.90,
    db_manager: Any = None,
) -> FastAPI:
```

在 `_run_lock = asyncio.Semaphore(1)` 定义之后，加：

```python
    # 库管理器与 run_lock 共享同一把锁（切换/清除前判占用）。接线方案B：
    # manager 在 inspector_serve 构造，此处回注锁。
    if db_manager is not None:
        db_manager.attach_lock(_run_lock)
```

- [ ] **Step 4: 加三个端点**

在 `app.py` 静态挂载（`app.mount("/", ...)`）**之前**加：

```python
    @app.get("/databases")
    async def list_databases():
        if db_manager is None:
            raise HTTPException(status_code=501, detail="库管理未启用")
        return {"current": str(db_manager.current()),
                "databases": db_manager.list_databases()}

    @app.post("/databases/switch")
    async def switch_database(payload: dict):
        if db_manager is None:
            raise HTTPException(status_code=501, detail="库管理未启用")
        path = (payload.get("path") or "").strip()
        if not path:
            raise HTTPException(status_code=400, detail="库路径不能为空")
        if _run_lock.locked():
            raise HTTPException(status_code=409, detail="有任务运行中，无法切换库")
        # 相对路径按当前库同目录解析
        import pathlib
        p = pathlib.Path(path)
        if not p.is_absolute():
            p = db_manager.current().parent / p
        db_manager.switch(p)
        return {"current": str(db_manager.current()),
                "databases": db_manager.list_databases()}

    @app.post("/databases/clear")
    async def clear_database():
        if db_manager is None:
            raise HTTPException(status_code=501, detail="库管理未启用")
        if _run_lock.locked():
            raise HTTPException(status_code=409, detail="有任务运行中，无法清除库")
        db_manager.clear_current()
        return {"current": str(db_manager.current()),
                "databases": db_manager.list_databases()}
```

- [ ] **Step 5: 跑确认通过**

Run: `uv run pytest tests/service/test_app_databases.py -v`
Expected: PASS（5 passed）

- [ ] **Step 6: 回归既有 app 测试（确认 db_manager=None 时行为不变）**

Run: `uv run pytest -m "not requires_model" tests/service/test_app.py tests/service/test_app_search_ingest.py -v`
Expected: PASS（全部，行为不变）

- [ ] **Step 7: 提交**

```bash
git add src/service/app.py tests/service/test_app_databases.py
git commit -m "feat(service): /databases 列举/切换/清除端点（运行中 409，无 manager 时 501）

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Task 3: inspector_serve 接线

**Files:**
- Modify: `scripts/inspector_serve.py`

- [ ] **Step 1: 改 store_factory 为闭包 + 构造 manager + 注入**

`scripts/inspector_serve.py` 的 `build_app()` 中：

1. 保留 `slice_store = SqliteSliceStore(db)` 作为初始 store（manager `_open` 会重建同路径的一份，可接受；或直接删这行让 manager 建——见下）。
2. `pipeline` 的 `store_factory` 改为读 manager：

现状：
```python
    slice_store = SqliteSliceStore(db)
    ...
    pipeline = TrajectoryPipeline(
        config=cfg, gateway=gateway, store_factory=lambda: slice_store)
    deps = PipelineDeps(
        compiler=compiler, pipeline=pipeline, select_fn=select_final_dataset,
        load_trajectories_fn=load_trajectories,
        problem_store=SqliteProblemStore(db),
        trajectory_store=SqliteTrajectoryStore(db),
        judge_cache=SqliteJudgeCache(db), embedder=emb)
```

改为：
```python
    # store_factory 延迟读 manager.slice_store：热切库后 factory 自然返回新实例。
    _mgr_holder = {}
    pipeline = TrajectoryPipeline(
        config=cfg, gateway=gateway,
        store_factory=lambda: _mgr_holder["mgr"].slice_store)
    deps = PipelineDeps(
        compiler=compiler, pipeline=pipeline, select_fn=select_final_dataset,
        load_trajectories_fn=load_trajectories,
        problem_store=None, trajectory_store=None, judge_cache=None, embedder=emb)

    from service.database import DatabaseManager
    import asyncio
    # 临时锁占位，create_app 会 attach 真正的锁；构造时 _open 会填 deps 三个 store
    mgr = DatabaseManager(db, deps, pipeline, run_lock=asyncio.Semaphore(1))
    _mgr_holder["mgr"] = mgr
```

3. `create_app` 调用加 `db_manager=mgr`：
```python
    app = create_app(store=store, deps=deps, tau_q=tau_q, db_manager=mgr)
```

删除原来独立的 `slice_store = SqliteSliceStore(db)` 行（manager 的 `_open` 已建 slice_store 并设 `pipeline._store`）。注意 `TrajectoryPipeline.__init__` 会调一次 `store_factory`（`self._store = self._store_factory()`）——此时 `_mgr_holder["mgr"]` 尚未赋值。**解决**：调整顺序——先建 manager（它会 `pipeline._store = slice_store`），或让 pipeline 构造时 `store_factory` 容忍。最简单：`TrajectoryPipeline` 构造时传 `store_factory=lambda: _mgr_holder.get("mgr").slice_store if _mgr_holder.get("mgr") else SqliteSliceStore(db)`。

**定稿顺序**（避免闭包时序问题）：
```python
    pipeline = TrajectoryPipeline(config=cfg, gateway=gateway,
                                  store_factory=lambda: SqliteSliceStore(db))
    deps = PipelineDeps(..., problem_store=None, trajectory_store=None,
                        judge_cache=None, embedder=emb)
    mgr = DatabaseManager(db, deps, pipeline, run_lock=asyncio.Semaphore(1))
    # manager 已把 pipeline._store 指向自己的 slice_store；现在把 factory 也指向 manager
    pipeline._store_factory = lambda: mgr.slice_store
    app = create_app(store=store, deps=deps, tau_q=tau_q, db_manager=mgr)
```

- [ ] **Step 2: 冒烟——服务能起且 /databases 可访问**

若环境可启动服务：
Run: `.venv/bin/python -c "from scripts.inspector_serve import build_app; app=build_app(); print([r.path for r in app.routes if 'database' in r.path])"`
Expected: 打印含 `/databases`、`/databases/switch`、`/databases/clear`。

（此步不需真实 LLM，只构造 app 对象；若 import 路径需调整用 `PYTHONPATH=src`。）

- [ ] **Step 3: 提交**

```bash
git add scripts/inspector_serve.py
git commit -m "feat(serve): 接入 DatabaseManager，store_factory 延迟读 manager.slice_store

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Task 4: 前端库控件

**Files:**
- Modify: `src/service/web/index.html`
- Modify: `src/service/web/app.js`
- Modify: `src/service/web/style.css`

- [ ] **Step 1: index.html 加库控件 DOM**

在 `<div id="stats-bar" class="stats-bar"></div>` 之后（`</div>` 前，仍在 `.modes` 内）加：

```html
      <div id="db-bar" class="db-bar">
        <span class="db-label">库</span>
        <select id="db-select" class="db-select"></select>
        <input type="text" id="db-new-path" class="db-new-path" placeholder="新建/切换到路径…" />
        <button id="db-switch" class="db-btn" type="button">切换</button>
        <button id="db-clear" class="db-btn db-clear" type="button">🗑 清空当前库</button>
      </div>
```

- [ ] **Step 2: app.js 加库管理逻辑**

在 `loadStats()` 定义附近新增：

```javascript
// ── 库管理（spec §7）───────────────────────────────────
async function loadDatabases() {
  try {
    const resp = await fetch("/databases");
    if (!resp.ok) { $("db-bar").classList.add("hidden"); return; }
    const { current, databases } = await resp.json();
    const sel = $("db-select");
    sel.innerHTML = "";
    for (const d of databases) {
      const opt = document.createElement("option");
      opt.value = d.path;
      const cnt = d.problems == null ? "?" : d.problems;
      opt.textContent = `${d.name} (${cnt}问题)` + (d.is_current ? " · 当前" : "");
      if (d.is_current) opt.selected = true;
      sel.appendChild(opt);
    }
  } catch (_) { /* best-effort */ }
}

async function switchDatabase(path) {
  if (!path) return;
  const resp = await fetch("/databases/switch", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ path }),
  });
  if (resp.status === 409) { alert("有任务运行中，无法切换库"); return; }
  if (!resp.ok) { alert("切换失败: HTTP " + resp.status); return; }
  afterDbChange();
}

async function clearDatabase() {
  if (!confirm("将永久删除当前库文件，不可恢复，确认？")) return;
  const resp = await fetch("/databases/clear", { method: "POST" });
  if (resp.status === 409) { alert("有任务运行中，无法清除库"); return; }
  if (!resp.ok) { alert("清除失败: HTTP " + resp.status); return; }
  afterDbChange();
}

function afterDbChange() {
  $("workspace").classList.add("hidden");
  hideDedupBanner();
  state.view = null;
  state.trajCache = {};
  loadStats();
  loadDatabases();
}
```

绑定事件（放在文件底部现有 `loadStats();` 附近）：

```javascript
$("db-select").addEventListener("change", (e) => switchDatabase(e.target.value));
$("db-switch").addEventListener("click", () => {
  const p = $("db-new-path").value.trim();
  if (p) switchDatabase(p);
});
$("db-clear").addEventListener("click", clearDatabase);

loadStats();
loadDatabases();
```

（删除文件末尾原有的单独 `loadStats();`，避免重复调用——保留上面这组即可。）

- [ ] **Step 3: style.css 加库控件样式**

追加到 `style.css` 末尾：

```css
/* ── 库管理控件（spec §7）──────────────────────────── */
.db-bar {
  display: flex;
  align-items: center;
  gap: 6px;
  margin-top: 6px;
  font-size: 12px;
  color: #fff;
}
.db-bar .db-label { opacity: .8; }
.db-select, .db-new-path {
  font-size: 12px;
  padding: 2px 6px;
  border-radius: 4px;
  border: 1px solid var(--line-2);
  background: rgba(255,255,255,.9);
  color: var(--fg);
}
.db-new-path { width: 180px; }
.db-btn {
  font-size: 12px;
  padding: 2px 8px;
  border-radius: 4px;
  border: 1px solid var(--line-2);
  background: rgba(255,255,255,.85);
  color: var(--fg);
  cursor: pointer;
}
.db-btn:hover { background: #fff; }
.db-clear { color: var(--hot); border-color: var(--hot); }
```

- [ ] **Step 4: 手动冒烟（需可启动服务）**

Run: `.venv/bin/python scripts/inspector_serve.py`，浏览器 `http://localhost:8000`。
Expected: header 出现"库"控件，下拉列出数据目录下 `.db`（当前库标"· 当前"）；切换到新路径 → stats 归零、下拉刷新；点"🗑 清空当前库"→ confirm → 清空后 stats 归零；运行中点切换/清空 → alert"有任务运行中"。

若无 LLM 后端，跳过，靠 Task 2/3 的自动化测试 + code review。

- [ ] **Step 5: 前端静态测试（若 test_inspector_frontend_static 断言 DOM）**

Run: `uv run pytest -m "not requires_model" tests/service/test_inspector_frontend_static.py -v`
Expected: PASS（若该测试校验特定 DOM，补断言 `db-bar` 存在）。

- [ ] **Step 6: 提交**

```bash
git add src/service/web/index.html src/service/web/app.js src/service/web/style.css
git commit -m "feat(inspector-web): header 库控件（列举/切换/清空，删库二次确认）

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## 验收标准（对齐 spec §10）

- [ ] `GET /databases` 列出目录下所有 `.db` 并标记当前库。
- [ ] 切到不存在路径 → 自动建空库并生效（stats 归零）。
- [ ] 切到已有库 → search/ingest 立即对新库生效（`pipeline._store` 已换）。
- [ ] 有任务运行中调 switch/clear → 409，运行不受影响。
- [ ] clear 删文件后重建空库，`-wal`/`-shm` 不残留。
- [ ] 不传 `db_manager` 的现有测试（`tests/service/test_app*.py`）全绿，行为不变。
