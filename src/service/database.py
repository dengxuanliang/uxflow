"""运行时 SQLite 库管理：列举 / 切换 / 清除（spec §7）。

DatabaseManager 持有当前库路径 + 4 个 store + pipeline 引用。切换/清除通过原地
改写 deps 字段与 pipeline._store 生效（端点/orchestrator 请求时才读这些字段）。
"""
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 dengxuanliang

from __future__ import annotations

import pathlib
import sqlite3

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

    def current(self) -> pathlib.Path:
        return self._path

    def attach_lock(self, run_lock) -> None:
        """create_app 建锁后回注（见接线方案 B）。"""
        self._run_lock = run_lock

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
        return None
