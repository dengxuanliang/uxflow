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
    assert deps.problem_store is not old_problem
    assert pipe._store is mgr.slice_store


def test_switch_to_missing_path_creates_empty_db(tmp_path):
    mgr, deps, pipe = _mgr(tmp_path, "a.db")
    deps.problem_store.add("q", [1.0, 0.0], {"k": "v"})
    mgr.switch(tmp_path / "fresh.db")
    assert deps.problem_store.count() == 0
    assert (tmp_path / "fresh.db").exists()


def test_clear_current_deletes_and_recreates_empty(tmp_path):
    mgr, deps, pipe = _mgr(tmp_path, "a.db")
    deps.problem_store.add("q", [1.0, 0.0], {"k": "v"})
    assert deps.problem_store.count() == 1
    mgr.clear_current()
    assert deps.problem_store.count() == 0
    assert (tmp_path / "a.db").exists()


def test_list_databases_marks_current(tmp_path):
    mgr, deps, pipe = _mgr(tmp_path, "a.db")
    import sqlite3
    sqlite3.connect(str(tmp_path / "b.db")).close()
    listing = mgr.list_databases()
    names = {d["name"]: d for d in listing}
    assert names["a.db"]["is_current"] is True
    assert names["b.db"]["is_current"] is False
    assert names["a.db"]["problems"] == 0
