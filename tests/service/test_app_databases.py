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
    # 占满信号量模拟"运行中"。Semaphore(1) 被 acquire 后 locked() 为 True；
    # asyncio.Semaphore.locked() 与事件循环无关，可在无运行 loop 时安全置位。
    asyncio.run(mgr._run_lock.acquire())
    assert mgr._run_lock.locked() is True
    client = TestClient(app)
    r = client.post("/databases/switch", json={"path": str(tmp_path / "b.db")})
    assert r.status_code == 409


def test_databases_endpoint_absent_without_manager(tmp_path):
    app = create_app()
    client = TestClient(app)
    r = client.get("/databases")
    assert r.status_code in (404, 501)
