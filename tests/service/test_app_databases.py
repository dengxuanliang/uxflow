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


def test_safe_count_rejects_unknown_table():
    """_safe_count must refuse any table name outside the allowlist,
    even if a caller tried to inject SQL via the table name."""
    import sqlite3
    from service.database import _safe_count
    conn = sqlite3.connect(":memory:")
    try:
        # benign unknown name -> None
        assert _safe_count(conn, "users") is None
        # malicious input -> None (no SQL injection surface)
        assert _safe_count(conn, "problems; DROP TABLE x") is None
        assert _safe_count(conn, "' OR 1=1 --") is None
        # allowlisted names still work (table doesn't exist -> OperationalError -> None)
        assert _safe_count(conn, "problems") is None
    finally:
        conn.close()


def _make_sqlite(path):
    """Create an empty SQLite file with a tiny dummy table."""
    import sqlite3
    conn = sqlite3.connect(str(path))
    conn.execute("CREATE TABLE IF NOT EXISTS problems (id INTEGER PRIMARY KEY)")
    conn.commit()
    conn.close()


def test_switch_rejects_dotdot_escape(tmp_path):
    """Relative path with .. must not escape the current DB's parent dir."""
    app, mgr = _app(tmp_path)
    client = TestClient(app)
    # malicious: try to write into a sibling dir via ../
    r = client.post("/databases/switch", json={"path": "../../../tmp/evil.db"})
    assert r.status_code == 400
    assert "逃逸" in r.json()["detail"]


def test_switch_rejects_absolute_path_outside_root(tmp_path):
    """Absolute path outside the DB root dir must be rejected."""
    app, mgr = _app(tmp_path)
    client = TestClient(app)
    # /tmp is not under tmp_path/a.db's parent
    r = client.post("/databases/switch", json={"path": "/etc/uxflow.db"})
    assert r.status_code == 400


def test_switch_rejects_non_db_extension(tmp_path):
    """Path with non-.db suffix must be rejected even if inside root."""
    app, mgr = _app(tmp_path)
    client = TestClient(app)
    r = client.post("/databases/switch", json={"path": str(tmp_path / "notes.txt")})
    assert r.status_code == 400
    assert ".db" in r.json()["detail"]


def test_switch_rejects_existing_non_sqlite_file(tmp_path):
    """Existing non-empty non-SQLite file must be rejected to avoid clobbering."""
    app, mgr = _app(tmp_path)
    client = TestClient(app)
    target = tmp_path / "not_a_db.db"
    target.write_bytes(b"NOT A DATABASE" * 100)
    r = client.post("/databases/switch", json={"path": str(target)})
    assert r.status_code == 400
    assert "SQLite" in r.json()["detail"]


def test_switch_accepts_relative_path_in_same_dir(tmp_path):
    """Relative path under the current DB's parent dir is accepted."""
    app, mgr = _app(tmp_path)
    client = TestClient(app)
    r = client.post("/databases/switch", json={"path": "c.db"})
    assert r.status_code == 200
    assert mgr.current().name == "c.db"
