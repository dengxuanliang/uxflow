# SPDX-License-Identifier: Apache-2.0
"""Tests for /search, /ingest, /stats and the persistent trajectory fallback."""
import json as _json
import pathlib

from fastapi.testclient import TestClient

from service.app import create_app
from service.runstore import MemoryRunStore


async def _fake_search(question, *, deps, emit, run_id, tau_q=0.90,
                       selection_config=None, general_config=None):
    emit({"stage": "search0", "status": "running", "msg": "分析"})
    emit({"stage": "search1", "status": "running", "msg": "检索 1/1",
          "index": 1, "total": 1})
    # run_search does NOT emit done; _bg_search must add it.
    return {"run_id": run_id, "mode": "search", "dedup": None,
            "problems": [], "manifest": {}}, {}


async def _fake_ingest(*, manifest_lines=None, trajectory_path=None, deps, emit,
                       run_id, tau_q=0.90):
    _fake_ingest.seen = {"manifest_lines": manifest_lines,
                         "trajectory_path": trajectory_path}
    emit({"stage": "done", "status": "ok", "msg": "入库完成"})  # run_ingest emits done
    return {"mode": "ingest",
            "summary": {"added": 1, "skipped_dup": 0, "failed": 0}}, {}


class _CountStore:
    def __init__(self, n, traj=None):
        self._n = n
        self._traj = traj or {}

    def count(self):
        return self._n

    def get(self, tid):
        return self._traj.get(tid)


class _Pipeline:
    def __init__(self, size):
        self._store = type("S", (), {"size": size})()


class _FakeDeps:
    def __init__(self, *, problems=0, trajectories=0, signatures=0, traj=None):
        self.problem_store = _CountStore(problems)
        self.trajectory_store = _CountStore(trajectories, traj)
        self.pipeline = _Pipeline(signatures)


def _client(*, deps=None):
    store = MemoryRunStore()
    app = create_app(store=store, search_fn=_fake_search, ingest_fn=_fake_ingest,
                     deps=deps if deps is not None else object())
    return TestClient(app), store


def _poll_view(client, rid):
    for _ in range(50):
        r = client.get(f"/runs/{rid}/view")
        if r.status_code == 200:
            return r
    return r


def test_search_empty_question_400():
    client, _ = _client()
    r = client.post("/search", json={"question": "   "})
    assert r.status_code == 400


def test_search_normal_produces_search_view():
    client, _ = _client()
    rid = client.post("/search", json={"question": "如何修复"}).json()["run_id"]
    r = _poll_view(client, rid)
    assert r.status_code == 200
    assert r.json()["mode"] == "search"


def test_search_stream_ends_with_done():
    client, _ = _client()
    rid = client.post("/search", json={"question": "q"}).json()["run_id"]
    events = []
    with client.stream("GET", f"/runs/{rid}/events") as resp:
        assert resp.status_code == 200
        for line in resp.iter_lines():
            if line.startswith("data:"):
                events.append(_json.loads(line[len("data:"):].strip()))
                if events[-1].get("stage") == "done":
                    break
    stages = [e["stage"] for e in events]
    assert "search1" in stages
    assert stages[-1] == "done"
    assert events[-1]["status"] == "ok"
    # exactly one done event in the stream
    assert stages.count("done") == 1


def test_ingest_both_missing_400():
    client, _ = _client()
    r = client.post("/ingest")
    assert r.status_code == 400


def test_ingest_manifest_only():
    client, _ = _client()
    rid = client.post("/ingest", files={
        "manifest": ("m.txt", "问题一\n问题二", "text/plain"),
    }).json()["run_id"]
    r = _poll_view(client, rid)
    assert r.status_code == 200
    body = r.json()
    assert body["mode"] == "ingest"
    assert body["summary"]["added"] == 1
    assert _fake_ingest.seen["manifest_lines"] is not None
    assert _fake_ingest.seen["trajectory_path"] is None


def test_ingest_trajectory_only():
    client, _ = _client()
    rid = client.post("/ingest", files={
        "trajectories": ("t.jsonl", '{"id":"t1","messages":[]}\n',
                         "application/x-ndjson"),
    }).json()["run_id"]
    r = _poll_view(client, rid)
    assert r.status_code == 200
    assert r.json()["mode"] == "ingest"
    assert _fake_ingest.seen["trajectory_path"] is not None
    assert _fake_ingest.seen["manifest_lines"] is None


def test_ingest_non_utf8_manifest_400():
    client, _ = _client()
    r = client.post("/ingest", files={
        "manifest": ("m.txt", b"\xff\xfe\x00bad", "text/plain"),
    })
    assert r.status_code == 400


def test_stats_reports_counts():
    deps = _FakeDeps(problems=3, trajectories=7, signatures=5)
    client, _ = _client(deps=deps)
    r = client.get("/stats")
    assert r.status_code == 200
    assert r.json() == {"problems": 3, "trajectories": 7, "signatures": 5}


def test_stats_defensive_with_bare_object_deps():
    client, _ = _client()  # deps = object()
    r = client.get("/stats")
    assert r.status_code == 200
    assert r.json() == {"problems": 0, "trajectories": 0, "signatures": 0}


def test_trajectory_persistent_fallback():
    steps = [{"index": 0, "role": "user", "content": "hi"}]
    deps = _FakeDeps(traj={"t1": {"trajectory_id": "t1", "steps": steps}})
    client, _ = _client(deps=deps)
    rid = client.post("/search", json={"question": "q"}).json()["run_id"]
    _poll_view(client, rid)  # search returns empty in-memory trajectories
    r = client.get(f"/runs/{rid}/trajectory/t1")
    assert r.status_code == 200
    assert r.json()["steps"][0]["content"] == "hi"


def test_trajectory_fallback_miss_404():
    deps = _FakeDeps(traj={})
    client, _ = _client(deps=deps)
    rid = client.post("/search", json={"question": "q"}).json()["run_id"]
    _poll_view(client, rid)
    r = client.get(f"/runs/{rid}/trajectory/missing")
    assert r.status_code == 404


def test_ingest_temp_jsonl_cleaned_up():
    seen = {}

    async def _capturing_ingest(*, manifest_lines=None, trajectory_path=None,
                                deps, emit, run_id, tau_q=0.90):
        seen["path"] = pathlib.Path(trajectory_path)
        assert seen["path"].exists()
        emit({"stage": "done", "status": "ok", "msg": "ok"})
        return {"mode": "ingest",
                "summary": {"added": 0, "skipped_dup": 0, "failed": 0}}, {}

    store = MemoryRunStore()
    app = create_app(store=store, ingest_fn=_capturing_ingest, deps=object())
    client = TestClient(app)
    rid = client.post("/ingest", files={
        "trajectories": ("t.jsonl", '{"id":"t1","messages":[]}\n',
                         "application/x-ndjson"),
    }).json()["run_id"]
    for _ in range(50):
        if store.status(rid) == "done":
            break
        client.get(f"/runs/{rid}/view")
    assert store.status(rid) == "done"
    assert not seen["path"].exists()
