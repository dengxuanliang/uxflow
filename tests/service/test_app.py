# SPDX-License-Identifier: Apache-2.0
import asyncio

from fastapi.testclient import TestClient

from service.app import create_app
from service.runstore import MemoryRunStore


async def _fake_run(*, manifest_lines, trajectory_path, deps, emit, run_id,
                    selection_config=None, general_config=None):
    emit({"stage": "module0", "status": "running", "msg": "x"})
    emit({"stage": "module3", "status": "running", "msg": "y"})
    view = {
        "run_id": run_id,
        "problems": [{"id": "L1.p1", "failure_summary": "s",
                      "confidence": 0.9, "capabilities": []}],
        "manifest": {"targeted_count": 0, "general_count": 0, "general_ratio": 0.3},
    }
    trajectories = {"t1": {"trajectory_id": "t1", "steps": [
        {"index": 0, "role": "user", "content": "hi",
         "tool_call_name": None, "tool_call_args": None, "tool_result": None}]}}
    return view, trajectories


def _client():
    store = MemoryRunStore()
    app = create_app(store=store, run_fn=_fake_run, deps=object())
    return TestClient(app), store


def test_post_runs_returns_run_id():
    client, _ = _client()
    resp = client.post(
        "/runs",
        files={
            "manifest": ("m.txt", "代码总有语法错误\n改完不验证", "text/plain"),
            "trajectories": ("t.jsonl", '{"id":"t1","messages":[]}\n',
                             "application/x-ndjson"),
        },
    )
    assert resp.status_code == 200
    assert "run_id" in resp.json()


def test_view_available_after_run_completes():
    client, store = _client()
    rid = client.post("/runs", files={
        "manifest": ("m.txt", "x", "text/plain"),
        "trajectories": ("t.jsonl", '{"id":"t1","messages":[]}\n', "application/x-ndjson"),
    }).json()["run_id"]

    # background task runs in the TestClient's event loop; poll view
    for _ in range(50):
        r = client.get(f"/runs/{rid}/view")
        if r.status_code == 200:
            break
    assert r.status_code == 200
    body = r.json()
    assert body["run_id"] == rid
    assert body["problems"][0]["id"] == "L1.p1"


def test_view_409_before_done():
    client, store = _client()
    rid = store.create()  # created but never run → still "running"
    r = client.get(f"/runs/{rid}/view")
    assert r.status_code == 409


def test_view_404_unknown_run():
    client, _ = _client()
    r = client.get("/runs/nope/view")
    assert r.status_code == 404


def test_trajectory_endpoint_returns_steps():
    client, _ = _client()
    rid = client.post("/runs", files={
        "manifest": ("m.txt", "x", "text/plain"),
        "trajectories": ("t.jsonl", '{"id":"t1","messages":[]}\n', "application/x-ndjson"),
    }).json()["run_id"]
    for _ in range(50):
        if client.get(f"/runs/{rid}/view").status_code == 200:
            break
    r = client.get(f"/runs/{rid}/trajectory/t1")
    assert r.status_code == 200
    assert r.json()["steps"][0]["content"] == "hi"


def test_trajectory_404_unknown():
    client, _ = _client()
    rid = client.post("/runs", files={
        "manifest": ("m.txt", "x", "text/plain"),
        "trajectories": ("t.jsonl", '{"id":"t1","messages":[]}\n', "application/x-ndjson"),
    }).json()["run_id"]
    for _ in range(50):
        if client.get(f"/runs/{rid}/view").status_code == 200:
            break
    r = client.get(f"/runs/{rid}/trajectory/missing")
    assert r.status_code == 404
