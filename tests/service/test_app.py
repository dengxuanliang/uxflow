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


async def _failing_run(*, manifest_lines, trajectory_path, deps, emit, run_id,
                       selection_config=None, general_config=None):
    emit({"stage": "module0", "status": "running", "msg": "x"})
    raise RuntimeError("boom")


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


def test_events_stream_ends_with_done():
    client, _ = _client()
    rid = client.post("/runs", files={
        "manifest": ("m.txt", "x", "text/plain"),
        "trajectories": ("t.jsonl", '{"id":"t1","messages":[]}\n', "application/x-ndjson"),
    }).json()["run_id"]
    events = []
    with client.stream("GET", f"/runs/{rid}/events") as resp:
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        for line in resp.iter_lines():
            if line.startswith("data:"):
                import json as _json
                events.append(_json.loads(line[len("data:"):].strip()))
                if events[-1].get("stage") == "done":
                    break
    stages = [e["stage"] for e in events]
    assert "module0" in stages
    assert stages[-1] == "done"
    assert events[-1]["status"] == "ok"


def test_events_404_unknown_run():
    client, _ = _client()
    with client.stream("GET", "/runs/nope/events") as resp:
        assert resp.status_code == 404


def test_background_error_marks_run_error_and_view_409():
    store = MemoryRunStore()
    app = create_app(store=store, run_fn=_failing_run, deps=object())
    client = TestClient(app)
    rid = client.post("/runs", files={
        "manifest": ("m.txt", "x", "text/plain"),
        "trajectories": ("t.jsonl", '{"id":"t1","messages":[]}\n', "application/x-ndjson"),
    }).json()["run_id"]
    # Poll until the background task drives the run into a terminal state.
    # Repeated client.get calls yield control so the background task runs.
    for _ in range(50):
        if store.status(rid) == "error":
            break
        client.get(f"/runs/{rid}/view")
    assert store.status(rid) == "error"
    r = client.get(f"/runs/{rid}/view")
    assert r.status_code == 409  # error run has no view → 409


def test_temp_jsonl_is_cleaned_up_after_run():
    import pathlib

    seen = {}

    async def _capturing_run(*, manifest_lines, trajectory_path, deps, emit,
                             run_id, selection_config=None, general_config=None):
        seen["path"] = pathlib.Path(trajectory_path)
        assert seen["path"].exists()  # temp file present while run consumes it
        return {"run_id": run_id}, {}

    store = MemoryRunStore()
    app = create_app(store=store, run_fn=_capturing_run, deps=object())
    client = TestClient(app)
    rid = client.post("/runs", files={
        "manifest": ("m.txt", "x", "text/plain"),
        "trajectories": ("t.jsonl", '{"id":"t1","messages":[]}\n', "application/x-ndjson"),
    }).json()["run_id"]
    for _ in range(50):
        if store.status(rid) == "done":
            break
        client.get(f"/runs/{rid}/view")
    assert store.status(rid) == "done"
    assert not seen["path"].exists()  # cleaned up in _background_run finally


def test_non_utf8_manifest_returns_400():
    client, _ = _client()
    resp = client.post("/runs", files={
        "manifest": ("m.txt", b"\xff\xfe\x00bad", "text/plain"),
        "trajectories": ("t.jsonl", '{"id":"t1","messages":[]}\n', "application/x-ndjson"),
    })
    assert resp.status_code == 400


def test_cancel_unknown_run_404():
    client, _ = _client()
    r = client.post("/runs/nope/cancel")
    assert r.status_code == 404


def test_cancel_stops_run_and_marks_error():
    # A run_fn that blocks lets us cancel mid-flight; the CancelledError branch
    # emits a terminal "cancelled" event and marks the run so /view returns 409.
    async def _blocking_run(*, manifest_lines, trajectory_path, deps, emit,
                            run_id, selection_config=None, general_config=None):
        emit({"stage": "module1", "status": "running", "msg": "召回中..."})
        await asyncio.sleep(10)  # long; cancelled before it finishes
        return {"run_id": run_id}, {}

    store = MemoryRunStore()
    app = create_app(store=store, run_fn=_blocking_run, deps=object())
    client = TestClient(app)
    rid = client.post("/runs", files={
        "manifest": ("m.txt", "x", "text/plain"),
        "trajectories": ("t.jsonl", '{"id":"t1","messages":[]}\n', "application/x-ndjson"),
    }).json()["run_id"]

    # let the background task start and reach the blocking await
    for _ in range(20):
        if store.events_snapshot(rid):
            break
        client.get(f"/runs/{rid}/view")

    r = client.post(f"/runs/{rid}/cancel")
    assert r.status_code == 200
    assert r.json()["cancelled"] is True

    # background task's CancelledError handler runs on the loop; poll for terminal
    for _ in range(50):
        if store.status(rid) == "error":
            break
        client.get(f"/runs/{rid}/view")
    assert store.status(rid) == "error"
    events = store.events_snapshot(rid)
    assert events[-1]["status"] == "cancelled"
    assert client.get(f"/runs/{rid}/view").status_code == 409


def test_cancel_finalizes_and_cleans_up_via_done_callback():
    import pathlib as _pl
    seen = {}

    async def _blocking_run(*, manifest_lines, trajectory_path, deps, emit,
                            run_id, selection_config=None, general_config=None):
        seen["path"] = _pl.Path(trajectory_path)
        emit({"stage": "module1", "status": "running", "msg": "..."})
        await asyncio.sleep(10)
        return {"run_id": run_id}, {}

    store = MemoryRunStore()
    app = create_app(store=store, run_fn=_blocking_run, deps=object())
    client = TestClient(app)
    rid = client.post("/runs", files={
        "manifest": ("m.txt", "x", "text/plain"),
        "trajectories": ("t.jsonl", '{"id":"t1","messages":[]}\n', "application/x-ndjson"),
    }).json()["run_id"]
    for _ in range(50):
        if store.events_snapshot(rid):
            break
        client.get(f"/runs/{rid}/view")
    client.post(f"/runs/{rid}/cancel")
    for _ in range(50):
        if store.status(rid) == "error":
            break
        client.get(f"/runs/{rid}/view")
    assert store.status(rid) == "error"
    # temp file cleaned by _background_run finally and/or _finalize (idempotent)
    assert not seen["path"].exists()


def test_concurrent_runs_are_serialized():
    import asyncio as _aio
    log = []

    async def _recording_run(*, manifest_lines, trajectory_path, deps, emit,
                             run_id, selection_config=None, general_config=None):
        log.append(("enter", run_id))
        await _aio.sleep(0.05)   # hold long enough to detect overlap
        log.append(("exit", run_id))
        return {"run_id": run_id, "problems": [], "manifest": {}}, {}

    store = MemoryRunStore()
    app = create_app(store=store, run_fn=_recording_run, deps=object())
    files = {
        "manifest": ("m.txt", "x", "text/plain"),
        "trajectories": ("t.jsonl", '{"id":"t1","messages":[]}\n', "application/x-ndjson"),
    }
    # NOTE: must use `with TestClient(app)` here — a bare TestClient(app) tears
    # down its anyio portal/event loop after every single request, which
    # cancels the still-sleeping background task between polls and makes this
    # test pass trivially (both runs end up "error", never "done"). The `with`
    # form keeps one event loop alive across requests so the two background
    # tasks genuinely race for the semaphore, which is what this test needs
    # to actually distinguish serialized from interleaved execution.
    with TestClient(app) as client:
        r1 = client.post("/runs", files=files).json()["run_id"]
        r2 = client.post("/runs", files=files).json()["run_id"]
        # drive the event loop until both runs reach terminal status
        for _ in range(300):
            if store.status(r1) == "done" and store.status(r2) == "done":
                break
            client.get(f"/runs/{r1}/view")
            client.get(f"/runs/{r2}/view")
        assert store.status(r1) == "done" and store.status(r2) == "done"
    # serialization invariant: no enter happens between another run's enter and exit
    # i.e. the sequence must be enter,exit,enter,exit — never enter,enter
    kinds = [k for (k, _rid) in log]
    assert kinds == ["enter", "exit", "enter", "exit"], f"runs overlapped: {log}"
