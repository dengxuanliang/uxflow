# SPDX-License-Identifier: Apache-2.0
import asyncio

from service.runstore import MemoryRunStore


def test_create_returns_unique_ids():
    store = MemoryRunStore()
    a = store.create()
    b = store.create()
    assert a != b
    assert store.status(a) == "running"


def test_append_and_read_events():
    store = MemoryRunStore()
    rid = store.create()
    store.append_event(rid, {"stage": "module0", "status": "running", "msg": "x"})
    store.append_event(rid, {"stage": "done", "status": "ok"})
    events = list(store.events_snapshot(rid))
    assert len(events) == 2
    assert events[0]["stage"] == "module0"
    assert events[-1]["stage"] == "done"


def test_set_and_get_view():
    store = MemoryRunStore()
    rid = store.create()
    store.set_view(rid, {"run_id": rid, "problems": []}, {"t1": {"steps": []}})
    assert store.get_view(rid)["run_id"] == rid
    assert store.get_trajectory(rid, "t1") == {"steps": []}
    assert store.get_trajectory(rid, "missing") is None


def test_status_transitions():
    store = MemoryRunStore()
    rid = store.create()
    assert store.status(rid) == "running"
    store.mark_done(rid)
    assert store.status(rid) == "done"

    rid2 = store.create()
    store.mark_error(rid2)
    assert store.status(rid2) == "error"


def test_unknown_run_returns_none():
    store = MemoryRunStore()
    assert store.get_view("nope") is None
    assert store.status("nope") is None


async def test_subscribe_receives_events_appended_after_subscribe():
    store = MemoryRunStore()
    rid = store.create()

    received = []

    async def consumer():
        async for ev in store.subscribe(rid):
            received.append(ev)
            if ev.get("stage") == "done":
                break

    task = asyncio.create_task(consumer())
    await asyncio.sleep(0.01)  # let consumer start
    store.append_event(rid, {"stage": "module0", "status": "running"})
    store.append_event(rid, {"stage": "done", "status": "ok"})
    await asyncio.wait_for(task, timeout=1.0)

    assert any(e.get("stage") == "module0" for e in received)
    assert received[-1]["stage"] == "done"


async def test_subscribe_replays_existing_events():
    store = MemoryRunStore()
    rid = store.create()
    store.append_event(rid, {"stage": "module0", "status": "running"})
    store.append_event(rid, {"stage": "done", "status": "ok"})

    received = []
    async for ev in store.subscribe(rid):
        received.append(ev)
        if ev.get("stage") == "done":
            break
    assert len(received) == 2  # 订阅前已有的事件也要重放
