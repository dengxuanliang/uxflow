"""FastAPI service for the Trajectory Inspector (spec §5.1)."""
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 dengxuanliang

from __future__ import annotations

import asyncio
import json
import pathlib
import tempfile
from typing import Any, Callable

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from service.runstore import MemoryRunStore

__all__ = ["create_app"]

_WEB_DIR = pathlib.Path(__file__).parent / "web"


def create_app(
    *,
    store: Any = None,
    run_fn: Callable | None = None,
    deps: Any = None,
) -> FastAPI:
    """Build the app. Inject store/run_fn/deps for testing; defaults for prod."""
    store = store or MemoryRunStore()
    if run_fn is None:
        from service.orchestrator import run_pipeline
        run_fn = run_pipeline

    app = FastAPI(title="Trajectory Inspector")

    # Track in-flight background tasks so a run can be cancelled (see /cancel).
    _tasks: dict[str, asyncio.Task] = {}
    # Serialize actual pipeline execution: the injected compiler/pipeline hold
    # per-run mutable state (self._store is reset each run_scored), so concurrent
    # runs would corrupt each other. A run waits its turn rather than interleave.
    _run_lock = asyncio.Semaphore(1)

    async def _background_run(run_id: str, manifest_text: str, traj_path: pathlib.Path):
        def emit(ev: dict) -> None:
            store.append_event(run_id, ev)
        try:
            # If a prior run holds the lock, tell the user we're queued rather
            # than leaving the UI frozen at "上传中".
            if _run_lock.locked():
                emit({"stage": "module0", "status": "running",
                      "msg": "前一个任务运行中，排队等待…"})
            async with _run_lock:      # serialize pipeline execution (C1)
                lines = manifest_text.splitlines()
                view, trajectories = await run_fn(
                    manifest_lines=lines,
                    trajectory_path=traj_path,
                    deps=deps,
                    emit=emit,
                    run_id=run_id,
                )
                store.set_view(run_id, view, trajectories)
                store.append_event(run_id, {"stage": "done", "status": "ok"})
                store.mark_done(run_id)
        except asyncio.CancelledError:
            # User stopped the run: emit a terminal event so the SSE stream
            # unblocks, mark the run so /view returns 409. Do not re-raise —
            # the task ends cleanly after finally cleanup.
            store.append_event(run_id, {"stage": "done", "status": "cancelled",
                                        "msg": "已停止"})
            store.mark_error(run_id)
        except Exception as exc:  # noqa: BLE001 — surface as error event, don't crash server
            store.append_event(run_id, {"stage": "done", "status": "error",
                                        "msg": str(exc)})
            store.mark_error(run_id)
        finally:
            # Clean up the uploaded jsonl temp file once the run has consumed it
            # (success or failure). Only safe here — create_run must not delete it
            # before the background task reads it.
            traj_path.unlink(missing_ok=True)
            _tasks.pop(run_id, None)

    @app.post("/runs")
    async def create_run(manifest: UploadFile, trajectories: UploadFile):
        manifest_text = (await manifest.read()).decode("utf-8")
        traj_bytes = await trajectories.read()
        # persist uploaded jsonl to a temp file for the loader/pipeline
        tmp = tempfile.NamedTemporaryFile(
            delete=False, suffix=".jsonl", mode="wb")
        tmp.write(traj_bytes)
        tmp.close()
        run_id = store.create()
        traj_path = pathlib.Path(tmp.name)
        task = asyncio.create_task(
            _background_run(run_id, manifest_text, traj_path)
        )
        _tasks[run_id] = task

        def _finalize(_t: asyncio.Task, rid: str = run_id, path: pathlib.Path = traj_path) -> None:
            # Safety net that runs no matter how the task ended — including the
            # narrow window where cancel() is delivered before the coroutine body
            # ever executed (so _background_run's try/finally never ran). All
            # actions here are idempotent, so they compose safely with that finally.
            path.unlink(missing_ok=True)          # idempotent (missing_ok)
            _tasks.pop(rid, None)                  # idempotent
            if store.status(rid) == "running":     # body never finalized → force terminal
                store.append_event(rid, {"stage": "done", "status": "cancelled",
                                         "msg": "已停止"})
                store.mark_error(rid)

        task.add_done_callback(_finalize)
        return {"run_id": run_id}

    @app.post("/runs/{run_id}/cancel")
    async def cancel_run(run_id: str):
        if store.status(run_id) is None:
            raise HTTPException(status_code=404, detail="unknown run")
        task = _tasks.get(run_id)
        if task is not None and not task.done():
            task.cancel()  # _background_run's CancelledError handler finalizes
        return {"run_id": run_id, "cancelled": True}

    @app.get("/runs/{run_id}/events")
    async def stream_events(run_id: str):
        if store.status(run_id) is None:
            raise HTTPException(status_code=404, detail="unknown run")

        async def gen():
            async for ev in store.subscribe(run_id):
                yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"

        return StreamingResponse(gen(), media_type="text/event-stream")

    @app.get("/runs/{run_id}/view")
    async def get_view(run_id: str):
        status = store.status(run_id)
        if status is None:
            raise HTTPException(status_code=404, detail="unknown run")
        view = store.get_view(run_id)
        if view is None:
            # run exists but not finished (or errored without view)
            raise HTTPException(status_code=409, detail=f"run not ready: {status}")
        return JSONResponse(view)

    @app.get("/runs/{run_id}/trajectory/{trajectory_id}")
    async def get_trajectory(run_id: str, trajectory_id: str):
        if store.status(run_id) is None:
            raise HTTPException(status_code=404, detail="unknown run")
        traj = store.get_trajectory(run_id, trajectory_id)
        if traj is None:
            raise HTTPException(status_code=404, detail="unknown trajectory")
        return JSONResponse(traj)

    # Static frontend (mounted last so API routes take precedence)
    if _WEB_DIR.exists():
        app.mount("/", StaticFiles(directory=str(_WEB_DIR), html=True), name="web")

    return app
