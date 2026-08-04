"""FastAPI service for the Trajectory Inspector (spec §5.1)."""
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 dengxuanliang

from __future__ import annotations

import asyncio
import json
import os
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
    search_fn: Callable | None = None,
    ingest_fn: Callable | None = None,
    deps: Any = None,
    tau_q: float = 0.90,
    db_manager: Any = None,
    lifespan: Any = None,
) -> FastAPI:
    """Build the app. Inject store/run_fn/deps for testing; defaults for prod."""
    store = store or MemoryRunStore()
    if run_fn is None:
        from service.orchestrator import run_pipeline
        run_fn = run_pipeline
    if search_fn is None:
        from service.orchestrator import run_search
        search_fn = run_search
    if ingest_fn is None:
        from service.orchestrator import run_ingest
        ingest_fn = run_ingest

    app = FastAPI(title="Trajectory Inspector", lifespan=lifespan)

    # Track in-flight background tasks so a run can be cancelled (see /cancel).
    _tasks: dict[str, asyncio.Task] = {}
    # Serialize actual pipeline execution: the injected compiler/pipeline hold
    # per-run mutable state (self._store is reset each run_scored), so concurrent
    # runs would corrupt each other. A run waits its turn rather than interleave.
    _run_lock = asyncio.Semaphore(1)

    # 库管理器与 run_lock 共享同一把锁（切换/清除前判占用）。接线方案B：
    # manager 在 inspector_serve 构造，此处回注锁。
    if db_manager is not None:
        db_manager.attach_lock(_run_lock)

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

    async def _bg_search(run_id: str, question: str):
        # Mirror _background_run's lock/try-except-finally, but for the read path:
        # no temp file to clean, and run_search does NOT emit its own done event
        # (unlike run_ingest), so we append the terminal "done" ourselves.
        def emit(ev: dict) -> None:
            store.append_event(run_id, ev)
        try:
            if _run_lock.locked():
                emit({"stage": "search0", "status": "running",
                      "msg": "前一个任务运行中，排队等待…"})
            async with _run_lock:      # serialize pipeline execution (C1)
                view, trajectories = await search_fn(
                    question, deps=deps, emit=emit, run_id=run_id, tau_q=tau_q)
                store.set_view(run_id, view, trajectories)
                store.append_event(run_id, {"stage": "done", "status": "ok"})
                store.mark_done(run_id)
        except asyncio.CancelledError:
            store.append_event(run_id, {"stage": "done", "status": "cancelled",
                                        "msg": "已停止"})
            store.mark_error(run_id)
        except Exception as exc:  # noqa: BLE001 — surface as error event, don't crash server
            store.append_event(run_id, {"stage": "done", "status": "error",
                                        "msg": str(exc)})
            store.mark_error(run_id)
        finally:
            _tasks.pop(run_id, None)

    async def _bg_ingest(run_id: str, manifest_text: str | None,
                         traj_path: pathlib.Path | None):
        # Mirror _background_run, but run_ingest already emits its own terminal
        # "done" event — so the success path only set_view + mark_done (no extra
        # done, to keep exactly one done in the SSE stream).
        def emit(ev: dict) -> None:
            store.append_event(run_id, ev)
        try:
            if _run_lock.locked():
                emit({"stage": "ingest_traj", "status": "running",
                      "msg": "前一个任务运行中，排队等待…"})
            async with _run_lock:      # serialize pipeline execution (C1)
                view, trajectories = await ingest_fn(
                    manifest_lines=(manifest_text.splitlines()
                                    if manifest_text else None),
                    trajectory_path=traj_path,
                    deps=deps, emit=emit, run_id=run_id, tau_q=tau_q)
                store.set_view(run_id, view, {})
                store.mark_done(run_id)
        except asyncio.CancelledError:
            store.append_event(run_id, {"stage": "done", "status": "cancelled",
                                        "msg": "已停止"})
            store.mark_error(run_id)
        except Exception as exc:  # noqa: BLE001 — surface as error event, don't crash server
            store.append_event(run_id, {"stage": "done", "status": "error",
                                        "msg": str(exc)})
            store.mark_error(run_id)
        finally:
            if traj_path is not None:
                traj_path.unlink(missing_ok=True)
            _tasks.pop(run_id, None)

    def _finalize_simple(_t: asyncio.Task, rid: str) -> None:
        # Safety net for search (no temp file): pop task, force terminal if the
        # body never finalized (e.g. cancel delivered before the coroutine ran).
        _tasks.pop(rid, None)
        if store.status(rid) == "running":
            store.append_event(rid, {"stage": "done", "status": "cancelled",
                                     "msg": "已停止"})
            store.mark_error(rid)

    def _finalize_ingest(_t: asyncio.Task, rid: str,
                         path: pathlib.Path | None) -> None:
        # Safety net for ingest: mirror _finalize including temp cleanup.
        if path is not None:
            path.unlink(missing_ok=True)          # idempotent (missing_ok)
        _tasks.pop(rid, None)
        if store.status(rid) == "running":
            store.append_event(rid, {"stage": "done", "status": "cancelled",
                                     "msg": "已停止"})
            store.mark_error(rid)

    @app.post("/runs")
    async def create_run(manifest: UploadFile, trajectories: UploadFile):
        raw = await manifest.read()
        try:
            manifest_text = raw.decode("utf-8")
        except UnicodeDecodeError:
            raise HTTPException(
                status_code=400,
                detail="用户清单必须是 UTF-8 编码的文本文件") from None
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

    @app.post("/search")
    async def create_search(payload: dict):
        question = (payload.get("question") or "").strip()
        if not question:
            raise HTTPException(status_code=400, detail="问题不能为空")
        run_id = store.create()
        task = asyncio.create_task(_bg_search(run_id, question))
        _tasks[run_id] = task
        task.add_done_callback(lambda t, rid=run_id: _finalize_simple(t, rid))
        return {"run_id": run_id}

    @app.post("/ingest")
    async def create_ingest(manifest: UploadFile | None = None,
                            trajectories: UploadFile | None = None):
        if manifest is None and trajectories is None:
            raise HTTPException(status_code=400, detail="至少上传一个文件")
        manifest_text = None
        if manifest is not None:
            raw = await manifest.read()
            try:
                manifest_text = raw.decode("utf-8")
            except UnicodeDecodeError:
                raise HTTPException(
                    status_code=400,
                    detail="用户清单必须是 UTF-8 编码的文本文件") from None
        traj_path = None
        if trajectories is not None:
            traj_bytes = await trajectories.read()
            tmp = tempfile.NamedTemporaryFile(
                delete=False, suffix=".jsonl", mode="wb")
            tmp.write(traj_bytes)
            tmp.close()
            traj_path = pathlib.Path(tmp.name)
        run_id = store.create()
        task = asyncio.create_task(_bg_ingest(run_id, manifest_text, traj_path))
        _tasks[run_id] = task
        task.add_done_callback(
            lambda t, rid=run_id, p=traj_path: _finalize_ingest(t, rid, p))
        return {"run_id": run_id}

    @app.get("/stats")
    async def get_stats():
        # async (not sync def): sync endpoints run in a threadpool, but the SQLite
        # store connections are bound to the event-loop thread (check_same_thread).
        # Keeping this on the loop avoids "SQLite objects created in a thread can
        # only be used in that same thread" — matches every other store-touching route.
        problems = (deps.problem_store.count()
                    if getattr(deps, "problem_store", None) else 0)
        trajectories = (deps.trajectory_store.count()
                        if getattr(deps, "trajectory_store", None) else 0)
        signatures = getattr(getattr(deps, "pipeline", None), "_store", None)
        sig_count = getattr(signatures, "size", 0) if signatures is not None else 0
        # Backends ride along on /stats (already polled by the UI) so the web
        # client can show a standing fake-backend warning: the terminal banner
        # from inspector_serve.py is invisible to anyone using the browser.
        return {"problems": problems, "trajectories": trajectories,
                "signatures": sig_count,
                "llm_backend": os.environ.get("UXFLOW_LLM_BACKEND", "real").lower(),
                "embed_backend": os.environ.get("UXFLOW_EMBED_BACKEND", "fake").lower()}

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
            # 回退持久 store（search 结果的详情跨重启可读）
            ts = getattr(deps, "trajectory_store", None)
            if ts is not None:
                traj = ts.get(trajectory_id)
        if traj is None:
            raise HTTPException(status_code=404, detail="unknown trajectory")
        return JSONResponse(traj)

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

    # Static frontend (mounted last so API routes take precedence)
    if _WEB_DIR.exists():
        app.mount("/", StaticFiles(directory=str(_WEB_DIR), html=True), name="web")

    return app
