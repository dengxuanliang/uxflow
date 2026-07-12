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

    async def _background_run(run_id: str, manifest_text: str, traj_path: pathlib.Path):
        def emit(ev: dict) -> None:
            store.append_event(run_id, ev)
        try:
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
        except Exception as exc:  # noqa: BLE001 — surface as error event, don't crash server
            store.append_event(run_id, {"stage": "done", "status": "error",
                                        "msg": str(exc)})
            store.mark_error(run_id)

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
        asyncio.create_task(
            _background_run(run_id, manifest_text, pathlib.Path(tmp.name))
        )
        return {"run_id": run_id}

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
