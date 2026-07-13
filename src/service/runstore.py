"""Run state storage seam (spec §8).

V1 is in-memory. The RunStore Protocol is the seam: swap in Redis/DB later
without touching app.py. Mirrors module1's SliceStore seam pattern.
"""
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 dengxuanliang

from __future__ import annotations

import asyncio
import uuid
from typing import AsyncIterator, Iterable, Protocol

__all__ = ["RunStore", "MemoryRunStore"]


class RunStore(Protocol):
    def create(self) -> str: ...
    def append_event(self, run_id: str, event: dict) -> None: ...
    def events_snapshot(self, run_id: str) -> Iterable[dict]: ...
    def subscribe(self, run_id: str) -> AsyncIterator[dict]: ...
    def set_view(self, run_id: str, view: dict, trajectories: dict) -> None: ...
    def get_view(self, run_id: str) -> dict | None: ...
    def get_trajectory(self, run_id: str, traj_id: str) -> dict | None: ...
    def status(self, run_id: str) -> str | None: ...
    def mark_done(self, run_id: str) -> None: ...
    def mark_error(self, run_id: str) -> None: ...


class _Run:
    def __init__(self) -> None:
        self.status: str = "running"
        self.events: list[dict] = []
        self.view: dict | None = None
        self.trajectories: dict = {}
        self.condition = asyncio.Condition()


class MemoryRunStore:
    """In-memory RunStore. Not persistent; lost on restart (spec §10)."""

    def __init__(self, max_runs: int = 50) -> None:
        self._runs: dict[str, _Run] = {}
        self._max_runs = max_runs
        self._notify_tasks: set = set()

    def create(self) -> str:
        self._evict_if_needed()
        run_id = uuid.uuid4().hex[:16]
        self._runs[run_id] = _Run()
        return run_id

    def _evict_if_needed(self) -> None:
        # Bound memory: when at capacity, drop the oldest TERMINAL runs (dict is
        # insertion-ordered). Never evict a still-running run.
        if len(self._runs) < self._max_runs:
            return
        for rid, run in list(self._runs.items()):
            if run.status in ("done", "error"):
                del self._runs[rid]
                if len(self._runs) < self._max_runs:
                    return

    def _get(self, run_id: str) -> _Run | None:
        return self._runs.get(run_id)

    def append_event(self, run_id: str, event: dict) -> None:
        run = self._get(run_id)
        if run is None:
            return
        run.events.append(event)
        # Wake any SSE subscribers waiting on new events.
        async def _notify():
            async with run.condition:
                run.condition.notify_all()
        try:
            loop = asyncio.get_running_loop()
            t = loop.create_task(_notify())
            self._notify_tasks.add(t)
            t.add_done_callback(self._notify_tasks.discard)
        except RuntimeError:
            # No running loop (sync context): subscribers, if any, still see the
            # event within the subscribe() 0.5s poll. NOTE: a call from a worker
            # thread (e.g. future run_in_executor producers) also lands here and
            # silently skips notify — for off-loop producers, capture the loop at
            # construction and use loop.call_soon_threadsafe instead.
            pass

    def events_snapshot(self, run_id: str) -> list[dict]:
        run = self._get(run_id)
        return list(run.events) if run else []

    async def subscribe(self, run_id: str) -> AsyncIterator[dict]:
        run = self._get(run_id)
        if run is None:
            return
        idx = 0
        while True:
            # Drain any events already buffered.
            while idx < len(run.events):
                yield run.events[idx]
                idx += 1
            # If run finished and we've drained everything, stop.
            if run.status in ("done", "error"):
                return
            async with run.condition:
                try:
                    await asyncio.wait_for(run.condition.wait(), timeout=0.5)
                except asyncio.TimeoutError:
                    pass  # re-check loop (also catches races)

    def set_view(self, run_id: str, view: dict, trajectories: dict) -> None:
        run = self._get(run_id)
        if run is None:
            return
        run.view = view
        run.trajectories = trajectories

    def get_view(self, run_id: str) -> dict | None:
        run = self._get(run_id)
        return run.view if run else None

    def get_trajectory(self, run_id: str, traj_id: str) -> dict | None:
        run = self._get(run_id)
        if run is None:
            return None
        return run.trajectories.get(traj_id)

    def status(self, run_id: str) -> str | None:
        run = self._get(run_id)
        return run.status if run else None

    def _mark(self, run_id: str, status: str) -> None:
        run = self._get(run_id)
        if run is None:
            return
        run.status = status
        async def _notify():
            async with run.condition:
                run.condition.notify_all()
        try:
            t = asyncio.get_running_loop().create_task(_notify())
            self._notify_tasks.add(t)
            t.add_done_callback(self._notify_tasks.discard)
        except RuntimeError:
            # See append_event: same off-loop / worker-thread caveat applies.
            pass

    def mark_done(self, run_id: str) -> None:
        self._mark(run_id, "done")

    def mark_error(self, run_id: str) -> None:
        self._mark(run_id, "error")
