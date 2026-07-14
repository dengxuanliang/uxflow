"""Backfill worker loop: claim job → run_backfill → complete/fail. run_backfill unchanged."""
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 dengxuanliang

from __future__ import annotations

import asyncio
from dataclasses import asdict
from typing import Callable

from module0_5.backfill import run_backfill
from module0_5.queue import SqliteBackfillQueue

__all__ = ["run_worker"]


async def run_worker(
    queue: SqliteBackfillQueue,
    slice_store,
    taxonomy_store,
    judge,
    *,
    now_fn: Callable[[], str],
    drain: bool = False,
    max_jobs: int | None = None,
    poll_interval: float = 1.0,
) -> int:
    """Consume backfill jobs. Returns number of jobs processed.

    drain=True: stop when queue is empty (one pass). Otherwise poll forever
    (until max_jobs reached, if set). now_fn injects timestamps (repo convention).
    Startup requeues any stale 'running' jobs (crash recovery, single-worker assumption).
    """
    queue.reset_stale(now=now_fn())
    processed = 0
    while True:
        job = queue.claim(now=now_fn())
        if job is None:
            if drain:
                return processed
            await asyncio.sleep(poll_interval)
            continue

        # snapshot() is outside the try/except by design: if it raises, the job
        # stays 'running' and reset_stale requeues it on next startup (single-worker V1).
        new_label = taxonomy_store.snapshot().get(job.label)
        if new_label is None:
            queue.fail(job, error=f"label not found in taxonomy: {job.label}", now=now_fn())
            processed += 1
        else:
            try:
                result = await run_backfill(new_label, slice_store, judge)
                queue.complete(job, result=asdict(result), now=now_fn())
            except Exception as e:  # run_backfill returns, doesn't raise; belt-and-suspenders
                queue.fail(job, error=f"worker: {e}", now=now_fn())
            processed += 1

        if max_jobs is not None and processed >= max_jobs:
            return processed
