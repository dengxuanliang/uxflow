"""Module 0.5 standing orchestration: proposals → ingest → enqueue → worker drain.

evolve_once is the testable assembly (pure wiring over injected stores/queue/judge).
The CLI (scripts/uxflow_evolve.py) builds the real gateway/embedding and calls this.
"""
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 dengxuanliang

from __future__ import annotations

from typing import Callable

from module0_5.evolution import ingest_proposal
from module0_5.models import LabelProposal
from module0_5.queue import SqliteBackfillQueue
from module0_5.worker import run_worker

__all__ = ["evolve_once"]


async def evolve_once(
    proposals: list[LabelProposal],
    slice_store,
    taxonomy_store,
    queue: SqliteBackfillQueue,
    judge,
    *,
    created_at: str,
    now_fn: Callable[[], str],
    dedup_threshold: float = 0.85,
    mount_threshold: float = 0.60,
) -> dict:
    """② ingest each proposal, enqueue non-duplicates (priority=associated sub-problem count), ④ drain worker."""
    # priority = number of DISTINCT passed sub-problems referencing this label (spec §6.3).
    # LabelProposal carries source_sub_problem_id for exactly this; robust to a sub-problem
    # proposing the same label more than once.
    subs_by_label: dict[str, set[str]] = {}
    for p in proposals:
        subs_by_label.setdefault(p.label, set()).add(p.source_sub_problem_id)

    ingested = 0
    enqueued = 0
    seen_enqueued: set[str] = set()
    for prop in proposals:
        res = ingest_proposal(prop, taxonomy_store, created_at=created_at,
                              dedup_threshold=dedup_threshold, mount_threshold=mount_threshold)
        if res.kind == "duplicate":
            continue  # decision: duplicates are not backfilled
        ingested += 1
        if prop.label not in seen_enqueued:  # one job per label even if multiple proposals
            queue.enqueue(prop.label, priority=len(subs_by_label[prop.label]),
                          created_at=created_at)
            seen_enqueued.add(prop.label)
            enqueued += 1

    jobs = await run_worker(queue, slice_store, taxonomy_store, judge,
                            now_fn=now_fn, drain=True)
    return {"ingested": ingested, "enqueued": enqueued, "jobs_processed": jobs}
