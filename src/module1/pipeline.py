"""Top-level pipeline orchestration.

Flow: load trajectories → slice → extract signatures → build index →
for each ProblemSpec sub_problem: recall → judge → collect SFTCandidates.
"""
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 dengxuanliang

from __future__ import annotations

import asyncio
import pathlib
from dataclasses import dataclass
from typing import Callable

from module1.models import SFTCandidate
from module1.loader import load_trajectories
from module1.slicer import slice_trajectory
from module1.signature import extract_signature
from module1.index import MemoryIndex
from module1.judge import Judge
from module2.models import ScoredCandidate
from module2.rerank import rerank

__all__ = ["TrajectoryPipeline", "PipelineConfig"]


@dataclass
class PipelineConfig:
    """Configuration for the trajectory pipeline."""
    judge_model: str = "gpt-4o-mini"
    judge_batch_size: int = 3
    recall_top_n: int = 20
    min_confidence: float = 0.7
    embedding_model: object | None = None  # Optional EmbeddingModel instance


class TrajectoryPipeline:
    """Full trajectory processing pipeline.

    Usage::

        pipeline = TrajectoryPipeline(config=cfg, gateway=gw)
        candidates = await pipeline.run(trajectory_paths=..., problem_specs=...)
    """

    def __init__(self, config: PipelineConfig, gateway,
                 store_factory: Callable[[], object] | None = None):
        self._config = config
        self._gateway = gateway
        self._store_factory = store_factory or MemoryIndex
        self._store = self._store_factory()
        self._judge = Judge(
            gateway=gateway,
            model=config.judge_model,
            batch_size=config.judge_batch_size,
        )
        # Mapping: trajectory_id → source file path
        self._traj_paths: dict[str, str] = {}

    def _reset_store(self):
        self._store = self._store_factory()

    async def run(
        self,
        *,
        trajectory_paths: list[str | pathlib.Path],
        problem_specs: list[dict],
    ) -> list[SFTCandidate]:
        """Execute the legacy Module 1 pipeline.

        This preserves the original SFTCandidate contract and hard-filters
        judge misses. Use run_scored() for the Module 1→2 soft-scoring entry
        consumed by Module 3 selection.

        Args:
            trajectory_paths: paths to JSONL trajectory files.
            problem_specs: list of ProblemSpec dicts (contract §1 format).

        Returns:
            List of SFTCandidate objects.
        """
        # Reset per-run state so reusing a pipeline instance doesn't accumulate.
        self._reset_store()
        self._traj_paths.clear()

        # Phase 1: Load + Slice + Sign + Index
        self._build_index(trajectory_paths)

        if self._store.size == 0:
            return []

        # Phase 2: For each sub_problem in each spec: recall + judge
        all_candidates: dict[str, SFTCandidate] = {}  # keyed by trajectory_id

        for spec in problem_specs:
            spec_id = spec.get("raw_input", "unknown")[:50]
            sub_problems = spec.get("sub_problems", [])

            for sp in sub_problems:
                candidates = await self._process_sub_problem(sp, spec_id)
                for c in candidates:
                    # Merge into existing candidate or create new
                    if c.trajectory_id in all_candidates:
                        all_candidates[c.trajectory_id].matched_problems.extend(
                            c.matched_problems
                        )
                    else:
                        all_candidates[c.trajectory_id] = c

        return list(all_candidates.values())

    async def run_scored(
        self,
        *,
        trajectory_paths: list[str | pathlib.Path],
        problem_specs: list[dict],
        on_progress: Callable[[int, int], None] | None = None,
    ) -> list[ScoredCandidate]:
        """Module 1→2 entry: recall, judge, and soft-score slice candidates.

        Unlike run(), this keeps judge misses as decayed ScoredCandidates so
        Module 2 ranking can preserve recall evidence before Module 3 selects
        the final trainable dataset.

        on_progress, if given, is called ``(done, total)`` after each sub_problem
        is scored — lets a caller surface per-sub_problem progress during the
        judge-heavy phase. Default None keeps behavior unchanged.
        """
        self._reset_store()
        self._traj_paths.clear()
        # Offload the synchronous, CPU-bound index build (slicing + embedding)
        # to a thread so it doesn't block the event loop — keeps SSE progress
        # flushing and makes cancellation responsive at the thread boundary (C2).
        # Shield it: a Python thread can't be interrupted, so if the run is
        # cancelled mid-build we must let the thread finish before unwinding —
        # otherwise the caller's Semaphore (C1) releases while this orphan thread
        # still mutates self._store/_traj_paths, and the next run corrupts its
        # index (reopens the C1 race). Thread-safety rests on that lock: only one
        # run touches this instance's _store/_traj_paths at a time.
        build = asyncio.ensure_future(asyncio.to_thread(self._build_index, trajectory_paths))
        try:
            await asyncio.shield(build)
        except asyncio.CancelledError:
            # The worker thread can't be interrupted; keep waiting (shielded, so
            # repeated cancels can't abandon it) until it truly finishes, then
            # propagate. Otherwise the lock releases while the orphan thread still
            # mutates self._store/_traj_paths → next run corrupts its index.
            while not build.done():
                try:
                    await asyncio.shield(build)
                except asyncio.CancelledError:
                    continue
            raise
        if self._store.size == 0:
            return []

        all_scored: list[ScoredCandidate] = []
        total = sum(len(spec.get("sub_problems", [])) for spec in problem_specs)
        done = 0
        for spec in problem_specs:
            for sp in spec.get("sub_problems", []):
                all_scored.extend(await self._score_sub_problem(sp))
                done += 1
                if on_progress is not None:
                    try:
                        on_progress(done, total)
                    except Exception:  # noqa: BLE001 — progress reporting must never abort scoring
                        pass

        all_scored.sort(key=lambda c: c.relevance_score, reverse=True)
        return all_scored

    def _build_index(self, trajectory_paths: list[str | pathlib.Path]) -> None:
        """Load trajectories, slice, extract signatures, build index."""
        for path in trajectory_paths:
            path = pathlib.Path(path)
            trajectories = load_trajectories(path)

            for traj in trajectories:
                self._traj_paths[traj.id] = str(path)
                slices = slice_trajectory(traj)

                for sl in slices:
                    sig = extract_signature(
                        sl, embedding_model=self._config.embedding_model
                    )
                    self._store.add(sig)
                    self._store.set_slice_source(sl.trajectory_id, sl.slice_index, sl)

    async def _process_sub_problem(
        self, sub_problem: dict, spec_id: str
    ) -> list[SFTCandidate]:
        """Recall + judge for a single sub_problem."""
        structured_filters = sub_problem.get("structured_filters", {})
        keywords = sub_problem.get("keywords", [])
        hyde_positive = sub_problem.get("hyde_positive", [])
        target_capability = sub_problem.get("target_capability", [])
        trajectory_signal = sub_problem.get("trajectory_signal", "")
        sub_problem_id = sub_problem.get("id", "unknown")

        # Compute query embeddings from hyde_positive (if embedding model available)
        query_embeddings = []
        if self._config.embedding_model and hyde_positive:
            query_embeddings = self._config.embedding_model.embed_batch(hyde_positive)

        # Recall
        recalled_hits = self._store.recall(
            structured_filters=structured_filters,
            keywords=keywords,
            query_embeddings=query_embeddings,
            top_n=self._config.recall_top_n,
        )

        if not recalled_hits:
            return []

        # Map signatures back to slices
        recalled_slices = []
        for hit in recalled_hits:
            sl = self._store.get_slice(
                hit.signature.trajectory_id, hit.signature.slice_index
            )
            if sl is not None:
                recalled_slices.append(sl)

        if not recalled_slices:
            return []

        # Judge
        judge_results = await self._judge.judge_batch(
            slices=recalled_slices,
            target_capability=target_capability,
            trajectory_signal=trajectory_signal,
        )

        # Collect matches into SFTCandidates
        candidates = []
        for sl, jr in zip(recalled_slices, judge_results):
            if jr.match and jr.confidence >= self._config.min_confidence:
                candidates.append(SFTCandidate(
                    trajectory_id=sl.trajectory_id,
                    trajectory_path=self._traj_paths.get(sl.trajectory_id, ""),
                    matched_problems=[{
                        "problem_spec_id": spec_id,
                        "sub_problem_id": sub_problem_id,
                        "capability": target_capability,
                        "confidence": jr.confidence,
                        "loss_mask_spans": jr.spans,
                    }],
                ))

        return candidates

    async def _score_sub_problem(self, sub_problem: dict) -> list[ScoredCandidate]:
        structured_filters = sub_problem.get("structured_filters", {})
        keywords = sub_problem.get("keywords", [])
        hyde_positive = sub_problem.get("hyde_positive", [])
        target_capability = sub_problem.get("target_capability", [])
        trajectory_signal = sub_problem.get("trajectory_signal", "")

        query_embeddings = []
        if self._config.embedding_model and hyde_positive:
            query_embeddings = self._config.embedding_model.embed_batch(hyde_positive)

        hits = self._store.recall(
            structured_filters=structured_filters,
            keywords=keywords,
            query_embeddings=query_embeddings,
            top_n=self._config.recall_top_n,
        )
        if not hits:
            return []

        kept_hits = []
        slices = []
        for hit in hits:
            sl = self._store.get_slice(
                hit.signature.trajectory_id, hit.signature.slice_index
            )
            if sl is not None:
                kept_hits.append(hit)
                slices.append(sl)
        if not slices:
            return []

        judge_results = await self._judge.judge_batch(
            slices=slices,
            target_capability=target_capability,
            trajectory_signal=trajectory_signal,
        )

        for hit, jr in zip(kept_hits, judge_results):
            if jr.match:
                self._store.update_labels(
                    hit.signature.trajectory_id,
                    hit.signature.slice_index,
                    target_capability,
                )

        scored = rerank(kept_hits, judge_results, sub_problem, trajectory_path="")
        for sc in scored:
            sc.trajectory_path = self._traj_paths.get(sc.trajectory_id, "")
        return scored
