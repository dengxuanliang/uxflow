# SPDX-License-Identifier: Apache-2.0

from module0.taxonomy import Taxonomy, TaxonomyLabel
from module0.sqlite_taxonomy import SqliteTaxonomyStore
from module0_5.queue import SqliteBackfillQueue
from module0_5.worker import run_worker
from module1.sqlite_store import SqliteSliceStore
from module1.models import Slice, Step, TrajectorySignature, JudgeResult


class FakeJudge:
    def __init__(self, matches): self._m = matches
    async def judge_batch(self, *, slices, target_capability, trajectory_signal, rubric=None):
        return [JudgeResult(match=self._m[i] if i < len(self._m) else False,
                            confidence=0.9, spans=[], reasoning="")
                for i in range(len(slices))]


def _sig(tid, tokens, emb, labels=None):
    return TrajectorySignature(
        trajectory_id=tid, slice_index=0, step_range=(0, 1), step_count=2, turn_count=3,
        languages=["python"], tools_used=["Edit"], has_error_pattern=True,
        has_success_pattern=True, has_verification_step=True, bm25_tokens=tokens,
        embedding=emb, capability_labels=labels)


def _tax():
    return Taxonomy(version="0.1.0", updated_at="t", labels=[
        TaxonomyLabel(label="error_recovery", parent=None, new_root=False,
                      description="错误恢复", keywords=["error"], description_embedding=[1.0, 0.0],
                      taxonomy_extension=False, created_at="t")])


def _new_label():
    return TaxonomyLabel(label="fix_runtime_exception", parent="error_recovery", new_root=False,
                         description="修复运行时异常", keywords=["TypeError"],
                         description_embedding=[0.9, 0.1], taxonomy_extension=True, created_at="t")


async def test_worker_backfills_and_marks_done(tmp_path):
    db = tmp_path / "t.db"
    ss = SqliteSliceStore(db)
    ss.add_batch([_sig("A", ["TypeError"], [0.9, 0.1])])
    ss.set_slice_source("A", 0, Slice(trajectory_id="A", slice_index=0,
                        steps=[Step(index=0, role="user", content="TypeError")],
                        start_step=0, end_step=1))
    ts = SqliteTaxonomyStore(db, seed=_tax())
    ts.add_label(_new_label())
    q = SqliteBackfillQueue(db)
    q.enqueue("fix_runtime_exception", priority=1, created_at="t0")

    n = await run_worker(q, ss, ts, FakeJudge([True]), now_fn=lambda: "t1", drain=True)
    assert n == 1
    assert q.count_by_status("done") == 1
    # label written back to the slice
    hits = ss.recall(structured_filters={}, keywords=["TypeError"], query_embeddings=[], top_n=1)
    assert "fix_runtime_exception" in (hits[0].signature.capability_labels or [])


async def test_worker_idempotent_on_rerun(tmp_path):
    db = tmp_path / "t.db"
    ss = SqliteSliceStore(db)
    ss.add_batch([_sig("A", ["TypeError"], [0.9, 0.1])])
    ss.set_slice_source("A", 0, Slice(trajectory_id="A", slice_index=0,
                        steps=[Step(index=0, role="user", content="x")], start_step=0, end_step=1))
    ts = SqliteTaxonomyStore(db, seed=_tax())
    ts.add_label(_new_label())
    q = SqliteBackfillQueue(db)
    q.enqueue("fix_runtime_exception", priority=1, created_at="t0")
    await run_worker(q, ss, ts, FakeJudge([True]), now_fn=lambda: "t1", drain=True)
    # requeue same label, rerun: label must not duplicate
    q.enqueue("fix_runtime_exception", priority=1, created_at="t0")
    await run_worker(q, ss, ts, FakeJudge([True]), now_fn=lambda: "t2", drain=True)
    hits = ss.recall(structured_filters={}, keywords=["TypeError"], query_embeddings=[], top_n=1)
    assert (hits[0].signature.capability_labels or []).count("fix_runtime_exception") == 1


async def test_worker_missing_label_marks_failed(tmp_path):
    db = tmp_path / "t.db"
    ss = SqliteSliceStore(db)
    ts = SqliteTaxonomyStore(db, seed=_tax())
    q = SqliteBackfillQueue(db)
    q.enqueue("nonexistent_label", priority=1, created_at="t0")
    await run_worker(q, ss, ts, FakeJudge([]), now_fn=lambda: "t1", drain=True)
    assert q.count_by_status("failed") == 1
