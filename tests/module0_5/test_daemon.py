# SPDX-License-Identifier: Apache-2.0
from module0.taxonomy import Taxonomy, TaxonomyLabel
from module0.sqlite_taxonomy import SqliteTaxonomyStore
from module0_5.queue import SqliteBackfillQueue
from module0_5.daemon import evolve_once
from module0_5.models import LabelProposal
from module1.sqlite_store import SqliteSliceStore
from module1.models import Slice, Step, TrajectorySignature, JudgeResult


class FakeJudge:
    def __init__(self, matches): self._m = matches
    async def judge_batch(self, *, slices, target_capability, trajectory_signal):
        return [JudgeResult(match=self._m[i] if i < len(self._m) else False,
                            confidence=0.9, spans=[], reasoning="") for i in range(len(slices))]


def _tax():
    return Taxonomy(version="0.1.0", updated_at="t", labels=[
        TaxonomyLabel(label="error_recovery", parent=None, new_root=False, description="错误恢复",
                      keywords=["error"], description_embedding=[1.0, 0.0],
                      taxonomy_extension=False, created_at="t")])


def _proposal():
    # embedding [0.0, 1.0] is orthogonal to the seed root [1.0, 0.0]:
    # cosine 0 < mount_threshold → resolves to new_root (NOT duplicate), so it ingests.
    return LabelProposal(label="fix_runtime_exception", description="修复运行时异常",
                         parent="error_recovery", description_embedding=[0.0, 1.0],
                         keywords=["TypeError"], source_sub_problem_id="p1")


async def test_evolve_once_full_chain(tmp_path):
    db = tmp_path / "t.db"
    ss = SqliteSliceStore(db)
    ss.add_batch([TrajectorySignature(
        trajectory_id="A", slice_index=0, step_range=(0, 1), step_count=2, turn_count=3,
        languages=["python"], tools_used=["Edit"], has_error_pattern=True,
        has_success_pattern=True, has_verification_step=True, bm25_tokens=["TypeError"],
        embedding=[0.0, 1.0], capability_labels=None)])
    ss.set_slice_source("A", 0, Slice(trajectory_id="A", slice_index=0,
                        steps=[Step(index=0, role="user", content="TypeError")], start_step=0, end_step=1))
    ts = SqliteTaxonomyStore(db, seed=_tax())
    q = SqliteBackfillQueue(db)

    summary = await evolve_once([_proposal()], ss, ts, q, FakeJudge([True]),
                                created_at="2026-07-13T00:00:00Z", now_fn=lambda: "t1")

    # ② ingested (new_root, since embedding is orthogonal to the seed root)
    assert ts.snapshot().get("fix_runtime_exception") is not None
    # ④ backfilled: label on the slice
    hits = ss.recall(structured_filters={}, keywords=["TypeError"], query_embeddings=[], top_n=1)
    assert "fix_runtime_exception" in (hits[0].signature.capability_labels or [])
    assert summary["ingested"] >= 1 and summary["jobs_processed"] >= 1


async def test_evolve_once_persists_across_reopen(tmp_path):
    db = tmp_path / "t.db"
    ss = SqliteSliceStore(db)
    ts = SqliteTaxonomyStore(db, seed=_tax())
    q = SqliteBackfillQueue(db)
    await evolve_once([_proposal()], ss, ts, q, FakeJudge([]),
                      created_at="2026-07-13T00:00:00Z", now_fn=lambda: "t1")
    # reopen taxonomy store from same file → new label survived
    ts2 = SqliteTaxonomyStore(db)
    assert ts2.snapshot().get("fix_runtime_exception") is not None
