"""Module 0.5 end-to-end closed loop: ② ingest → ④ backfill → ③ exact rerank."""

from dataclasses import dataclass

from module0.taxonomy import Taxonomy, TaxonomyLabel, MemoryTaxonomyStore
from module0_5 import LabelProposal, ingest_proposal, rerank_with_inheritance, run_backfill
from tests.module0_5.conftest import FakeJudge


def _tax_with_family():
    def lbl(name, parent, emb):
        return TaxonomyLabel(label=name, parent=parent, new_root=(parent is None),
                             description=f"d{name}", keywords=[name], description_embedding=emb,
                             taxonomy_extension=False, created_at="")
    return Taxonomy(version="0.1.0", updated_at="", labels=[
        lbl("error_recovery", None, [1.0, 0.0]),
        lbl("effective_error_fix", "error_recovery", [0.0, 1.0]),
    ])


@dataclass
class _Sig:
    trajectory_id: str
    capability_labels: list | None


@dataclass
class _Hit:
    signature: _Sig
    rrf_score: float


async def test_0_5_closed_loop(populated_index):
    # ② ingest a new leaf under error_recovery
    store = MemoryTaxonomyStore(_tax_with_family())
    prop = LabelProposal(label="handle_async_race", description="异步竞态",
                         parent="error_recovery", description_embedding=[0.8, 0.6],
                         keywords=["async", "race"], source_sub_problem_id="p1")
    res = ingest_proposal(prop, store, created_at="2026-07-10T00:00:00Z")
    assert res.kind == "new_leaf"

    # ④ backfill: judge marks A,B true → they get the label
    new_label = store.snapshot().get("handle_async_race")
    judge = FakeJudge(matches=[True, True, False])
    bf = await run_backfill(new_label, populated_index, judge, top_k=10)
    assert bf.slices_written >= 1

    # ③ after backfill, a recall hit carrying the label reranks to exact ×1.0
    hits = [_Hit(_Sig("A", ["handle_async_race"]), 1.0)]
    out = rerank_with_inheritance(hits, target_label="handle_async_race", taxonomy=store.snapshot())
    assert out[0].rrf_score == 1.0
