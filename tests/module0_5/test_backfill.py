
from module0.taxonomy import TaxonomyLabel
from module0_5.backfill import run_backfill
from tests.module0_5.conftest import FakeJudge


def _new_label():
    return TaxonomyLabel(
        label="handle_async_race", parent="error_recovery", new_root=False,
        description="正确处理异步竞态", keywords=["async", "race"],
        description_embedding=[1.0, 0.0], taxonomy_extension=True,
        created_at="2026-07-10T00:00:00Z",
    )


async def test_backfill_writes_labels_to_judged_true_slices(populated_index):
    judge = FakeJudge(matches=[True, True, False])
    result = await run_backfill(_new_label(), populated_index, judge, top_k=10)
    assert result.label == "handle_async_race"
    assert result.slices_written >= 1
    a = populated_index._get_signature("A", 0)
    assert "handle_async_race" in (a.capability_labels or [])
    c = populated_index._get_signature("C", 0)
    assert "handle_async_race" not in (c.capability_labels or [])


async def test_backfill_is_idempotent(populated_index):
    judge = FakeJudge(matches=[True, True, False])
    await run_backfill(_new_label(), populated_index, judge, top_k=10)
    a1 = list(populated_index._get_signature("A", 0).capability_labels)
    await run_backfill(_new_label(), populated_index, judge, top_k=10)
    a2 = populated_index._get_signature("A", 0).capability_labels
    assert a2.count("handle_async_race") == 1
    assert a1 == a2


async def test_backfill_result_counts(populated_index):
    judge = FakeJudge(matches=[True, True, False])
    result = await run_backfill(_new_label(), populated_index, judge, top_k=10)
    assert result.candidates_screened >= 2
    assert result.judged_true == 2


async def test_backfill_judge_failure_returns_errors_not_crash(populated_index):
    class ExplodingJudge:
        async def judge_batch(self, **kwargs):
            raise RuntimeError("llm down")

    result = await run_backfill(_new_label(), populated_index, ExplodingJudge(), top_k=10)
    assert result.slices_written == 0
    assert any("judge" in e for e in result.errors)
