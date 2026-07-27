
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


async def test_backfill_judge_length_mismatch_recorded(populated_index):
    class ShortJudge:
        async def judge_batch(self, *, slices, target_capability, trajectory_signal, rubric=None):
            from module1.models import JudgeResult
            # returns FEWER results than slices submitted
            return [JudgeResult(match=True, confidence=0.9, spans=[], reasoning="")]

    result = await run_backfill(_new_label(), populated_index, ShortJudge(), top_k=10)
    assert any("results for" in e for e in result.errors)
    # Behavior contract under length mismatch:
    #   - candidates_screened reflects the full recall (3 slices)
    #   - the one match=True returned by the judge IS written back (not lost)
    #   - trailing candidates with no judge result are skipped (not written)
    assert result.candidates_screened == 3
    assert result.judged_true == 1
    assert result.slices_written == 1
    a = populated_index._get_signature("A", 0)
    assert "handle_async_race" in (a.capability_labels or [])
    b = populated_index._get_signature("B", 0)
    assert "handle_async_race" not in (b.capability_labels or [])


async def test_backfill_judge_returns_extra_results_does_not_crash(populated_index):
    """Judge returning MORE results than slices must not crash; extra results
    are silently dropped (candidates is the authoritative length)."""
    class LongJudge:
        async def judge_batch(self, *, slices, target_capability, trajectory_signal, rubric=None):
            from module1.models import JudgeResult
            # returns 5 results for 3 slices — extras must be dropped
            return [JudgeResult(match=True, confidence=0.9, spans=[], reasoning="")
                    for _ in range(5)]

    result = await run_backfill(_new_label(), populated_index, LongJudge(), top_k=10)
    # No length-mismatch error is recorded because candidates (3) <= results (5);
    # the iteration is bounded by candidates. All 3 candidates get match=True.
    assert result.judged_true == 3
    assert result.slices_written == 3
