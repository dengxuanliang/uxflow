from module0_5 import LabelProposal, BackfillResult


def test_label_proposal_fields():
    p = LabelProposal(
        label="handle_async_race",
        description="正确处理异步竞态",
        parent="error_recovery",
        description_embedding=[0.1, 0.2],
        keywords=["async", "race"],
        source_sub_problem_id="p1",
    )
    assert p.label == "handle_async_race"
    assert p.parent == "error_recovery"
    assert p.description_embedding == [0.1, 0.2]
    assert p.source_sub_problem_id == "p1"


def test_label_proposal_parent_optional():
    p = LabelProposal(
        label="x", description="d", parent=None,
        description_embedding=[], keywords=[], source_sub_problem_id="p2",
    )
    assert p.parent is None


def test_backfill_result_fields():
    r = BackfillResult(
        label="handle_async_race", candidates_screened=200,
        judged_true=12, slices_written=12, errors=[],
    )
    assert r.label == "handle_async_race"
    assert r.candidates_screened == 200
    assert r.judged_true == 12
    assert r.slices_written == 12
    assert r.errors == []
