from module3.compose import GeneralDataConfig
from module3.pipeline import select_final_dataset
from module3.selection import SelectionConfig


def test_end_to_end_dedup_select_compose(mk_candidate):
    cands = [
        mk_candidate("t1", 0, 0.9, [1.0] + [0.0] * 1023, sub="p1"),
        mk_candidate("t2", 0, 0.4, [0.999] + [0.0] * 1023, sub="p1"),
        mk_candidate("t3", 0, 0.8, [0.0, 1.0] + [0.0] * 1022, sub="p1"),
        mk_candidate("t4", 0, 0.7, [0.0, 0.0, 1.0] + [0.0] * 1021, sub="p2"),
        mk_candidate("t5", 0, 0.6, [0.0, 0.0, 0.0, 1.0] + [0.0] * 1020, sub="p2"),
        mk_candidate("t6", 0, 0.5, [0.0] * 4 + [1.0] + [0.0] * 1019, sub="p2"),
    ]
    out = select_final_dataset(
        cands,
        sub_problem_ids=["p1", "p2"],
        selection=SelectionConfig(n=4, min_per_problem=1),
        general=GeneralDataConfig(ratio=0.3, source_path=None),
    )
    assert out["manifest"]["targeted_count"] == 4
    ids = {c.trajectory_id for c in out["targeted"]}
    assert "t2" not in ids


def test_final_dataset_excludes_unmatched_empty_span_candidates(mk_candidate):
    matched = mk_candidate("t1", 0, 0.5, [1.0] + [0.0] * 1023, sub="p1")
    matched.loss_mask_spans = [{"start_step": 0, "end_step": 1}]
    matched.judge_match = True
    unmatched = mk_candidate("t2", 0, 0.9, [0.0, 1.0] + [0.0] * 1022, sub="p1")
    unmatched.loss_mask_spans = []
    unmatched.judge_match = False
    out = select_final_dataset(
        [matched, unmatched],
        sub_problem_ids=["p1"],
        selection=SelectionConfig(n=2),
        general=GeneralDataConfig(ratio=0.3, source_path=None),
    )
    assert out["targeted"] == [matched]
