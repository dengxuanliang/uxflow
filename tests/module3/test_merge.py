from module3.merge import MergedCandidate, merge_by_slice


def _sc(traj, idx, sub, score, spans, emb=None):
    class C:
        pass
    c = C()
    c.trajectory_id = traj
    c.slice_index = idx
    c.trajectory_path = "/x.jsonl"
    c.sub_problem_id = sub
    c.capability = ["cap_" + sub]
    c.relevance_score = score
    c.judge_confidence = 0.8
    c.judge_match = True
    c.loss_mask_spans = spans
    c.embedding = emb or ([1.0] + [0.0] * 1023)
    c.bm25_tokens = ["tok"]
    return c


def test_single_candidate_becomes_single_merged():
    out = merge_by_slice([_sc("t1", 0, "p1", 0.9, [{"start_step": 0, "end_step": 1}])])
    assert len(out) == 1
    m = out[0]
    assert isinstance(m, MergedCandidate)
    assert m.trajectory_id == "t1"
    assert m.slice_index == 0
    assert m.sub_problem_ids == ["p1"]
    assert m.relevance_by_problem == {"p1": 0.9}
    assert m.relevance_score == 0.9


def test_same_slice_two_subproblems_merges_to_one():
    a = _sc("t1", 0, "p1", 0.9, [{"start_step": 0, "end_step": 1}])
    b = _sc("t1", 0, "p2", 0.7, [{"start_step": 2, "end_step": 3}])
    out = merge_by_slice([a, b])
    assert len(out) == 1
    m = out[0]
    assert m.sub_problem_ids == ["p1", "p2"]
    assert m.relevance_by_problem == {"p1": 0.9, "p2": 0.7}
    assert m.relevance_score == 0.9
    assert m.loss_mask_spans == [
        {"start_step": 0, "end_step": 1},
        {"start_step": 2, "end_step": 3},
    ]


def test_span_union_dedups_identical():
    a = _sc("t1", 0, "p1", 0.9, [{"start_step": 0, "end_step": 1}])
    b = _sc("t1", 0, "p2", 0.7, [{"start_step": 0, "end_step": 1}])
    out = merge_by_slice([a, b])
    assert out[0].loss_mask_spans == [{"start_step": 0, "end_step": 1}]


def test_distinct_slices_stay_separate():
    a = _sc("t1", 0, "p1", 0.9, [{"start_step": 0, "end_step": 1}])
    b = _sc("t2", 0, "p1", 0.7, [{"start_step": 0, "end_step": 1}])
    out = merge_by_slice([a, b])
    assert len(out) == 2
