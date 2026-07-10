from module3.selection import SelectionConfig, select_set


def _emb(i):
    v = [0.0] * 1024
    v[i % 1024] = 1.0
    return v


def _mk(mk_candidate, traj, idx, score, sub, embi):
    return mk_candidate(traj, idx, score, _emb(embi), sub=sub)


def test_budget_respected(mk_candidate):
    cands = [_mk(mk_candidate, f"t{i}", 0, 0.5 + i * 0.01, "p1", i) for i in range(20)]
    out = select_set(cands, sub_problem_ids=["p1"], config=SelectionConfig(n=5))
    assert len(out) == 5


def test_coverage_spreads_across_subproblems(mk_candidate):
    p1 = [_mk(mk_candidate, f"a{i}", 0, 0.9, "p1", i) for i in range(10)]
    p2 = [_mk(mk_candidate, f"b{i}", 0, 0.3, "p2", 100 + i) for i in range(3)]
    out = select_set(
        p1 + p2,
        sub_problem_ids=["p1", "p2"],
        config=SelectionConfig(n=6, min_per_problem=2),
    )
    subs = [c.sub_problem_id for c in out]
    assert subs.count("p2") >= 2


def test_cap_prevents_single_problem_monopoly(mk_candidate):
    p1 = [_mk(mk_candidate, f"a{i}", 0, 0.9, "p1", i) for i in range(20)]
    out = select_set(
        p1,
        sub_problem_ids=["p1"],
        config=SelectionConfig(n=10, cap_per_problem=4),
    )
    assert sum(1 for c in out if c.sub_problem_id == "p1") <= 4


def test_fewer_candidates_than_budget(mk_candidate):
    cands = [_mk(mk_candidate, "t1", 0, 0.5, "p1", 0)]
    out = select_set(cands, sub_problem_ids=["p1"], config=SelectionConfig(n=5))
    assert len(out) == 1


def test_capped_coverage_makes_saturated_problem_zero_marginal(mk_candidate):
    p1_a = _mk(mk_candidate, "a1", 0, 0.9, "p1", 0)
    p1_b = _mk(mk_candidate, "a2", 0, 0.8, "p1", 0)
    p2 = _mk(mk_candidate, "b1", 0, 0.1, "p2", 1)
    out = select_set(
        [p1_a, p1_b, p2],
        sub_problem_ids=["p1", "p2"],
        config=SelectionConfig(n=2, coverage_cap_per_problem=0.9, lam=0.01),
    )
    assert [c.trajectory_id for c in out] == ["a1", "b1"]
