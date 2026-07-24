from module3.selection import SelectionConfig, select_set
from module3.merge import MergedCandidate


def _emb(i):
    v = [0.0] * 1024
    v[i % 1024] = 1.0
    return v


def _mk(traj, idx, score, sub, embi):
    # 单归属 MergedCandidate 便捷构造（替代原 ScoredCandidate-like mk_candidate 夹具）
    return MergedCandidate(
        trajectory_id=traj, slice_index=idx, trajectory_path="/x.jsonl",
        sub_problem_ids=[sub], relevance_by_problem={sub: score},
        relevance_score=score, loss_mask_spans=[{"start_step": 0, "end_step": 1}],
        embedding=_emb(embi), bm25_tokens=[],
    )


def test_budget_respected():
    cands = [_mk(f"t{i}", 0, 0.5 + i * 0.01, "p1", i) for i in range(20)]
    out = select_set(cands, sub_problem_ids=["p1"], config=SelectionConfig(n=5))
    assert len(out) == 5


def test_coverage_spreads_across_subproblems():
    p1 = [_mk(f"a{i}", 0, 0.9, "p1", i) for i in range(10)]
    p2 = [_mk(f"b{i}", 0, 0.3, "p2", 100 + i) for i in range(3)]
    out = select_set(
        p1 + p2,
        sub_problem_ids=["p1", "p2"],
        config=SelectionConfig(n=6, min_per_problem=2),
    )
    subs = [sid for c in out for sid in c.sub_problem_ids]
    assert subs.count("p2") >= 2


def test_cap_prevents_single_problem_monopoly():
    p1 = [_mk(f"a{i}", 0, 0.9, "p1", i) for i in range(20)]
    out = select_set(
        p1,
        sub_problem_ids=["p1"],
        config=SelectionConfig(n=10, cap_per_problem=4),
    )
    assert sum(1 for c in out if "p1" in c.sub_problem_ids) <= 4


def test_fewer_candidates_than_budget():
    cands = [_mk("t1", 0, 0.5, "p1", 0)]
    out = select_set(cands, sub_problem_ids=["p1"], config=SelectionConfig(n=5))
    assert len(out) == 1


def test_capped_coverage_makes_saturated_problem_zero_marginal():
    p1_a = _mk("a1", 0, 0.9, "p1", 0)
    p1_b = _mk("a2", 0, 0.8, "p1", 0)
    p2 = _mk("b1", 0, 0.1, "p2", 1)
    out = select_set(
        [p1_a, p1_b, p2],
        sub_problem_ids=["p1", "p2"],
        config=SelectionConfig(n=2, coverage_cap_per_problem=0.9, lam=0.01),
    )
    assert [c.trajectory_id for c in out] == ["a1", "b1"]


def _merged(traj, idx, rel_by_problem, embi):
    v = [0.0] * 1024
    v[embi % 1024] = 1.0
    return MergedCandidate(
        trajectory_id=traj, slice_index=idx, trajectory_path="/x.jsonl",
        sub_problem_ids=list(rel_by_problem.keys()),
        relevance_by_problem=dict(rel_by_problem),
        relevance_score=max(rel_by_problem.values()),
        loss_mask_spans=[{"start_step": 0, "end_step": 1}],
        embedding=v, bm25_tokens=[],
    )


def test_shared_slice_satisfies_both_min_per_problem():
    shared = _merged("t1", 0, {"p1": 0.9, "p2": 0.8}, 0)
    out = select_set([shared], sub_problem_ids=["p1", "p2"],
                     config=SelectionConfig(n=5, min_per_problem=1))
    assert len(out) == 1
    assert set(out[0].sub_problem_ids) == {"p1", "p2"}


def test_multi_attribution_not_double_appended_in_min_phase():
    shared = _merged("t1", 0, {"p1": 0.9, "p2": 0.8}, 0)
    extra = _merged("t2", 0, {"p1": 0.5}, 1)
    out = select_set([shared, extra], sub_problem_ids=["p1", "p2"],
                     config=SelectionConfig(n=5, min_per_problem=1))
    keys = [(c.trajectory_id, c.slice_index) for c in out]
    assert len(keys) == len(set(keys))


def test_cap_blocks_when_any_subproblem_saturated():
    a = _merged("t1", 0, {"p1": 0.9}, 0)
    b = _merged("t2", 0, {"p1": 0.8}, 1)
    out = select_set([a, b], sub_problem_ids=["p1"],
                     config=SelectionConfig(n=5, cap_per_problem=1))
    assert len(out) == 1
