from module3.dedup import deduplicate
from module3.merge import MergedCandidate


def _m(traj, idx, subs_rel, emb, tokens=None):
    return MergedCandidate(
        trajectory_id=traj, slice_index=idx, trajectory_path="/x.jsonl",
        sub_problem_ids=list(subs_rel.keys()), relevance_by_problem=dict(subs_rel),
        relevance_score=max(subs_rel.values()),
        loss_mask_spans=[{"start_step": 0, "end_step": 1}],
        embedding=emb, bm25_tokens=tokens or [],
    )


def test_embedding_near_duplicates_collapsed_and_absorbed():
    a = _m("t1", 0, {"p1": 0.9}, [1.0] + [0.0] * 1023)
    b = _m("t2", 0, {"p1": 0.5}, [0.999] + [0.0] * 1023)
    out = deduplicate([a, b], cosine_threshold=0.95)
    assert len(out) == 1
    assert out[0].trajectory_id == "t1"   # 高分幸存


def test_distinct_embeddings_all_kept():
    a = _m("t1", 0, {"p1": 0.9}, [1.0] + [0.0] * 1023)
    b = _m("t2", 0, {"p1": 0.5}, [0.0, 1.0] + [0.0] * 1022)
    out = deduplicate([a, b], cosine_threshold=0.95)
    assert len(out) == 2


def test_near_dup_across_subproblems_absorbs_attribution():
    # 两个近重 slice 分属 p1/p2 → 塌缩为 1 条，幸存者吸收 p2 归属（保覆盖）
    a = _m("t1", 0, {"p1": 0.9}, [1.0] + [0.0] * 1023)
    b = _m("t2", 0, {"p2": 0.5}, [0.999] + [0.0] * 1023)
    out = deduplicate([a, b], cosine_threshold=0.95)
    assert len(out) == 1
    assert set(out[0].sub_problem_ids) == {"p1", "p2"}


def test_empty_input():
    assert deduplicate([], cosine_threshold=0.95) == []


def test_minhash_verbatim_duplicates_collapsed():
    a = _m("t1", 0, {"p1": 0.9}, [1.0] + [0.0] * 1023,
           tokens=["import", "os", "sys", "re"])
    b = _m("t2", 0, {"p1": 0.4}, [0.0, 1.0] + [0.0] * 1022,
           tokens=["import", "os", "sys", "re"])
    out = deduplicate([a, b], cosine_threshold=0.95, minhash_threshold=0.9)
    assert len(out) == 1
    assert out[0].trajectory_id == "t1"


def test_minhash_disjoint_tokens_kept():
    a = _m("t1", 0, {"p1": 0.9}, [1.0] + [0.0] * 1023, tokens=["import", "os"])
    b = _m("t2", 0, {"p1": 0.4}, [0.0, 1.0] + [0.0] * 1022, tokens=["print", "input"])
    out = deduplicate([a, b], cosine_threshold=0.95, minhash_threshold=0.9)
    assert len(out) == 2
