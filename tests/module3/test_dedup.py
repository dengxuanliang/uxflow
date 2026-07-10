from module3.dedup import deduplicate


def _emb(v):
    return [v] + [0.0] * 1023


def test_embedding_near_duplicates_collapsed(mk_candidate):
    a = mk_candidate("t1", 0, 0.9, _emb(1.0))
    b = mk_candidate("t2", 0, 0.5, _emb(0.999))
    out = deduplicate([a, b], cosine_threshold=0.95)
    assert len(out) == 1
    assert out[0].trajectory_id == "t1"


def test_distinct_embeddings_all_kept(mk_candidate):
    a = mk_candidate("t1", 0, 0.9, _emb(1.0))
    b = mk_candidate("t2", 0, 0.5, [0.0, 1.0] + [0.0] * 1022)
    out = deduplicate([a, b], cosine_threshold=0.95)
    assert len(out) == 2


def test_empty_input():
    assert deduplicate([], cosine_threshold=0.95) == []


def test_minhash_verbatim_duplicates_collapsed(mk_candidate):
    a = mk_candidate(
        "t1",
        0,
        0.9,
        [1.0] + [0.0] * 1023,
        tokens=["import", "os", "sys", "re"],
    )
    b = mk_candidate(
        "t2",
        0,
        0.4,
        [0.0, 1.0] + [0.0] * 1022,
        tokens=["import", "os", "sys", "re"],
    )
    out = deduplicate([a, b], cosine_threshold=0.95, minhash_threshold=0.9)
    assert len(out) == 1
    assert out[0].trajectory_id == "t1"


def test_minhash_disjoint_tokens_kept(mk_candidate):
    a = mk_candidate(
        "t1", 0, 0.9, [1.0] + [0.0] * 1023, tokens=["import", "os"]
    )
    b = mk_candidate(
        "t2", 0, 0.4, [0.0, 1.0] + [0.0] * 1022, tokens=["print", "input"]
    )
    out = deduplicate([a, b], cosine_threshold=0.95, minhash_threshold=0.9)
    assert len(out) == 2


def test_same_slice_kept_when_it_covers_different_subproblems(mk_candidate):
    a = mk_candidate("t1", 0, 0.9, [1.0] + [0.0] * 1023, sub="p1")
    b = mk_candidate("t1", 0, 0.8, [1.0] + [0.0] * 1023, sub="p2")
    out = deduplicate([a, b], cosine_threshold=0.95, minhash_threshold=0.9)
    assert {c.sub_problem_id for c in out} == {"p1", "p2"}
