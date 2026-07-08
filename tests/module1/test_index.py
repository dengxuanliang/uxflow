import numpy as np
from module1.models import TrajectorySignature
from module1.index import MemoryIndex


def _make_sig(traj_id, slice_idx=0, languages=None, tools_used=None,
              bm25_tokens=None, embedding=None, turn_count=1,
              has_verification_step=False):
    return TrajectorySignature(
        trajectory_id=traj_id,
        slice_index=slice_idx,
        step_range=(0, 5),
        step_count=6,
        turn_count=turn_count,
        languages=languages or ["python"],
        tools_used=tools_used or ["Bash"],
        has_error_pattern=False,
        has_success_pattern=True,
        has_verification_step=has_verification_step,
        bm25_tokens=bm25_tokens or ["syntaxerror", "python"],
        embedding=embedding or [0.0] * 1536,
    )


def test_add_and_size():
    idx = MemoryIndex()
    sig = _make_sig("t1")
    idx.add(sig)
    assert idx.size == 1


def test_add_batch():
    idx = MemoryIndex()
    sigs = [_make_sig(f"t{i}") for i in range(10)]
    idx.add_batch(sigs)
    assert idx.size == 10


def test_filter_languages():
    idx = MemoryIndex()
    idx.add(_make_sig("t1", languages=["python"]))
    idx.add(_make_sig("t2", languages=["java"]))
    idx.add(_make_sig("t3", languages=["python", "bash"]))

    results = idx.recall(
        structured_filters={"languages": ["python"]},
        keywords=[],
        query_embeddings=[],
        top_n=10,
    )
    traj_ids = [r.trajectory_id for r in results]
    assert "t1" in traj_ids
    assert "t3" in traj_ids
    assert "t2" not in traj_ids


def test_filter_tools_used():
    idx = MemoryIndex()
    idx.add(_make_sig("t1", tools_used=["Write", "Edit"]))
    idx.add(_make_sig("t2", tools_used=["Read", "Grep"]))

    results = idx.recall(
        structured_filters={"tools_used": ["Write"]},
        keywords=[],
        query_embeddings=[],
        top_n=10,
    )
    traj_ids = [r.trajectory_id for r in results]
    assert "t1" in traj_ids
    assert "t2" not in traj_ids


def test_filter_min_turns():
    idx = MemoryIndex()
    idx.add(_make_sig("t1", turn_count=3))
    idx.add(_make_sig("t2", turn_count=1))

    results = idx.recall(
        structured_filters={"min_turns": 2},
        keywords=[],
        query_embeddings=[],
        top_n=10,
    )
    traj_ids = [r.trajectory_id for r in results]
    assert "t1" in traj_ids
    assert "t2" not in traj_ids


def test_filter_has_verification_step():
    idx = MemoryIndex()
    idx.add(_make_sig("t1", has_verification_step=True))
    idx.add(_make_sig("t2", has_verification_step=False))

    results = idx.recall(
        structured_filters={"has_verification_step": True},
        keywords=[],
        query_embeddings=[],
        top_n=10,
    )
    traj_ids = [r.trajectory_id for r in results]
    assert "t1" in traj_ids
    assert "t2" not in traj_ids


def test_bm25_recall():
    idx = MemoryIndex()
    idx.add(_make_sig("t1", bm25_tokens=["syntaxerror", "python", "import"]))
    idx.add(_make_sig("t2", bm25_tokens=["timeout", "network", "retry"]))
    idx.add(_make_sig("t3", bm25_tokens=["syntaxerror", "java"]))

    results = idx.recall(
        structured_filters={},
        keywords=["syntaxerror", "python"],
        query_embeddings=[],
        top_n=10,
    )
    # t1 should rank highest (both keywords match)
    assert results[0].trajectory_id == "t1"


def test_vector_recall():
    idx = MemoryIndex()
    # t1 embedding close to query
    emb_query = np.random.randn(1536).astype(np.float32)
    emb_query = emb_query / np.linalg.norm(emb_query)
    # t1 = very similar to query
    emb_t1 = emb_query + np.random.randn(1536) * 0.01
    emb_t1 = emb_t1 / np.linalg.norm(emb_t1)
    # t2 = random direction
    emb_t2 = np.random.randn(1536).astype(np.float32)
    emb_t2 = emb_t2 / np.linalg.norm(emb_t2)

    idx.add(_make_sig("t1", embedding=emb_t1.tolist()))
    idx.add(_make_sig("t2", embedding=emb_t2.tolist()))

    results = idx.recall(
        structured_filters={},
        keywords=[],
        query_embeddings=[emb_query.tolist()],
        top_n=2,
    )
    assert results[0].trajectory_id == "t1"


def test_rrf_fusion():
    """BM25 and vector scores fuse via RRF to produce final ranking."""
    idx = MemoryIndex()
    # t1: good BM25, mediocre vector
    emb_query = [1.0] + [0.0] * 1535
    idx.add(_make_sig("t1", bm25_tokens=["syntaxerror", "python", "import"],
                      embedding=[0.5] + [0.0] * 1535))
    # t2: mediocre BM25, good vector
    idx.add(_make_sig("t2", bm25_tokens=["timeout"],
                      embedding=[0.99] + [0.0] * 1535))

    results = idx.recall(
        structured_filters={},
        keywords=["syntaxerror", "python"],
        query_embeddings=[emb_query],
        top_n=2,
    )
    # Both should appear (RRF merges both signals)
    traj_ids = [r.trajectory_id for r in results]
    assert "t1" in traj_ids
    assert "t2" in traj_ids


def test_empty_index_returns_empty():
    idx = MemoryIndex()
    results = idx.recall(
        structured_filters={},
        keywords=["python"],
        query_embeddings=[],
        top_n=10,
    )
    assert results == []


def test_top_n_limits_output():
    idx = MemoryIndex()
    for i in range(20):
        idx.add(_make_sig(f"t{i}", bm25_tokens=["python"]))

    results = idx.recall(
        structured_filters={},
        keywords=["python"],
        query_embeddings=[],
        top_n=5,
    )
    assert len(results) <= 5


def test_null_filters_skip_filtering():
    """None values in structured_filters are ignored (no filtering on that field)."""
    idx = MemoryIndex()
    idx.add(_make_sig("t1", languages=["python"], turn_count=1))
    idx.add(_make_sig("t2", languages=["java"], turn_count=5))

    results = idx.recall(
        structured_filters={"languages": None, "min_turns": None},
        keywords=[],
        query_embeddings=[],
        top_n=10,
    )
    assert len(results) == 2
