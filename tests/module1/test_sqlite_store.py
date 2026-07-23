# SPDX-License-Identifier: Apache-2.0
import pytest

from module1.sqlite_store import SqliteSliceStore
from module1.models import Slice, Step, TrajectorySignature


def _sig(tid, idx, tokens, emb, labels=None):
    return TrajectorySignature(
        trajectory_id=tid, slice_index=idx, step_range=(0, 1), step_count=2,
        turn_count=3, languages=["python"], tools_used=["Edit"],
        has_error_pattern=True, has_success_pattern=True, has_verification_step=True,
        bm25_tokens=tokens, embedding=emb, capability_labels=labels,
    )


def _slice(tid, idx):
    return Slice(trajectory_id=tid, slice_index=idx,
                 steps=[Step(index=0, role="user", content="hi")], start_step=0, end_step=1)


def test_add_and_recall(tmp_path):
    store = SqliteSliceStore(tmp_path / "t.db")
    store.add_batch([_sig("A", 0, ["async", "race"], [1.0, 0.0]),
                     _sig("C", 0, ["unrelated"], [0.0, 1.0])])
    hits = store.recall(structured_filters={}, keywords=["async"],
                        query_embeddings=[[1.0, 0.0]], top_n=10)
    assert hits[0].signature.trajectory_id == "A"


def test_update_labels_merge_semantics(tmp_path):
    store = SqliteSliceStore(tmp_path / "t.db")
    store.add_batch([_sig("A", 0, ["x"], [1.0, 0.0], labels=["l1"])])
    store.update_labels("A", 0, ["l2", "l1"])  # merge, dedup
    hits = store.recall(structured_filters={}, keywords=["x"],
                        query_embeddings=[], top_n=1)
    assert hits[0].signature.capability_labels == ["l1", "l2"]


def test_persistence_across_reopen(tmp_path):
    db = tmp_path / "t.db"
    s1 = SqliteSliceStore(db)
    s1.add_batch([_sig("A", 0, ["x"], [1.0, 0.0])])
    s1.update_labels("A", 0, ["done"])
    s1.set_slice_source("A", 0, _slice("A", 0))
    s2 = SqliteSliceStore(db)  # reopen: same file
    hits = s2.recall(structured_filters={}, keywords=["x"], query_embeddings=[], top_n=1)
    assert hits[0].signature.capability_labels == ["done"]
    assert s2.get_slice("A", 0).steps[0].content == "hi"


def test_slice_roundtrip(tmp_path):
    store = SqliteSliceStore(tmp_path / "t.db")
    store.set_slice_source("A", 0, _slice("A", 0))
    got = store.get_slice("A", 0)
    assert got.trajectory_id == "A" and got.steps[0].role == "user"


def test_dim_mismatch_fails_loud(tmp_path):
    store = SqliteSliceStore(tmp_path / "t.db")
    store.add_batch([_sig("A", 0, ["x"], [1.0, 0.0])])
    with pytest.raises(ValueError, match="dimension"):
        store.add_batch([_sig("B", 0, ["y"], [1.0, 0.0, 0.0])])


def test_re_add_same_key_no_mirror_duplicate(tmp_path):
    """Second add_batch of the same key must not duplicate in the in-memory mirror."""
    store = SqliteSliceStore(tmp_path / "t.db")
    store.add_batch([_sig("A", 0, ["x"], [1.0, 0.0], labels=["v1"])])
    store.add_batch([_sig("A", 0, ["x"], [1.0, 0.0], labels=["v2"])])  # same (A,0)
    assert store.size == 1
    hits = store.recall(structured_filters={}, keywords=["x"], query_embeddings=[], top_n=10)
    assert len(hits) == 1
    assert hits[0].signature.capability_labels == ["v2"]  # last write wins (REPLACE)


def test_satisfies_slicestore_protocol(tmp_path):
    from module1.store import SliceStore
    store = SqliteSliceStore(tmp_path / "t.db")
    assert isinstance(store, SliceStore)  # runtime_checkable Protocol


def test_embedding_blob_survives_reopen_for_recall(tmp_path):
    db = tmp_path / "t.db"
    s1 = SqliteSliceStore(db)
    s1.add_batch([_sig("A", 0, ["x"], [1.0, 0.0]),
                  _sig("B", 0, ["y"], [0.0, 1.0])])
    s2 = SqliteSliceStore(db)  # reopen: embeddings loaded from BLOB
    hits = s2.recall(structured_filters={}, keywords=[],
                     query_embeddings=[[1.0, 0.0]], top_n=1)
    assert hits[0].signature.trajectory_id == "A"


def test_store_usable_from_a_different_thread(tmp_path):
    """Regression: under ASGI the store is built at startup in one thread but the
    pipeline touches it from a worker thread. Without check_same_thread=False the
    /runs path died with 'SQLite objects created in a thread can only be used in
    that same thread'. Build here, then add+recall from another thread."""
    import threading

    store = SqliteSliceStore(tmp_path / "t.db")  # built in this (main) thread
    result = {}

    def worker():
        # These calls run in a DIFFERENT thread than the one that opened the conn.
        store.add_batch([_sig("A", 0, ["async"], [1.0, 0.0])])
        hits = store.recall(structured_filters={}, keywords=["async"],
                            query_embeddings=[[1.0, 0.0]], top_n=1)
        result["tid"] = hits[0].signature.trajectory_id if hits else None

    t = threading.Thread(target=worker)
    t.start()
    t.join()
    assert result["tid"] == "A"  # no cross-thread SQLite error, recall works
