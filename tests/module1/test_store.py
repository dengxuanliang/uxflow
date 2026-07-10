from module1.index import MemoryIndex
from module1.models import Slice, Step, TrajectorySignature
from module1.store import RecallHit, SliceStore


def _sig(traj_id="t1"):
    return TrajectorySignature(
        trajectory_id=traj_id,
        slice_index=0,
        step_range=(0, 5),
        step_count=6,
        turn_count=1,
        languages=["python"],
        tools_used=["Bash"],
        has_error_pattern=False,
        has_success_pattern=True,
        has_verification_step=False,
        bm25_tokens=["python"],
    )


def test_recall_hit_carries_signature_and_score():
    hit = RecallHit(signature=_sig(), rrf_score=0.42)
    assert hit.signature.trajectory_id == "t1"
    assert hit.rrf_score == 0.42


def test_slicestore_is_runtime_checkable_protocol():
    for name in ("add_batch", "recall", "get_slice", "set_slice_source", "update_labels"):
        assert hasattr(SliceStore, name)


def test_memory_index_get_slice_returns_slice_source():
    idx = MemoryIndex()
    sig = _sig()
    sl = Slice(
        trajectory_id="t1",
        slice_index=0,
        steps=[Step(index=0, role="assistant", content="ok")],
        start_step=0,
        end_step=0,
    )
    idx.add(sig)
    idx.set_slice_source("t1", 0, sl)
    assert idx.get_slice("t1", 0) is sl
