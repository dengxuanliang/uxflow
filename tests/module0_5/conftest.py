import pytest

from module1.index import MemoryIndex
from module1.models import Slice, TrajectorySignature, JudgeResult


class FakeJudge:
    """judge_batch returns scripted match booleans by slice order."""
    def __init__(self, matches):
        self._matches = matches

    async def judge_batch(self, *, slices, target_capability, trajectory_signal):
        out = []
        for i in range(len(slices)):
            m = self._matches[i] if i < len(self._matches) else False
            out.append(JudgeResult(match=m, confidence=0.9 if m else 0.0, spans=[], reasoning=""))
        return out


def _make_sig(tid, idx, tokens, emb, labels=None):
    return TrajectorySignature(
        trajectory_id=tid, slice_index=idx, step_range=(0, 1), step_count=2,
        turn_count=3, languages=["python"], tools_used=["Edit"],
        has_error_pattern=True, has_success_pattern=True, has_verification_step=True,
        bm25_tokens=tokens, embedding=emb, capability_labels=labels,
    )


def _make_slice(tid, idx):
    return Slice(trajectory_id=tid, slice_index=idx, steps=[], start_step=0, end_step=1)


@pytest.fixture
def populated_index():
    """MemoryIndex with 3 slices; A/B match 'async race', C is unrelated."""
    idx = MemoryIndex()
    for sig, sl in [
        (_make_sig("A", 0, ["async", "race"], [1.0, 0.0]), _make_slice("A", 0)),
        (_make_sig("B", 0, ["async", "race"], [0.9, 0.1]), _make_slice("B", 0)),
        (_make_sig("C", 0, ["unrelated"], [0.0, 1.0]), _make_slice("C", 0)),
    ]:
        idx.add(sig)
        idx.set_slice_source(sig.trajectory_id, sig.slice_index, sl)
    return idx
