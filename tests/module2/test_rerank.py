from module1.models import JudgeResult, TrajectorySignature
from module1.store import RecallHit
from module2.rerank import rerank


def _hit(traj_id, idx, score, emb=None):
    sig = TrajectorySignature(
        trajectory_id=traj_id,
        slice_index=idx,
        step_range=(0, 5),
        step_count=6,
        turn_count=1,
        languages=["python"],
        tools_used=["Bash"],
        has_error_pattern=False,
        has_success_pattern=True,
        has_verification_step=False,
        bm25_tokens=["python"],
        embedding=emb or [0.1] * 1024,
    )
    return RecallHit(signature=sig, rrf_score=score)


_SUB = {
    "id": "p1",
    "target_capability": ["valid_syntax_in_toolcall"],
}


def test_matched_keeps_full_score():
    hits = [_hit("t1", 0, 0.5)]
    judged = [
        JudgeResult(
            match=True,
            confidence=0.9,
            spans=[{"start_step": 0, "end_step": 2}],
        )
    ]
    out = rerank(hits, judged, _SUB, trajectory_path="/x.jsonl")
    assert len(out) == 1
    assert out[0].relevance_score == 0.5
    assert out[0].judge_match is True
    assert out[0].judge_confidence == 0.9
    assert out[0].sub_problem_id == "p1"
    assert out[0].bm25_tokens == ["python"]


def test_unmatched_decayed_not_dropped():
    hits = [_hit("t1", 0, 0.5)]
    judged = [JudgeResult(match=False, confidence=0.1, spans=[])]
    out = rerank(hits, judged, _SUB, trajectory_path="/x.jsonl")
    assert len(out) == 1
    assert out[0].judge_match is False
    assert abs(out[0].relevance_score - 0.15) < 1e-9


def test_sorted_by_relevance_desc():
    hits = [_hit("t1", 0, 0.2), _hit("t2", 1, 0.9)]
    judged = [
        JudgeResult(match=True, confidence=0.8, spans=[]),
        JudgeResult(match=False, confidence=0.1, spans=[]),
    ]
    out = rerank(hits, judged, _SUB, trajectory_path="/x.jsonl")
    assert out[0].trajectory_id == "t2"


def test_rerank_module_imports_in_fresh_process():
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import module2.rerank; print('ok')",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_module2_package_exports_rerank_without_import_cycle():
    import module2

    assert module2.rerank is rerank


def test_rerank_rejects_mismatched_lengths():
    hits = [_hit("t1", 0, 0.5)]
    try:
        rerank(hits, [], _SUB, trajectory_path="/x.jsonl")
    except ValueError as exc:
        assert "same length" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_rerank_carries_evidence_step_and_criteria_hit():
    """PR-3 传播链：rerank 把 JudgeResult 的 evidence_step/criteria_hit 搬进 ScoredCandidate。"""
    hits = [_hit("t1", 0, 0.5)]
    judged = [
        JudgeResult(
            match=True,
            confidence=0.9,
            spans=[{"start_step": 2, "end_step": 2}],
            evidence_step=2,
            criteria_hit=["writes_test_first"],
        )
    ]
    out = rerank(hits, judged, _SUB, trajectory_path="/x.jsonl")
    assert out[0].evidence_step == 2
    assert out[0].criteria_hit == ["writes_test_first"]


def test_rerank_default_evidence_fields_when_absent():
    """老 JudgeResult（默认 evidence_step=None/criteria_hit=[]）→ ScoredCandidate 同默认。"""
    hits = [_hit("t1", 0, 0.5)]
    judged = [JudgeResult(match=False, confidence=0.1, spans=[])]
    out = rerank(hits, judged, _SUB, trajectory_path="/x.jsonl")
    assert out[0].evidence_step is None
    assert out[0].criteria_hit == []
