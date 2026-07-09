from module2.models import ScoredCandidate


def test_scored_candidate_fields():
    sc = ScoredCandidate(
        trajectory_id="t1",
        slice_index=0,
        trajectory_path="/x.jsonl",
        sub_problem_id="p1",
        capability=["valid_syntax_in_toolcall"],
        relevance_score=0.9,
        judge_confidence=0.88,
        loss_mask_spans=[{"start_step": 0, "end_step": 3}],
        judge_match=True,
        embedding=[0.1] * 1024,
        bm25_tokens=["python", "syntaxerror"],
    )
    assert sc.relevance_score == 0.9
    assert sc.sub_problem_id == "p1"
    assert sc.judge_match is True
    assert sc.loss_mask_spans[0]["end_step"] == 3
    assert sc.bm25_tokens == ["python", "syntaxerror"]


def test_importing_module2_models_does_not_trigger_rerank_cycle():
    import module2.models

    assert module2.models.ScoredCandidate is ScoredCandidate
