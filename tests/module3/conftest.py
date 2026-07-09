import pytest


@pytest.fixture
def mk_candidate():
    def _mk(traj_id, idx, score, emb, sub="p1", tokens=None):
        class Candidate:
            pass

        candidate = Candidate()
        candidate.trajectory_id = traj_id
        candidate.slice_index = idx
        candidate.trajectory_path = "/x.jsonl"
        candidate.sub_problem_id = sub
        candidate.capability = ["valid_syntax_in_toolcall"]
        candidate.relevance_score = score
        candidate.judge_confidence = 0.8
        candidate.judge_match = True
        candidate.loss_mask_spans = [{"start_step": 0, "end_step": 1}]
        candidate.embedding = emb
        candidate.bm25_tokens = tokens or []
        if tokens is not None:
            candidate._tokens = tokens
        return candidate

    return _mk
