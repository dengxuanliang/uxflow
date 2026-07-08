import pytest
from module1.models import Step, Slice, JudgeResult
from module1.judge import Judge, _build_judge_prompt, _parse_judge_response


class FakeGateway:
    """Mock gateway returning scripted responses."""
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    async def call(self, messages, model, **kwargs):
        self.calls.append((messages, model))
        resp = self._responses.pop(0)
        return resp, {"status_code": 200, "prompt_tokens": 100, "completion_tokens": 50}


def _make_slice(n_steps=5):
    steps = []
    for i in range(n_steps):
        if i % 2 == 0:
            steps.append(Step(index=i, role="assistant", content=f"doing step {i}",
                             tool_call_name="Bash", tool_call_args=f"cmd_{i}"))
        else:
            steps.append(Step(index=i, role="tool", content=f"result_{i}",
                             tool_result=f"result_{i}"))
    return Slice(trajectory_id="t1", slice_index=0, steps=steps,
                 start_step=0, end_step=n_steps-1)


def test_build_judge_prompt():
    slices = [_make_slice(4)]
    prompt = _build_judge_prompt(
        slices=slices,
        target_capability=["valid_syntax_in_toolcall"],
        trajectory_signal="observation 含 SyntaxError",
    )
    assert "valid_syntax_in_toolcall" in prompt
    assert "SyntaxError" in prompt
    assert "Step 0" in prompt


def test_parse_judge_response_single():
    raw = '''[{"match": true, "confidence": 0.88, "spans": [{"start_step": 0, "end_step": 3}], "reasoning": "good"}]'''
    results = _parse_judge_response(raw, n_expected=1)
    assert len(results) == 1
    assert results[0].match is True
    assert results[0].confidence == 0.88
    assert results[0].spans == [{"start_step": 0, "end_step": 3}]


def test_parse_judge_response_batch():
    raw = '''[
        {"match": true, "confidence": 0.9, "spans": [{"start_step": 0, "end_step": 3}], "reasoning": "a"},
        {"match": false, "confidence": 0.3, "spans": [], "reasoning": "b"},
        {"match": true, "confidence": 0.85, "spans": [{"start_step": 2, "end_step": 5}], "reasoning": "c"}
    ]'''
    results = _parse_judge_response(raw, n_expected=3)
    assert len(results) == 3
    assert results[0].match is True
    assert results[1].match is False
    assert results[2].match is True


def test_parse_judge_response_malformed():
    """Malformed response returns no-match defaults."""
    raw = "this is not valid json at all"
    results = _parse_judge_response(raw, n_expected=2)
    assert len(results) == 2
    assert all(not r.match for r in results)
    assert all(r.confidence == 0.0 for r in results)


async def test_judge_single_batch():
    """Judge processes a batch of slices in one LLM call."""
    response = '''[
        {"match": true, "confidence": 0.88, "spans": [{"start_step": 0, "end_step": 3}], "reasoning": "correct demo"},
        {"match": false, "confidence": 0.2, "spans": [], "reasoning": "not relevant"}
    ]'''
    gw = FakeGateway([response])
    judge = Judge(gateway=gw, model="test-model")

    slices = [_make_slice(4), _make_slice(6)]
    results = await judge.judge_batch(
        slices=slices,
        target_capability=["valid_syntax_in_toolcall"],
        trajectory_signal="observation 含 SyntaxError",
    )
    assert len(results) == 2
    assert results[0].match is True
    assert results[1].match is False
    assert len(gw.calls) == 1  # single batch call


async def test_judge_batching_multiple_calls():
    """When >3 slices, judge splits into multiple LLM calls (batch_size=3)."""
    resp1 = '''[
        {"match": true, "confidence": 0.9, "spans": [{"start_step": 0, "end_step": 3}], "reasoning": "a"},
        {"match": true, "confidence": 0.85, "spans": [{"start_step": 0, "end_step": 3}], "reasoning": "b"},
        {"match": false, "confidence": 0.1, "spans": [], "reasoning": "c"}
    ]'''
    resp2 = '''[
        {"match": true, "confidence": 0.8, "spans": [{"start_step": 0, "end_step": 3}], "reasoning": "d"}
    ]'''
    gw = FakeGateway([resp1, resp2])
    judge = Judge(gateway=gw, model="test-model", batch_size=3)

    slices = [_make_slice(4) for _ in range(4)]
    results = await judge.judge_batch(
        slices=slices,
        target_capability=["x"],
        trajectory_signal="s",
    )
    assert len(results) == 4
    assert len(gw.calls) == 2  # 3+1 split


async def test_judge_gateway_returns_none():
    """If gateway returns None (failure), judge returns no-match for that batch."""
    gw = FakeGateway([None])
    judge = Judge(gateway=gw, model="test-model")

    slices = [_make_slice(4)]
    results = await judge.judge_batch(
        slices=slices,
        target_capability=["x"],
        trajectory_signal="s",
    )
    assert len(results) == 1
    assert results[0].match is False
    assert results[0].confidence == 0.0
