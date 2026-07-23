from module1.models import Step, Slice
from module1.judge import (
    Judge,
    _build_judge_prompt,
    _parse_judge_response,
    _JUDGE_SYSTEM_PROMPT,
    _JUDGE_SYSTEM_PROMPT_RUBRIC,
)


_RUBRIC = {
    "positive_criteria": ["工具调用前先检查已读文件", "不重复读同一文件"],
    "negative_criteria": ["连续多次 Read 同一路径"],
    "decisive_evidence": "本可重复读却转而复用已有内容的决策步",
    "capability_kind": "avoidance",
}


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


def test_parse_judge_response_confidence_string():
    """Non-numeric confidence string like 'high' must not raise; defaults to no-match."""
    raw = '[{"match": true, "confidence": "high", "spans": [], "reasoning": "ok"}]'
    results = _parse_judge_response(raw, n_expected=1)
    assert len(results) == 1
    # coercion error → fallback no-match
    assert results[0].match is False
    assert results[0].confidence == 0.0
    assert results[0].reasoning == "field_coercion_error"


def test_parse_judge_response_match_string_false():
    """String 'false' must be parsed as False, not truthy bool(str)."""
    raw = '[{"match": "false", "confidence": 0.9, "spans": [], "reasoning": "nope"}]'
    results = _parse_judge_response(raw, n_expected=1)
    assert len(results) == 1
    assert results[0].match is False
    assert results[0].confidence == 0.9


def test_parse_judge_response_match_string_true():
    """String 'true' must be parsed as True."""
    raw = '[{"match": "true", "confidence": 0.75, "spans": [], "reasoning": "yes"}]'
    results = _parse_judge_response(raw, n_expected=1)
    assert len(results) == 1
    assert results[0].match is True
    assert results[0].confidence == 0.75


# ── PR-3: rubric 扩参 + evidence_step/criteria_hit 解析 ──

def test_build_judge_prompt_none_rubric_legacy_format():
    """rubric=None → 走旧格式，prompt 不含 rubric 对照段。"""
    prompt = _build_judge_prompt(
        slices=[_make_slice(4)],
        target_capability=["avoid_redundant_repetition"],
        trajectory_signal="signal",
        rubric=None,
    )
    assert "判据卡" not in prompt
    assert "positive_criteria" not in prompt
    assert "avoid_redundant_repetition" in prompt


def test_build_judge_prompt_with_rubric_injects_criteria():
    """rubric=dict → prompt 含 positive/negative_criteria + kind + decisive_evidence。"""
    prompt = _build_judge_prompt(
        slices=[_make_slice(4)],
        target_capability=["avoid_redundant_repetition"],
        trajectory_signal="signal",
        rubric=_RUBRIC,
    )
    assert "判据卡" in prompt
    assert "不重复读同一文件" in prompt          # positive_criteria
    assert "连续多次 Read 同一路径" in prompt      # negative_criteria
    assert "avoidance" in prompt                   # capability_kind
    assert "本可重复读却转而复用已有内容的决策步" in prompt  # decisive_evidence


def test_parse_judge_response_extracts_evidence_step_and_criteria_hit():
    raw = ('[{"match": true, "confidence": 0.9, "evidence_step": 3, '
           '"criteria_hit": ["不重复读同一文件"], '
           '"spans": [{"start_step": 3, "end_step": 3}], "reasoning": "ok"}]')
    results = _parse_judge_response(raw, n_expected=1)
    assert results[0].evidence_step == 3
    assert results[0].criteria_hit == ["不重复读同一文件"]


def test_parse_judge_response_missing_new_fields_degrade():
    """旧格式响应（无 evidence_step/criteria_hit）→ 降级默认 None/[]，不崩。"""
    raw = '[{"match": true, "confidence": 0.8, "spans": [], "reasoning": "x"}]'
    results = _parse_judge_response(raw, n_expected=1)
    assert results[0].evidence_step is None
    assert results[0].criteria_hit == []


def test_parse_judge_response_malformed_new_fields_degrade():
    """evidence_step 非 int / criteria_hit 非 list → 降级默认，不崩。"""
    raw = ('[{"match": true, "confidence": 0.8, "evidence_step": "third", '
           '"criteria_hit": "not a list", "spans": [], "reasoning": "x"}]')
    results = _parse_judge_response(raw, n_expected=1)
    assert results[0].evidence_step is None
    assert results[0].criteria_hit == []


def test_parse_judge_response_evidence_step_bool_rejected():
    """bool 是 int 子类，evidence_step=true 应被拒 → None（不误当 1）。"""
    raw = ('[{"match": true, "confidence": 0.8, "evidence_step": true, '
           '"criteria_hit": [1, "keep", 2], "spans": [], "reasoning": "x"}]')
    results = _parse_judge_response(raw, n_expected=1)
    assert results[0].evidence_step is None
    assert results[0].criteria_hit == ["keep"]  # 非 str 项被过滤


async def test_judge_batch_none_rubric_uses_legacy_system_prompt():
    resp = '[{"match": false, "confidence": 0.2, "spans": [], "reasoning": "x"}]'
    gw = FakeGateway([resp])
    judge = Judge(gateway=gw, model="test-model")
    await judge.judge_batch(
        slices=[_make_slice(4)],
        target_capability=["x"],
        trajectory_signal="s",
        rubric=None,
    )
    system_content = gw.calls[0][0][0]["content"]
    assert system_content == _JUDGE_SYSTEM_PROMPT


async def test_judge_batch_with_rubric_uses_rubric_system_prompt():
    resp = ('[{"match": true, "confidence": 0.9, "evidence_step": 2, '
            '"criteria_hit": ["不重复读同一文件"], '
            '"spans": [{"start_step": 2, "end_step": 2}], "reasoning": "ok"}]')
    gw = FakeGateway([resp])
    judge = Judge(gateway=gw, model="test-model")
    results = await judge.judge_batch(
        slices=[_make_slice(4)],
        target_capability=["avoid_redundant_repetition"],
        trajectory_signal="s",
        rubric=_RUBRIC,
    )
    system_content = gw.calls[0][0][0]["content"]
    user_content = gw.calls[0][0][1]["content"]
    assert system_content == _JUDGE_SYSTEM_PROMPT_RUBRIC
    assert "不重复读同一文件" in user_content
    assert results[0].evidence_step == 2
    assert results[0].criteria_hit == ["不重复读同一文件"]


async def test_judge_batch_default_rubric_backward_compatible():
    """不传 rubric（现有调用方）仍走旧路径、不破坏行为。"""
    resp = '[{"match": true, "confidence": 0.7, "spans": [], "reasoning": "x"}]'
    gw = FakeGateway([resp])
    judge = Judge(gateway=gw, model="test-model")
    results = await judge.judge_batch(
        slices=[_make_slice(4)],
        target_capability=["x"],
        trajectory_signal="s",
    )
    assert gw.calls[0][0][0]["content"] == _JUDGE_SYSTEM_PROMPT
    assert results[0].match is True
    assert results[0].evidence_step is None
    assert results[0].criteria_hit == []
