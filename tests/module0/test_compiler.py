import pytest
from module0.compiler import QueryCompiler
from module0.taxonomy import Taxonomy


class FakeGateway:
    """Mock gateway returning scripted responses in call order."""
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    async def call(self, messages, model, **kwargs):
        self.calls.append((messages, model))
        resp = self._responses.pop(0)
        return resp, {"status_code": 200, "prompt_tokens": 10, "completion_tokens": 5}


@pytest.fixture
def taxonomy(taxonomy_v0_path):
    return Taxonomy.load(taxonomy_v0_path)


async def test_two_call_happy_path(taxonomy):
    """No ambiguous drops → exactly 2 calls (Call 1 + Call 2)."""
    call1_resp = '{"sub_problems": [{"id": "p1", "raw_text": "写入py文件有语法错误", "failure_summary": "写入 py 语法错误"}]}'
    call2_resp = '''[{
        "id": "p1",
        "target_capability": ["valid_syntax_in_toolcall"],
        "trajectory_signal": "observation 含 SyntaxError",
        "hyde_positive": ["正例片段一，超过二十字符的假设轨迹", "正例片段二，超过二十字符的假设轨迹"],
        "keywords": ["SyntaxError", "python"],
        "structured_filters": {"languages": ["python"]},
        "confidence": 0.92,
        "route": "pass"
    }]'''
    gw = FakeGateway([call1_resp, call2_resp])
    compiler = QueryCompiler(gateway=gw, taxonomy=taxonomy, model="test-model", embedding_model=None)

    spec = await compiler.compile("写入py文件有语法错误")

    assert len(gw.calls) == 2
    assert spec.domain == "agentic_swe"
    assert len(spec.sub_problems) == 1
    assert spec.sub_problems[0].id == "p1"
    assert spec.sub_problems[0].origin == "original"
    assert spec.sub_problems[0].parent_id is None


async def test_drop_non_ambiguous_excluded(taxonomy):
    """route=drop, reason!=ambiguous → excluded, no Call 3."""
    call1_resp = '{"sub_problems": [{"id": "p1", "raw_text": "前端样式问题", "failure_summary": "前端视觉问题"}]}'
    call2_resp = '''[{
        "id": "p1",
        "target_capability": ["x"],
        "trajectory_signal": "s",
        "hyde_positive": ["hyp padding one text", "hyp padding two text"],
        "keywords": ["k"],
        "structured_filters": {},
        "confidence": 0.3,
        "route": "drop",
        "drop_reason": "not_applicable"
    }]'''
    gw = FakeGateway([call1_resp, call2_resp])
    compiler = QueryCompiler(gateway=gw, taxonomy=taxonomy, model="test-model", embedding_model=None)

    spec = await compiler.compile("前端样式问题")

    assert len(gw.calls) == 2
    assert len(spec.sub_problems) == 0


async def test_ambiguous_triggers_clarification(taxonomy):
    """route=drop, reason=ambiguous → Call 3 + Call 2' (4 calls total)."""
    call1_resp = '{"sub_problems": [{"id": "p12", "raw_text": "解题过程中途停止", "failure_summary": "多步任务中途停止"}]}'
    call2_resp = '''[{
        "id": "p12",
        "target_capability": ["persist_through_truncation"],
        "trajectory_signal": "s",
        "hyde_positive": ["hyp padding one text", "hyp padding two text"],
        "keywords": ["k"],
        "structured_filters": {},
        "confidence": 0.70,
        "route": "drop",
        "drop_reason": "ambiguous"
    }]'''
    call3_resp = '''[{
        "original_id": "p12",
        "clarified": [
            {"id": "p12a", "raw_text": "解题过程中途停止", "failure_summary": "被 max_turns 截断而未完成"},
            {"id": "p12b", "raw_text": "解题过程中途停止", "failure_summary": "主动收尾误判为完成"}
        ]
    }]'''
    call2prime_resp = '''[
        {"id": "p12a", "target_capability": ["persist_through_truncation"], "trajectory_signal": "末轮命中 max_turns",
         "hyde_positive": ["hyp padding one text", "hyp padding two text"], "keywords": ["max_turns"],
         "structured_filters": {"min_turns": 5}, "confidence": 0.84, "route": "pass"},
        {"id": "p12b", "target_capability": ["x"], "trajectory_signal": "s",
         "hyde_positive": ["hyp padding one text", "hyp padding two text"], "keywords": ["k"],
         "structured_filters": {}, "confidence": 0.60, "route": "pass"}
    ]'''
    gw = FakeGateway([call1_resp, call2_resp, call3_resp, call2prime_resp])
    compiler = QueryCompiler(gateway=gw, taxonomy=taxonomy, model="test-model", embedding_model=None)

    spec = await compiler.compile("解题过程中途停止")

    assert len(gw.calls) == 4
    assert len(spec.sub_problems) == 1
    assert spec.sub_problems[0].id == "p12a"
    assert spec.sub_problems[0].origin == "clarified"
    assert spec.sub_problems[0].parent_id == "p12"


async def test_no_recursion_after_clarification(taxonomy):
    """Call 2' output never triggers another Call 3."""
    call1_resp = '{"sub_problems": [{"id": "p1", "raw_text": "模糊问题", "failure_summary": "模糊"}]}'
    call2_resp = '[{"id": "p1", "target_capability": ["x"], "trajectory_signal": "s", "hyde_positive": ["padding one text here", "padding two text here"], "keywords": ["k"], "structured_filters": {}, "confidence": 0.5, "route": "drop", "drop_reason": "ambiguous"}]'
    call3_resp = '[{"original_id": "p1", "clarified": [{"id": "p1a", "raw_text": "t", "failure_summary": "clarified"}]}]'
    call2prime_resp = '[{"id": "p1a", "target_capability": ["x"], "trajectory_signal": "s", "hyde_positive": ["padding one text here", "padding two text here"], "keywords": ["k"], "structured_filters": {}, "confidence": 0.4, "route": "pass"}]'
    gw = FakeGateway([call1_resp, call2_resp, call3_resp, call2prime_resp])
    compiler = QueryCompiler(gateway=gw, taxonomy=taxonomy, model="test-model", embedding_model=None)

    spec = await compiler.compile("模糊问题")

    assert len(gw.calls) == 4  # exactly 4, no 5th call
    assert len(spec.sub_problems) == 0  # p1a dropped (0.4 < 0.8)


async def test_dropped_audit_records(taxonomy):
    """Dropped sub-problems are accessible for audit."""
    call1_resp = '{"sub_problems": [{"id": "p1", "raw_text": "环境问题", "failure_summary": "基础设施问题"}]}'
    call2_resp = '[{"id": "p1", "target_capability": ["x"], "trajectory_signal": "s", "hyde_positive": ["padding one text here", "padding two text here"], "keywords": ["k"], "structured_filters": {}, "confidence": 0.3, "route": "drop", "drop_reason": "not_applicable"}]'
    gw = FakeGateway([call1_resp, call2_resp])
    compiler = QueryCompiler(gateway=gw, taxonomy=taxonomy, model="test-model", embedding_model=None)

    spec = await compiler.compile("环境问题")
    dropped = compiler.dropped_records

    assert len(spec.sub_problems) == 0
    assert len(dropped) == 1
    assert dropped[0].drop_reason == "not_applicable"


# ── Regression tests for review bug fixes ──

async def test_c2_invalid_pass_item_degrades_not_crashes(taxonomy):
    """C2: a schema-invalid pass item (4 labels) degrades to drop, does not crash."""
    call1_resp = '{"sub_problems": [{"id": "p1", "raw_text": "问题", "failure_summary": "描述"}]}'
    # 4 target_capability labels violates the 1-3 constraint (valid JSON,
    # fails at schema validation — exactly what C2 must isolate)
    call2_resp = '''[{
        "id": "p1",
        "target_capability": ["a", "b", "c", "d"],
        "trajectory_signal": "s",
        "hyde_positive": ["padding one text here", "padding two text here"],
        "keywords": ["k"],
        "structured_filters": {},
        "confidence": 0.9,
        "route": "pass"
    }]'''
    gw = FakeGateway([call1_resp, call2_resp])
    compiler = QueryCompiler(gateway=gw, taxonomy=taxonomy, model="test-model", embedding_model=None)

    # Must not raise
    spec = await compiler.compile("问题")
    # Invalid pass degraded to drop, not in sub_problems
    assert len(spec.sub_problems) == 0
    assert len(compiler.dropped_records) == 1


async def test_c2_invalid_hyde_count_degrades(taxonomy):
    """C2: pass item with 1 hyde segment (needs 2-3) degrades, does not crash."""
    call1_resp = '{"sub_problems": [{"id": "p1", "raw_text": "问题", "failure_summary": "描述"}]}'
    call2_resp = '''[{
        "id": "p1",
        "target_capability": ["valid_syntax_in_toolcall"],
        "trajectory_signal": "s",
        "hyde_positive": ["only one segment"],
        "keywords": ["k"],
        "structured_filters": {},
        "confidence": 0.9,
        "route": "pass"
    }]'''
    gw = FakeGateway([call1_resp, call2_resp])
    compiler = QueryCompiler(gateway=gw, taxonomy=taxonomy, model="test-model", embedding_model=None)

    spec = await compiler.compile("问题")
    assert len(spec.sub_problems) == 0
    assert len(compiler.dropped_records) == 1


async def test_c2_invalid_drop_reason_falls_back_to_other(taxonomy):
    """C2: an out-of-enum drop_reason from LLM falls back to 'other', no crash."""
    call1_resp = '{"sub_problems": [{"id": "p1", "raw_text": "问题", "failure_summary": "描述"}]}'
    call2_resp = '''[{
        "id": "p1",
        "target_capability": ["x"],
        "trajectory_signal": "s",
        "hyde_positive": ["padding one text here", "padding two text here"],
        "keywords": ["k"],
        "structured_filters": {},
        "confidence": 0.3,
        "route": "drop",
        "drop_reason": "totally_made_up_reason"
    }]'''
    gw = FakeGateway([call1_resp, call2_resp])
    compiler = QueryCompiler(gateway=gw, taxonomy=taxonomy, model="test-model", embedding_model=None)

    await compiler.compile("问题")
    assert len(compiler.dropped_records) == 1
    assert compiler.dropped_records[0].drop_reason == "other"


async def test_c3_embeddings_stored(taxonomy):
    """C3: hyde_positive embeddings are stored in hyde_embeddings, not discarded."""

    class FakeEmbedding:
        dimension = 1024
        def embed_batch(self, texts):
            return [[0.1] * 1024 for _ in texts]

    call1_resp = '{"sub_problems": [{"id": "p1", "raw_text": "写入py文件有语法错误", "failure_summary": "语法错误"}]}'
    call2_resp = '''[{
        "id": "p1",
        "target_capability": ["valid_syntax_in_toolcall"],
        "trajectory_signal": "s",
        "hyde_positive": ["padding one text here", "padding two text here"],
        "keywords": ["k"],
        "structured_filters": {},
        "confidence": 0.9,
        "route": "pass"
    }]'''
    gw = FakeGateway([call1_resp, call2_resp])
    compiler = QueryCompiler(gateway=gw, taxonomy=taxonomy, model="test-model", embedding_model=FakeEmbedding())

    spec = await compiler.compile("写入py文件有语法错误")
    assert len(spec.sub_problems) == 1
    # Embeddings stored keyed by sub-problem id
    assert "p1" in compiler.hyde_embeddings
    assert len(compiler.hyde_embeddings["p1"]) == 2  # 2 hyde segments
    assert len(compiler.hyde_embeddings["p1"][0]) == 1024


async def test_i1_dropped_records_reset_per_compile(taxonomy):
    """I1: each compile() resets dropped_records (no cross-call leakage)."""
    call1_resp = '{"sub_problems": [{"id": "p1", "raw_text": "环境问题", "failure_summary": "基础设施"}]}'
    call2_resp = '[{"id": "p1", "target_capability": ["x"], "trajectory_signal": "s", "hyde_positive": ["padding one text here", "padding two text here"], "keywords": ["k"], "structured_filters": {}, "confidence": 0.3, "route": "drop", "drop_reason": "not_applicable"}]'
    # First compile
    gw1 = FakeGateway([call1_resp, call2_resp])
    compiler = QueryCompiler(gateway=gw1, taxonomy=taxonomy, model="test-model", embedding_model=None)
    await compiler.compile("环境问题")
    assert len(compiler.dropped_records) == 1

    # Second compile with a clean pass — dropped_records must reset to 0
    call2_pass = '[{"id": "p1", "target_capability": ["valid_syntax_in_toolcall"], "trajectory_signal": "s", "hyde_positive": ["padding one text here", "padding two text here"], "keywords": ["k"], "structured_filters": {}, "confidence": 0.9, "route": "pass"}]'
    gw2 = FakeGateway([call1_resp, call2_pass])
    compiler._gateway = gw2
    await compiler.compile("环境问题")
    assert len(compiler.dropped_records) == 0  # reset, not accumulated


def test_compile_error_is_importable_and_is_exception():
    from module0 import CompileError
    assert issubclass(CompileError, Exception)


_C2_OK = '''[{
    "id": "p1",
    "target_capability": ["valid_syntax_in_toolcall"],
    "trajectory_signal": "observation 含 SyntaxError",
    "hyde_positive": ["正例片段一,超过二十字符的假设轨迹", "正例片段二,超过二十字符的假设轨迹"],
    "keywords": ["SyntaxError", "python"],
    "structured_filters": {"languages": ["python"]},
    "confidence": 0.92,
    "route": "pass"
}]'''

_C1_OK = '{"sub_problems": [{"id": "p1", "raw_text": "写入py有语法错误", "failure_summary": "语法错误"}]}'


async def test_call1_empty_then_retry_succeeds(taxonomy):
    gw = FakeGateway([None, _C1_OK, _C2_OK])
    compiler = QueryCompiler(gateway=gw, taxonomy=taxonomy, model="test-model", embedding_model=None)
    spec = await compiler.compile("写入py文件有语法错误")
    assert len(spec.sub_problems) == 1
    assert compiler.robustness_report["retries"]["call1"] == 1


async def test_call1_hard_fail_raises_compile_error(taxonomy):
    from module0 import CompileError
    gw = FakeGateway([None, None])
    compiler = QueryCompiler(gateway=gw, taxonomy=taxonomy, model="test-model", embedding_model=None)
    with pytest.raises(CompileError):
        await compiler.compile("写入py文件有语法错误")


async def test_call2_empty_then_retry_succeeds(taxonomy):
    # Call 1 ok; Call 2 empty then valid.
    gw = FakeGateway([_C1_OK, None, _C2_OK])
    compiler = QueryCompiler(gateway=gw, taxonomy=taxonomy, model="test-model", embedding_model=None)
    spec = await compiler.compile("写入py文件有语法错误")
    assert len(spec.sub_problems) == 1
    assert compiler.robustness_report["retries"]["call2"] == 1


async def test_call2_exhausted_degrades_to_empty_spec(taxonomy):
    # Call 1 ok; Call 2 empty on both attempts → degrade, empty spec, no crash.
    gw = FakeGateway([_C1_OK, None, None])
    compiler = QueryCompiler(gateway=gw, taxonomy=taxonomy, model="test-model", embedding_model=None)
    spec = await compiler.compile("写入py文件有语法错误")
    assert spec.sub_problems == []
    assert spec.domain == "agentic_swe"
    assert "call2" in compiler.robustness_report["degraded"]


# Call 2 that drops p1 as ambiguous (triggers Call 3 + Call 2').
_C2_AMBIGUOUS = '''[{
    "id": "p1",
    "target_capability": ["valid_syntax_in_toolcall"],
    "trajectory_signal": "含糊",
    "hyde_positive": ["片段一超过二十字符的假设正例轨迹", "片段二超过二十字符的假设正例轨迹"],
    "keywords": ["python"],
    "structured_filters": {"languages": ["python"]},
    "confidence": 0.3,
    "route": "drop",
    "drop_reason": "ambiguous"
}]'''

# Call 3 clarifies p1 into p1a.
_C3_OK = '[{"original_id": "p1", "clarified": [{"id": "p1a", "raw_text": "澄清后的问题", "failure_summary": "澄清"}]}]'

# Call 2' valid response recovering p1a.
_C2P_OK = '''[{
    "id": "p1a",
    "target_capability": ["valid_syntax_in_toolcall"],
    "trajectory_signal": "observation 含 SyntaxError",
    "hyde_positive": ["片段一超过二十字符的假设正例轨迹", "片段二超过二十字符的假设正例轨迹"],
    "keywords": ["SyntaxError"],
    "structured_filters": {"languages": ["python"]},
    "confidence": 0.9,
    "route": "pass"
}]'''

# Call 2' response missing the required "route" field (the field the live LLM dropped).
_C2P_MISSING_ROUTE = '''[{
    "id": "p1a",
    "target_capability": ["valid_syntax_in_toolcall"],
    "trajectory_signal": "observation 含 SyntaxError",
    "hyde_positive": ["片段一超过二十字符的假设正例轨迹", "片段二超过二十字符的假设正例轨迹"],
    "keywords": ["SyntaxError"],
    "structured_filters": {"languages": ["python"]},
    "confidence": 0.9
}]'''


async def test_call2prime_missing_route_then_retry_succeeds(taxonomy):
    gw = FakeGateway([_C1_OK, _C2_AMBIGUOUS, _C3_OK, _C2P_MISSING_ROUTE, _C2P_OK])
    compiler = QueryCompiler(gateway=gw, taxonomy=taxonomy, model="test-model", embedding_model=None)
    spec = await compiler.compile("解题过程中途停止")
    assert any(sp.id == "p1a" for sp in spec.sub_problems)
    assert compiler.robustness_report["retries"]["call2prime"] == 1


async def test_call2prime_exhausted_degrades_without_crash(taxonomy):
    gw = FakeGateway([_C1_OK, _C2_AMBIGUOUS, _C3_OK, _C2P_MISSING_ROUTE, _C2P_MISSING_ROUTE])
    compiler = QueryCompiler(gateway=gw, taxonomy=taxonomy, model="test-model", embedding_model=None)
    spec = await compiler.compile("解题过程中途停止")
    assert all(sp.id != "p1a" for sp in spec.sub_problems)
    assert "clarify" in compiler.robustness_report["degraded"]


async def test_bad_json_and_empty_both_retried(taxonomy):
    # Call 2 returns non-JSON garbage first, then valid → should retry & succeed.
    gw = FakeGateway([_C1_OK, "not json {{{", _C2_OK])
    compiler = QueryCompiler(gateway=gw, taxonomy=taxonomy, model="test-model", embedding_model=None)
    spec = await compiler.compile("写入py文件有语法错误")
    assert len(spec.sub_problems) == 1
    assert compiler.robustness_report["retries"]["call2"] == 1


async def test_happy_path_records_no_retries_no_degrade(taxonomy):
    gw = FakeGateway([_C1_OK, _C2_OK])
    compiler = QueryCompiler(gateway=gw, taxonomy=taxonomy, model="test-model", embedding_model=None)
    spec = await compiler.compile("写入py文件有语法错误")
    assert len(spec.sub_problems) == 1
    report = compiler.robustness_report
    assert report["retries"] == {"call1": 0, "call2": 0, "call3": 0, "call2prime": 0}
    assert report["degraded"] == []


from uxflow_embed import FakeEmbedder

_C2_WITH_PROPOSAL = '''[{
    "id": "p1",
    "target_capability": ["new_cap"],
    "trajectory_signal": "observation 含 SyntaxError",
    "hyde_positive": ["正例片段一,超过二十字符的假设轨迹", "正例片段二,超过二十字符的假设轨迹"],
    "keywords": ["SyntaxError"],
    "structured_filters": {"languages": ["python"]},
    "confidence": 0.92,
    "route": "pass",
    "label_proposals": [{"label": "new_cap", "description": "新能力描述", "parent": "code_generation", "keywords": ["SyntaxError"], "taxonomy_extension": true}]
}]'''


async def test_compiler_collects_label_proposals(taxonomy):
    gw = FakeGateway([_C1_OK, _C2_WITH_PROPOSAL])
    compiler = QueryCompiler(gateway=gw, taxonomy=taxonomy, model="test-model",
                             embedding_model=FakeEmbedder(dimension=8))
    await compiler.compile("写入py文件有语法错误")
    assert len(compiler.label_proposals) == 1
    prop = compiler.label_proposals[0]
    assert prop.label == "new_cap"
    assert prop.parent == "code_generation"
    assert prop.source_sub_problem_id == "p1"
    assert len(prop.description_embedding) == 8


async def test_compiler_no_proposals_when_none_offered(taxonomy):
    gw = FakeGateway([_C1_OK, _C2_OK])
    compiler = QueryCompiler(gateway=gw, taxonomy=taxonomy, model="test-model",
                             embedding_model=FakeEmbedder(dimension=8))
    await compiler.compile("写入py文件有语法错误")
    assert compiler.label_proposals == []


async def test_compiler_drops_proposals_from_dropped_items(taxonomy):
    c2_dropped = _C2_WITH_PROPOSAL.replace('"route": "pass"', '"route": "drop", "drop_reason": "not_applicable"').replace('"confidence": 0.92', '"confidence": 0.2')
    gw = FakeGateway([_C1_OK, c2_dropped])
    compiler = QueryCompiler(gateway=gw, taxonomy=taxonomy, model="test-model",
                             embedding_model=FakeEmbedder(dimension=8))
    await compiler.compile("写入py文件有语法错误")
    assert compiler.label_proposals == []
