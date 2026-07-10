import pytest
from module0.parsing import (
    parse_call1_response,
    parse_call2_response,
    parse_call3_response,
    ParseError,
)


def test_parse_call1_success():
    raw = '{"sub_problems": [{"id": "p1", "raw_text": "写入py文件有语法错误", "failure_summary": "写入 py 文件语法错误"}]}'
    result = parse_call1_response(raw)
    assert len(result) == 1
    assert result[0]["id"] == "p1"


def test_parse_call1_with_markdown_fence():
    raw = '```json\n{"sub_problems": [{"id": "p1", "raw_text": "test", "failure_summary": "test"}]}\n```'
    result = parse_call1_response(raw)
    assert len(result) == 1


def test_parse_call1_invalid_json():
    with pytest.raises(ParseError):
        parse_call1_response("not json at all")


def test_parse_call1_missing_field():
    raw = '{"sub_problems": [{"id": "p1"}]}'
    with pytest.raises(ParseError, match="raw_text|failure_summary"):
        parse_call1_response(raw)


def test_parse_call2_success():
    raw = '''[{
        "id": "p1",
        "target_capability": ["valid_syntax_in_toolcall"],
        "trajectory_signal": "observation 含 SyntaxError",
        "hyde_positive": ["hyp1 text here over 20 chars for validity", "hyp2 text here over 20 chars"],
        "keywords": ["SyntaxError", "python"],
        "structured_filters": {"languages": ["python"]},
        "confidence": 0.92,
        "route": "pass"
    }]'''
    result = parse_call2_response(raw)
    assert len(result) == 1
    assert result[0]["confidence"] == 0.92
    assert result[0]["route"] == "pass"


def test_parse_call2_with_drop():
    raw = '''[{
        "id": "p5",
        "target_capability": ["x"],
        "trajectory_signal": "s",
        "hyde_positive": ["h1 padding text", "h2 padding text"],
        "keywords": ["k"],
        "structured_filters": {},
        "confidence": 0.3,
        "route": "drop",
        "drop_reason": "not_applicable"
    }]'''
    result = parse_call2_response(raw)
    assert result[0]["route"] == "drop"
    assert result[0]["drop_reason"] == "not_applicable"


def test_parse_call3_success():
    raw = '''[{
        "original_id": "p12",
        "clarified": [
            {"id": "p12a", "raw_text": "text", "failure_summary": "被 max_turns 截断"},
            {"id": "p12b", "raw_text": "text", "failure_summary": "主动收尾误判"}
        ]
    }]'''
    result = parse_call3_response(raw)
    assert len(result) == 1
    assert result[0]["original_id"] == "p12"
    assert len(result[0]["clarified"]) == 2


def test_parse_call3_max_4_clarified():
    raw = '''[{
        "original_id": "p1",
        "clarified": [
            {"id": "p1a", "raw_text": "t", "failure_summary": "s1"},
            {"id": "p1b", "raw_text": "t", "failure_summary": "s2"},
            {"id": "p1c", "raw_text": "t", "failure_summary": "s3"},
            {"id": "p1d", "raw_text": "t", "failure_summary": "s4"},
            {"id": "p1e", "raw_text": "t", "failure_summary": "s5"}
        ]
    }]'''
    result = parse_call3_response(raw)
    assert len(result[0]["clarified"]) <= 4


# ── Regression tests for C1: braces/brackets inside JSON string values ──

def test_parse_braces_in_string_value():
    """C1 regression: braces inside string values must not truncate extraction."""
    raw = '{"sub_problems": [{"id": "p1", "raw_text": "code: if err != nil { return }", "failure_summary": "closing brace } in text"}]}'
    result = parse_call1_response(raw)
    assert len(result) == 1
    assert "}" in result[0]["raw_text"]


def test_parse_brackets_in_trajectory_signal():
    """C1 regression: brackets in trajectory_signal (common with code)."""
    raw = '''[{
        "id": "p1",
        "target_capability": ["valid_syntax_in_toolcall"],
        "trajectory_signal": "observation contains: for i in range(10) { print(i) }",
        "hyde_positive": ["hyp1 padding text here", "hyp2 padding text here"],
        "keywords": ["syntax"],
        "structured_filters": {},
        "confidence": 0.9,
        "route": "pass"
    }]'''
    result = parse_call2_response(raw)
    assert result[0]["id"] == "p1"
    assert "}" in result[0]["trajectory_signal"]


def test_parse_escaped_quotes_in_string():
    """C1 regression: escaped quotes inside strings must not break scanner."""
    raw = r'{"sub_problems": [{"id": "p1", "raw_text": "he said \"hello\"", "failure_summary": "test"}]}'
    result = parse_call1_response(raw)
    assert len(result) == 1


def test_parse_call2_preserves_label_proposals():
    from module0.parsing import parse_call2_response
    raw = '''[{
        "id": "p1",
        "target_capability": ["new_cap"],
        "trajectory_signal": "sig",
        "hyde_positive": ["seg1 twentychars.........", "seg2 twentychars........."],
        "keywords": ["k"],
        "structured_filters": {},
        "confidence": 0.9,
        "route": "pass",
        "label_proposals": [{"label": "new_cap", "description": "d", "parent": null, "keywords": ["k"], "taxonomy_extension": true}]
    }]'''
    items = parse_call2_response(raw)
    assert items[0]["label_proposals"][0]["label"] == "new_cap"


def test_parse_call2_ok_without_label_proposals():
    from module0.parsing import parse_call2_response
    raw = '''[{
        "id": "p1", "target_capability": ["c"], "trajectory_signal": "s",
        "hyde_positive": ["seg1 twentychars.........", "seg2 twentychars........."],
        "keywords": ["k"], "structured_filters": {}, "confidence": 0.9, "route": "pass"
    }]'''
    items = parse_call2_response(raw)
    assert "label_proposals" not in items[0]
