import pytest
from module0.prompts import build_call1_messages, build_call2_messages, build_call3_messages
from module0.taxonomy import Taxonomy


def test_call1_messages_structure():
    msgs = build_call1_messages("写入py文件有语法错误，并且工具调用结构经常出错")
    assert len(msgs) >= 2
    assert msgs[0]["role"] == "system"
    assert msgs[-1]["role"] == "user"
    assert "写入py文件有语法错误" in msgs[-1]["content"]


def test_call1_system_prompt_contains_rules():
    msgs = build_call1_messages("test input")
    system = msgs[0]["content"]
    assert "子问题" in system or "sub-problem" in system.lower()
    assert "JSON" in system


def test_call2_messages_with_taxonomy(taxonomy_v0_path):
    taxonomy = Taxonomy.load(taxonomy_v0_path)
    sub_problems = [
        {"id": "p1", "raw_text": "写入py文件有语法错误", "failure_summary": "写入 py 文件语法错误"},
    ]
    msgs = build_call2_messages(sub_problems, taxonomy)
    assert len(msgs) >= 2
    system = msgs[0]["content"]
    assert "valid_syntax_in_toolcall" in system
    assert "structured_filters" in system or "languages" in system


def test_call2_messages_empty_taxonomy():
    taxonomy = Taxonomy.from_dict({"version": "0.0.0", "updated_at": "", "labels": []})
    sub_problems = [
        {"id": "p1", "raw_text": "test", "failure_summary": "test"},
    ]
    msgs = build_call2_messages(sub_problems, taxonomy)
    system = msgs[0]["content"]
    assert "自由提议" in system or "propose" in system.lower() or "freely" in system.lower()


def test_call3_messages():
    ambiguous_problems = [
        {"id": "p12", "raw_text": "多步任务未完成即终止", "failure_summary": "多步任务中途停止",
         "target_capability": ["persist_through_truncation"]},
    ]
    msgs = build_call3_messages(ambiguous_problems)
    assert len(msgs) >= 2
    system = msgs[0]["content"]
    assert "消歧" in system or "disambiguat" in system.lower()
    user = msgs[-1]["content"]
    assert "p12" in user


def test_call2_output_schema_mentioned():
    taxonomy = Taxonomy.from_dict({"version": "0.0.0", "updated_at": "", "labels": []})
    msgs = build_call2_messages([{"id": "p1", "raw_text": "t", "failure_summary": "s"}], taxonomy)
    system = msgs[0]["content"]
    for field in ["target_capability", "trajectory_signal", "hyde_positive", "keywords", "confidence", "route"]:
        assert field in system, f"Missing field '{field}' in Call 2 system prompt"
