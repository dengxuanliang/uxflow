import pytest
from module0.taxonomy import Taxonomy


def test_load_from_file(taxonomy_v0_path):
    t = Taxonomy.load(taxonomy_v0_path)
    assert t.version == "0.1.0"
    assert len(t.labels) == 16  # 5 parent + 11 leaf


def test_leaf_labels(taxonomy_v0_path):
    t = Taxonomy.load(taxonomy_v0_path)
    leaves = t.leaf_labels()
    assert len(leaves) == 11
    assert "valid_syntax_in_toolcall" in [l.label for l in leaves]


def test_get_label(taxonomy_v0_path):
    t = Taxonomy.load(taxonomy_v0_path)
    label = t.get("valid_syntax_in_toolcall")
    assert label is not None
    assert label.parent == "code_generation"


def test_get_nonexistent(taxonomy_v0_path):
    t = Taxonomy.load(taxonomy_v0_path)
    assert t.get("nonexistent_label") is None


def test_prompt_injection_text(taxonomy_v0_path):
    t = Taxonomy.load(taxonomy_v0_path)
    text = t.to_prompt_text()
    assert "code_generation" in text
    assert "valid_syntax_in_toolcall" in text
    assert "工具调用中生成合法代码" in text


def test_empty_taxonomy():
    t = Taxonomy.from_dict({"version": "0.0.0", "updated_at": "2026-01-01T00:00:00Z", "labels": []})
    assert len(t.labels) == 0
    assert t.to_prompt_text() == ""
    assert t.leaf_labels() == []


def test_is_empty_vs_non_empty(taxonomy_v0_path):
    t = Taxonomy.load(taxonomy_v0_path)
    assert not t.is_empty
    empty = Taxonomy.from_dict({"version": "0.0.0", "updated_at": "", "labels": []})
    assert empty.is_empty
