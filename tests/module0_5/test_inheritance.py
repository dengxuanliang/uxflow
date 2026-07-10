from dataclasses import dataclass

from module0.taxonomy import Taxonomy, TaxonomyLabel
from module0_5.inheritance import rerank_with_inheritance


@dataclass
class _Sig:
    trajectory_id: str
    capability_labels: list | None


@dataclass
class _Hit:
    signature: _Sig
    rrf_score: float


def _lbl(name, parent):
    return TaxonomyLabel(
        label=name, parent=parent, new_root=(parent is None),
        description=f"d{name}", keywords=[], description_embedding=[],
        taxonomy_extension=False, created_at="",
    )


def _tax():
    return Taxonomy(version="0.1.0", updated_at="", labels=[
        _lbl("error_recovery", None),
        _lbl("effective_error_fix", "error_recovery"),
        _lbl("handle_async_race", "error_recovery"),
    ])


def test_exact_label_gets_full_weight():
    hits = [_Hit(_Sig("A", ["handle_async_race"]), 1.0)]
    out = rerank_with_inheritance(hits, target_label="handle_async_race", taxonomy=_tax())
    assert out[0].rrf_score == 1.0


def test_sibling_leaf_gets_inherited_weight():
    hits = [_Hit(_Sig("B", ["effective_error_fix"]), 1.0)]
    out = rerank_with_inheritance(hits, target_label="handle_async_race", taxonomy=_tax())
    assert abs(out[0].rrf_score - 0.3) < 1e-9


def test_parent_labeled_gets_inherited_weight():
    hits = [_Hit(_Sig("C", ["error_recovery"]), 1.0)]
    out = rerank_with_inheritance(hits, target_label="handle_async_race", taxonomy=_tax())
    assert abs(out[0].rrf_score - 0.3) < 1e-9


def test_unrelated_label_no_boost_no_penalty():
    hits = [_Hit(_Sig("D", ["some_other_label"]), 1.0)]
    out = rerank_with_inheritance(hits, target_label="handle_async_race", taxonomy=_tax())
    assert out[0].rrf_score == 1.0


def test_none_capability_labels_treated_as_unrelated():
    hits = [_Hit(_Sig("E", None), 1.0)]
    out = rerank_with_inheritance(hits, target_label="handle_async_race", taxonomy=_tax())
    assert out[0].rrf_score == 1.0


def test_results_sorted_desc_by_weighted_score():
    hits = [_Hit(_Sig("B", ["effective_error_fix"]), 1.0), _Hit(_Sig("A", ["handle_async_race"]), 1.0)]
    out = rerank_with_inheritance(hits, target_label="handle_async_race", taxonomy=_tax())
    assert out[0].signature.trajectory_id == "A"  # exact(1.0) before inherited(0.3)


def test_input_hits_not_mutated():
    hit = _Hit(_Sig("B", ["effective_error_fix"]), 1.0)
    rerank_with_inheritance([hit], target_label="handle_async_race", taxonomy=_tax())
    assert hit.rrf_score == 1.0  # original untouched (pure function)


def test_empty_hits_returns_empty():
    assert rerank_with_inheritance([], target_label="handle_async_race", taxonomy=_tax()) == []
