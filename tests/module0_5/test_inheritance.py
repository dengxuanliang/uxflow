from module0.taxonomy import Taxonomy, TaxonomyLabel
from module0_5.inheritance import rerank_with_inheritance


class _Sig:
    def __init__(self, tid, labels):
        self.trajectory_id = tid
        self.capability_labels = labels


class _Hit:
    def __init__(self, tid, labels, score):
        self.signature = _Sig(tid, labels)
        self.rrf_score = score


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
    hits = [_Hit("A", ["handle_async_race"], 1.0)]
    out = rerank_with_inheritance(hits, target_label="handle_async_race", taxonomy=_tax())
    assert out[0].rrf_score == 1.0


def test_sibling_leaf_gets_inherited_weight():
    hits = [_Hit("B", ["effective_error_fix"], 1.0)]
    out = rerank_with_inheritance(hits, target_label="handle_async_race", taxonomy=_tax())
    assert abs(out[0].rrf_score - 0.3) < 1e-9


def test_parent_labeled_gets_inherited_weight():
    hits = [_Hit("C", ["error_recovery"], 1.0)]
    out = rerank_with_inheritance(hits, target_label="handle_async_race", taxonomy=_tax())
    assert abs(out[0].rrf_score - 0.3) < 1e-9


def test_unrelated_label_no_boost_no_penalty():
    hits = [_Hit("D", ["some_other_label"], 1.0)]
    out = rerank_with_inheritance(hits, target_label="handle_async_race", taxonomy=_tax())
    assert out[0].rrf_score == 1.0


def test_none_capability_labels_treated_as_unrelated():
    hits = [_Hit("E", None, 1.0)]
    out = rerank_with_inheritance(hits, target_label="handle_async_race", taxonomy=_tax())
    assert out[0].rrf_score == 1.0


def test_results_sorted_desc_by_weighted_score():
    hits = [_Hit("B", ["effective_error_fix"], 1.0), _Hit("A", ["handle_async_race"], 1.0)]
    out = rerank_with_inheritance(hits, target_label="handle_async_race", taxonomy=_tax())
    assert out[0].signature.trajectory_id == "A"  # exact(1.0) before inherited(0.3)
