from module0.taxonomy import TaxonomyLabel
from module0_5 import LabelProposal
from module0_5.evolution import resolve_proposal, ProposalResolution


def _existing(name, parent, emb):
    return TaxonomyLabel(
        label=name, parent=parent, new_root=(parent is None),
        description=f"desc {name}", keywords=[name],
        description_embedding=emb, taxonomy_extension=False,
        created_at="2026-01-01T00:00:00Z",
    )


def _proposal(emb, parent="error_recovery"):
    return LabelProposal(
        label="new_leaf", description="新能力", parent=parent,
        description_embedding=emb, keywords=["kw"], source_sub_problem_id="p1",
    )


def _labels():
    return [
        _existing("error_recovery", None, [1.0, 0.0, 0.0]),
        _existing("effective_error_fix", "error_recovery", [0.9, 0.1, 0.0]),
    ]


def test_duplicate_when_cosine_above_dedup_threshold():
    prop = _proposal([0.9, 0.1, 0.0])  # ~identical to existing leaf
    res = resolve_proposal(prop, _labels())
    assert res.kind == "duplicate"
    assert res.maps_to == "effective_error_fix"


def test_new_leaf_when_close_to_a_root():
    prop = _proposal([0.7, 0.7, 0.2])  # cosine to root ~0.69 (mount, not dup)
    res = resolve_proposal(prop, _labels())
    assert res.kind == "new_leaf"
    assert res.parent == "error_recovery"
    assert res.new_root is False


def test_new_root_when_far_from_all_roots():
    prop = _proposal([0.0, 0.0, 1.0])  # orthogonal to root
    res = resolve_proposal(prop, _labels())
    assert res.kind == "new_root"
    assert res.parent is None
    assert res.new_root is True


def test_thresholds_are_configurable():
    prop = _proposal([0.9, 0.1, 0.0])
    res = resolve_proposal(prop, _labels(), dedup_threshold=1.0)
    assert res.kind != "duplicate"
