from module0.taxonomy import TaxonomyLabel, MemoryTaxonomyStore, Taxonomy
from module0_5 import LabelProposal
from module0_5.evolution import resolve_proposal, ingest_proposal


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


def test_calibrated_mount_threshold_mounts_mid_similarity_child():
    # Guards the real-Qwen calibration (mount_threshold default 0.50, see spec
    # appendix §4): a true child sits at cosine ~0.55 to its root — above the
    # calibrated 0.50 but BELOW the old 0.60. It MUST mount (new_leaf); if the
    # default silently reverts to 0.60 this flips to new_root and fails here.
    prop = _proposal([0.55, 0.835, 0.0])  # cosine to root [1,0,0] ≈ 0.55
    res = resolve_proposal(prop, _labels())
    assert res.kind == "new_leaf"
    assert res.parent == "error_recovery"


def test_thresholds_are_configurable():
    prop = _proposal([0.9, 0.1, 0.0])
    res = resolve_proposal(prop, _labels(), dedup_threshold=1.0)
    assert res.kind != "duplicate"


def _store():
    tax = Taxonomy(version="0.1.0", updated_at="2026-01-01T00:00:00Z", labels=_labels())
    return MemoryTaxonomyStore(tax)


def test_ingest_new_leaf_adds_to_store():
    store = _store()
    prop = _proposal([0.7, 0.7, 0.2])  # new_leaf under error_recovery (mount zone, not dup)
    res = ingest_proposal(prop, store, created_at="2026-07-10T00:00:00Z")
    assert res.kind == "new_leaf"
    added = store.snapshot().get("new_leaf")
    assert added is not None
    assert added.parent == "error_recovery"
    assert added.taxonomy_extension is True
    assert added.created_at == "2026-07-10T00:00:00Z"


def test_ingest_duplicate_does_not_add():
    store = _store()
    before = len(store.existing_labels())
    prop = _proposal([0.9, 0.1, 0.0])  # duplicate of effective_error_fix
    res = ingest_proposal(prop, store, created_at="2026-07-10T00:00:00Z")
    assert res.kind == "duplicate"
    assert len(store.existing_labels()) == before  # not added


def test_ingest_new_root_sets_new_root_flag():
    store = _store()
    prop = _proposal([0.0, 0.0, 1.0])  # new_root (orthogonal)
    ingest_proposal(prop, store, created_at="2026-07-10T00:00:00Z")
    added = store.snapshot().get("new_leaf")  # proposal.label is "new_leaf"
    assert added.new_root is True
    assert added.parent is None
