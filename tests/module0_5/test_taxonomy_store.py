from module0.taxonomy import Taxonomy, TaxonomyLabel, TaxonomyStore


def _lbl(name, parent=None, emb=None):
    return TaxonomyLabel(
        label=name, parent=parent, new_root=(parent is None),
        description=f"desc {name}", keywords=[name],
        description_embedding=emb or [0.0], taxonomy_extension=False,
        created_at="2026-01-01T00:00:00Z",
    )


def _base():
    return Taxonomy(version="0.1.0", updated_at="2026-01-01T00:00:00Z",
                    labels=[_lbl("error_recovery"), _lbl("effective_error_fix", "error_recovery")])


def test_snapshot_returns_readonly_taxonomy():
    store = TaxonomyStore(_base())
    snap = store.snapshot()
    assert isinstance(snap, Taxonomy)
    assert snap.get("error_recovery") is not None


def test_existing_labels_lists_current():
    store = TaxonomyStore(_base())
    names = {l.label for l in store.existing_labels()}
    assert names == {"error_recovery", "effective_error_fix"}


def test_add_label_appends_and_bumps_patch_version():
    store = TaxonomyStore(_base())
    new = _lbl("handle_async_race", "error_recovery")
    store.add_label(new)
    assert store.snapshot().get("handle_async_race") is not None
    assert store.snapshot().version == "0.1.1"  # patch +1


def test_add_label_does_not_touch_existing():
    store = TaxonomyStore(_base())
    store.add_label(_lbl("handle_async_race", "error_recovery"))
    assert store.snapshot().get("error_recovery").label == "error_recovery"


def test_save_writes_contract_json(tmp_path):
    import json
    store = TaxonomyStore(_base())
    store.add_label(_lbl("handle_async_race", "error_recovery"))
    out = tmp_path / "tax.json"
    store.save(out)
    data = json.loads(out.read_text())
    assert data["version"] == "0.1.1"
    labels = {l["label"] for l in data["labels"]}
    assert "handle_async_race" in labels
