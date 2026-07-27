from module0.taxonomy import Taxonomy, TaxonomyLabel, MemoryTaxonomyStore


def _lbl(name, parent=None):
    return TaxonomyLabel(
        label=name, parent=parent, new_root=(parent is None),
        description=f"desc {name}", keywords=[name],
        description_embedding=[1.0, 0.0], taxonomy_extension=False,
        created_at="2026-01-01T00:00:00Z",
    )


def test_load_from_file(taxonomy_v0_path):
    t = Taxonomy.load(taxonomy_v0_path)
    assert t.version == "0.1.0"
    assert len(t.labels) == 16  # 5 parent + 11 leaf


def test_leaf_labels(taxonomy_v0_path):
    t = Taxonomy.load(taxonomy_v0_path)
    leaves = t.leaf_labels()
    assert len(leaves) == 11
    assert "valid_syntax_in_toolcall" in [lbl.label for lbl in leaves]


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


def test_init_dedupes_same_name_labels():
    """Taxonomy.__init__ must not retain duplicate label names — without
    this guard, a duplicated list would leave entries in self.labels
    unreachable via get() (last-wins in _by_name)."""
    dup = [_lbl("x"), _lbl("y"), _lbl("x")]  # "x" appears twice
    t = Taxonomy(version="0.1.0", updated_at="2026-01-01T00:00:00Z", labels=dup)
    assert len(t.labels) == 2  # "x" deduped
    assert t.get("x") is not None
    assert t.get("y") is not None


def test_memory_store_add_label_replaces_same_name():
    """MemoryTaxonomyStore.add_label must replace an existing same-name label
    (semantic parity with SqliteTaxonomyStore's INSERT OR REPLACE), not append
    a duplicate that would later be silently deduped by Taxonomy.__init__."""
    tax = Taxonomy(version="0.1.0", updated_at="2026-01-01T00:00:00Z",
                   labels=[_lbl("root"), _lbl("child", parent="root")])
    store = MemoryTaxonomyStore(tax)
    assert len(store.existing_labels()) == 2
    # add a same-name replacement with a new description
    replacement = TaxonomyLabel(
        label="child", parent="root", new_root=False,
        description="updated desc", keywords=["new"],
        description_embedding=[0.5, 0.5], taxonomy_extension=True,
        created_at="2026-07-27T00:00:00Z",
    )
    store.add_label(replacement)
    labels = store.existing_labels()
    assert len(labels) == 2  # not 3 — replaced, not appended
    snap = store.snapshot()
    child = snap.get("child")
    assert child.description == "updated desc"
