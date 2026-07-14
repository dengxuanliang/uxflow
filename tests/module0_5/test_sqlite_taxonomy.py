# SPDX-License-Identifier: Apache-2.0
from module0.taxonomy import Taxonomy, TaxonomyLabel, TaxonomyStore
from module0.sqlite_taxonomy import SqliteTaxonomyStore


def _tax():
    return Taxonomy(version="0.1.0", updated_at="2026-07-13T00:00:00Z", labels=[
        TaxonomyLabel(label="error_recovery", parent=None, new_root=False,
                      description="错误恢复", keywords=["error"],
                      description_embedding=[1.0, 0.0], taxonomy_extension=False,
                      created_at="2026-01-01T00:00:00Z"),
    ])


def _new_label():
    return TaxonomyLabel(label="fix_runtime_exception", parent="error_recovery",
                         new_root=False, description="修复运行时异常", keywords=["TypeError"],
                         description_embedding=[0.9, 0.1], taxonomy_extension=True,
                         created_at="2026-07-13T00:00:00Z")


def test_seed_and_snapshot(tmp_path):
    store = SqliteTaxonomyStore(tmp_path / "t.db", seed=_tax())
    snap = store.snapshot()
    assert snap.get("error_recovery") is not None
    assert snap.version == "0.1.0"


def test_add_label_bumps_patch(tmp_path):
    store = SqliteTaxonomyStore(tmp_path / "t.db", seed=_tax())
    store.add_label(_new_label())
    assert store.snapshot().version == "0.1.1"
    assert store.snapshot().get("fix_runtime_exception").parent == "error_recovery"


def test_existing_untouched_after_add(tmp_path):
    store = SqliteTaxonomyStore(tmp_path / "t.db", seed=_tax())
    store.add_label(_new_label())
    assert store.snapshot().get("error_recovery").description == "错误恢复"


def test_persistence_across_reopen(tmp_path):
    db = tmp_path / "t.db"
    SqliteTaxonomyStore(db, seed=_tax()).add_label(_new_label())
    store2 = SqliteTaxonomyStore(db)  # no seed: load existing
    assert store2.snapshot().get("fix_runtime_exception") is not None
    assert store2.snapshot().version == "0.1.1"


def test_seed_idempotent_does_not_reseed(tmp_path):
    db = tmp_path / "t.db"
    SqliteTaxonomyStore(db, seed=_tax()).add_label(_new_label())
    # reopen WITH seed again: must not wipe/duplicate existing rows
    store2 = SqliteTaxonomyStore(db, seed=_tax())
    labels = store2.snapshot().labels
    names = [l.label for l in labels]
    assert names.count("error_recovery") == 1
    assert "fix_runtime_exception" in names


def test_satisfies_protocol(tmp_path):
    store = SqliteTaxonomyStore(tmp_path / "t.db", seed=_tax())
    assert isinstance(store, TaxonomyStore)


def test_ingest_proposal_on_sqlite_backend(tmp_path):
    from module0_5 import ingest_proposal
    from module0_5.models import LabelProposal
    store = SqliteTaxonomyStore(tmp_path / "t.db", seed=_tax())
    prop = LabelProposal(
        label="fix_runtime_exception", description="修复运行时异常",
        parent="error_recovery", description_embedding=[0.0, 1.0],  # orthogonal → not duplicate
        keywords=["TypeError"], source_sub_problem_id="p1")
    res = ingest_proposal(prop, store, created_at="2026-07-13T00:00:00Z")
    assert res.kind in ("new_leaf", "new_root")  # orthogonal embedding never dedups
    assert store.snapshot().get("fix_runtime_exception") is not None
