"""Persistent TaxonomyStore backed by SQLite. Implements the TaxonomyStore Protocol."""
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 dengxuanliang

from __future__ import annotations

import json
import pathlib
import sqlite3

import numpy as np

from module0.taxonomy import Taxonomy, TaxonomyLabel, _bump_patch

__all__ = ["SqliteTaxonomyStore"]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS tax_meta (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    version TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS tax_labels (
    label TEXT PRIMARY KEY,
    parent TEXT,
    new_root INTEGER NOT NULL,
    description TEXT NOT NULL,
    keywords TEXT NOT NULL,
    description_embedding BLOB,
    taxonomy_extension INTEGER NOT NULL,
    created_at TEXT NOT NULL
);
"""


def _emb_to_blob(emb):
    return np.asarray(emb, dtype=np.float32).tobytes() if emb else None


def _blob_to_emb(blob):
    return np.frombuffer(blob, dtype=np.float32).tolist() if blob is not None else []


def _row_to_label(row) -> TaxonomyLabel:
    return TaxonomyLabel(
        label=row["label"], parent=row["parent"], new_root=bool(row["new_root"]),
        description=row["description"], keywords=json.loads(row["keywords"]),
        description_embedding=_blob_to_emb(row["description_embedding"]),
        taxonomy_extension=bool(row["taxonomy_extension"]), created_at=row["created_at"],
    )


class SqliteTaxonomyStore:
    """可变演化层，SQLite 持久实现。唯一写入点。"""

    def __init__(self, db_path: str | pathlib.Path, *, seed: Taxonomy | None = None):
        self._conn = sqlite3.connect(str(db_path), isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=30000")
        self._conn.executescript(_SCHEMA)
        if seed is not None and self._is_empty():
            self._seed(seed)

    def _is_empty(self) -> bool:
        return self._conn.execute("SELECT COUNT(*) FROM tax_meta").fetchone()[0] == 0

    def _seed(self, tax: Taxonomy) -> None:
        self._conn.execute("INSERT INTO tax_meta (id, version, updated_at) VALUES (1,?,?)",
                           (tax.version, tax.updated_at))
        for lbl in tax.labels:
            self._insert_label(lbl)

    def _insert_label(self, lbl: TaxonomyLabel) -> None:
        self._conn.execute(
            """INSERT OR REPLACE INTO tax_labels
               (label, parent, new_root, description, keywords,
                description_embedding, taxonomy_extension, created_at)
               VALUES (?,?,?,?,?,?,?,?)""",
            (lbl.label, lbl.parent, int(lbl.new_root), lbl.description,
             json.dumps(lbl.keywords), _emb_to_blob(lbl.description_embedding),
             int(lbl.taxonomy_extension), lbl.created_at))

    def snapshot(self) -> Taxonomy:
        meta = self._conn.execute("SELECT version, updated_at FROM tax_meta WHERE id=1").fetchone()
        version = meta["version"] if meta else "0.0.0"
        updated_at = meta["updated_at"] if meta else ""
        rows = self._conn.execute("SELECT * FROM tax_labels").fetchall()
        return Taxonomy(version=version, updated_at=updated_at,
                        labels=[_row_to_label(r) for r in rows])

    def existing_labels(self) -> list[TaxonomyLabel]:
        rows = self._conn.execute("SELECT * FROM tax_labels").fetchall()
        return [_row_to_label(r) for r in rows]

    def add_label(self, label: TaxonomyLabel, *, now: str | None = None) -> None:
        self._insert_label(label)
        meta = self._conn.execute("SELECT version FROM tax_meta WHERE id=1").fetchone()
        if meta:
            self._conn.execute(
                "UPDATE tax_meta SET version=?, updated_at=COALESCE(?, updated_at) WHERE id=1",
                (_bump_patch(meta["version"]), now))
        else:
            self._conn.execute(
                "INSERT INTO tax_meta (id, version, updated_at) VALUES (1,?,?)",
                ("0.0.1", now or ""))

    def save(self, path) -> None:
        """Export contract §5.2 JSON (parity with MemoryTaxonomyStore.save)."""
        snap = self.snapshot()
        data = {"version": snap.version, "updated_at": snap.updated_at, "labels": [
            {"label": l.label, "parent": l.parent, "new_root": l.new_root,
             "description": l.description, "keywords": l.keywords,
             "description_embedding": l.description_embedding,
             "taxonomy_extension": l.taxonomy_extension, "created_at": l.created_at}
            for l in snap.labels]}
        with open(path, "w") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def close(self) -> None:
        self._conn.close()
