"""Persistent SliceStore backed by a single SQLite file. Vectors as float32 BLOB.

Startup loads all signatures into an in-memory mirror; recall reuses recall_core
(identical scoring to MemoryIndex). Writes go to both the mirror and SQLite.
"""
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 dengxuanliang

from __future__ import annotations

import json
import pathlib
import sqlite3

import numpy as np

from module1 import recall_core
from module1.models import Slice, Step, TrajectorySignature

__all__ = ["SqliteSliceStore"]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS signatures (
    trajectory_id TEXT NOT NULL,
    slice_index INTEGER NOT NULL,
    step_range TEXT NOT NULL,
    step_count INTEGER NOT NULL,
    turn_count INTEGER NOT NULL,
    languages TEXT NOT NULL,
    tools_used TEXT NOT NULL,
    has_error_pattern INTEGER NOT NULL,
    has_success_pattern INTEGER NOT NULL,
    has_verification_step INTEGER NOT NULL,
    bm25_tokens TEXT NOT NULL,
    embedding BLOB,
    capability_labels TEXT,
    PRIMARY KEY (trajectory_id, slice_index)
);
CREATE TABLE IF NOT EXISTS slice_sources (
    trajectory_id TEXT NOT NULL,
    slice_index INTEGER NOT NULL,
    slice_json TEXT NOT NULL,
    PRIMARY KEY (trajectory_id, slice_index)
);
"""


def _emb_to_blob(emb: list[float]) -> bytes | None:
    if not emb:
        return None
    return np.asarray(emb, dtype=np.float32).tobytes()


def _blob_to_emb(blob: bytes | None) -> list[float]:
    if blob is None:
        return []
    return np.frombuffer(blob, dtype=np.float32).tolist()


def _row_to_sig(row: sqlite3.Row) -> TrajectorySignature:
    labels = row["capability_labels"]
    return TrajectorySignature(
        trajectory_id=row["trajectory_id"],
        slice_index=row["slice_index"],
        step_range=tuple(json.loads(row["step_range"])),
        step_count=row["step_count"],
        turn_count=row["turn_count"],
        languages=json.loads(row["languages"]),
        tools_used=json.loads(row["tools_used"]),
        has_error_pattern=bool(row["has_error_pattern"]),
        has_success_pattern=bool(row["has_success_pattern"]),
        has_verification_step=bool(row["has_verification_step"]),
        bm25_tokens=json.loads(row["bm25_tokens"]),
        embedding=_blob_to_emb(row["embedding"]),
        capability_labels=json.loads(labels) if labels is not None else None,
    )


def _slice_to_json(sl: Slice) -> str:
    return json.dumps({
        "trajectory_id": sl.trajectory_id, "slice_index": sl.slice_index,
        "start_step": sl.start_step, "end_step": sl.end_step,
        "steps": [vars(s) for s in sl.steps],
    })


def _json_to_slice(text: str) -> Slice:
    d = json.loads(text)
    return Slice(
        trajectory_id=d["trajectory_id"], slice_index=d["slice_index"],
        start_step=d["start_step"], end_step=d["end_step"],
        steps=[Step(**s) for s in d["steps"]],
    )


class SqliteSliceStore:
    """Persistent SliceStore. Implements the SliceStore Protocol."""

    def __init__(self, db_path: str | pathlib.Path):
        # isolation_level=None → autocommit; we control transactions explicitly where needed.
        self._conn = sqlite3.connect(str(db_path), isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=30000")
        self._conn.executescript(_SCHEMA)
        self._signatures: list[TrajectorySignature] = []
        self._by_key: dict[tuple[str, int], TrajectorySignature] = {}
        self._dim: int | None = None
        self._load()

    def _load(self) -> None:
        rows = self._conn.execute("SELECT * FROM signatures").fetchall()
        self._signatures = [_row_to_sig(r) for r in rows]
        self._by_key = {(s.trajectory_id, s.slice_index): s for s in self._signatures}
        for sig in self._signatures:
            if sig.embedding:
                self._dim = len(sig.embedding)
                break

    @property
    def size(self) -> int:
        return len(self._signatures)

    def _check_dim(self, sig: TrajectorySignature) -> None:
        if not sig.embedding:
            return
        if self._dim is None:
            self._dim = len(sig.embedding)
        elif len(sig.embedding) != self._dim:
            raise ValueError(
                f"embedding dimension {len(sig.embedding)} != expected {self._dim}"
            )

    def _mirror_upsert(self, sig: TrajectorySignature) -> None:
        """Keep the in-memory mirror consistent with DB INSERT OR REPLACE (upsert by key)."""
        key = (sig.trajectory_id, sig.slice_index)
        old = self._by_key.get(key)
        if old is not None:
            self._signatures.remove(old)
        self._by_key[key] = sig
        self._signatures.append(sig)

    def add(self, sig) -> None:
        self.add_batch([sig])

    def add_batch(self, sigs: list[TrajectorySignature]) -> None:
        for sig in sigs:
            self._check_dim(sig)
        self._conn.executemany(
            """INSERT OR REPLACE INTO signatures
               (trajectory_id, slice_index, step_range, step_count, turn_count,
                languages, tools_used, has_error_pattern, has_success_pattern,
                has_verification_step, bm25_tokens, embedding, capability_labels)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            [(s.trajectory_id, s.slice_index, json.dumps(list(s.step_range)),
              s.step_count, s.turn_count, json.dumps(s.languages),
              json.dumps(s.tools_used), int(s.has_error_pattern),
              int(s.has_success_pattern), int(s.has_verification_step),
              json.dumps(s.bm25_tokens), _emb_to_blob(s.embedding),
              json.dumps(s.capability_labels) if s.capability_labels is not None else None)
             for s in sigs])
        for sig in sigs:
            self._mirror_upsert(sig)

    def recall(self, *, structured_filters, keywords, query_embeddings, top_n=20):
        return recall_core.recall(
            self._signatures, structured_filters=structured_filters,
            keywords=keywords, query_embeddings=query_embeddings, top_n=top_n)

    def _get_signature(self, trajectory_id, slice_index):
        return self._by_key.get((trajectory_id, slice_index))

    def update_labels(self, trajectory_id, slice_index, labels):
        sig = self._get_signature(trajectory_id, slice_index)
        if sig is None:
            return
        existing = sig.capability_labels or []
        merged = list(dict.fromkeys([*existing, *labels]))
        sig.capability_labels = merged
        self._conn.execute(
            "UPDATE signatures SET capability_labels=? WHERE trajectory_id=? AND slice_index=?",
            (json.dumps(merged), trajectory_id, slice_index))

    def set_slice_source(self, trajectory_id, slice_index, slice_obj):
        self._conn.execute(
            "INSERT OR REPLACE INTO slice_sources VALUES (?,?,?)",
            (trajectory_id, slice_index, _slice_to_json(slice_obj)))

    def get_slice(self, trajectory_id, slice_index):
        row = self._conn.execute(
            "SELECT slice_json FROM slice_sources WHERE trajectory_id=? AND slice_index=?",
            (trajectory_id, slice_index)).fetchone()
        return _json_to_slice(row["slice_json"]) if row else None

    def get_slice_obj(self, trajectory_id, slice_index):
        return self.get_slice(trajectory_id, slice_index)

    def close(self) -> None:
        self._conn.close()
