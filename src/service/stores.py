"""SQLite-backed stores for the Inspector service: problems, trajectories, judge cache.

Each store mirrors SqliteSliceStore's connection setup (WAL, autocommit, Row factory)
and is paired with a runtime_checkable Protocol. No LLM/network dependencies.
"""
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 dengxuanliang

from __future__ import annotations

import json
import pathlib
import sqlite3
from typing import Protocol, runtime_checkable

import numpy as np

from service.dedup import cosine, problem_id_of

__all__ = [
    "SqliteProblemStore",
    "SqliteTrajectoryStore",
    "SqliteJudgeCache",
    "ProblemStore",
    "TrajectoryStore",
    "JudgeCache",
]


def _emb_to_blob(emb: list[float]) -> bytes | None:
    if not emb:
        return None
    return np.asarray(emb, dtype=np.float32).tobytes()


def _blob_to_emb(blob: bytes | None) -> list[float]:
    if blob is None:
        return []
    return np.frombuffer(blob, dtype=np.float32).tolist()


_PROBLEM_SCHEMA = """
CREATE TABLE IF NOT EXISTS problems (
    problem_id TEXT PRIMARY KEY,
    raw_question TEXT NOT NULL,
    question_embedding BLOB,
    spec_json TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
"""

_TRAJECTORY_SCHEMA = """
CREATE TABLE IF NOT EXISTS trajectories (
    trajectory_id TEXT PRIMARY KEY,
    steps_json TEXT NOT NULL,
    source_path TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
"""

_JUDGE_SCHEMA = """
CREATE TABLE IF NOT EXISTS judge_cache (
    sub_problem_id TEXT NOT NULL,
    trajectory_id TEXT NOT NULL,
    slice_index INTEGER NOT NULL,
    matched INTEGER NOT NULL,
    confidence REAL NOT NULL,
    spans_json TEXT NOT NULL,
    evidence_step INTEGER,
    criteria_hit_json TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (sub_problem_id, trajectory_id, slice_index)
);
"""

# 旧库迁移（关键）：CREATE TABLE IF NOT EXISTS 对**已存在**的旧库不加列，故对可回溯字段
# 显式跑幂等 ALTER TABLE。重复运行时 "duplicate column name" 由 OperationalError 吞掉。
_JUDGE_MIGRATIONS = (
    "ALTER TABLE judge_cache ADD COLUMN evidence_step INTEGER",
    "ALTER TABLE judge_cache ADD COLUMN criteria_hit_json TEXT",
)


def _migrate_judge_cache(conn: sqlite3.Connection) -> None:
    """幂等补列：旧库（仅 matched/confidence/spans_json 三列）升级出可回溯两列。"""
    for stmt in _JUDGE_MIGRATIONS:
        try:
            conn.execute(stmt)
        except sqlite3.OperationalError:
            pass  # duplicate column name → 已存在，幂等跳过


def _connect(db_path: str | pathlib.Path, schema: str) -> sqlite3.Connection:
    # isolation_level=None → autocommit: each statement commits immediately (single-writer V1).
    # check_same_thread=False: the ASGI server may touch a store from the event-loop
    # thread AND from a worker thread (FastAPI runs sync deps in a threadpool; the
    # legacy /runs path offloads index build via asyncio.to_thread). Safe here because
    # writes are serialized by the app's _run_lock and reads are atomic single statements
    # over a WAL connection.
    conn = sqlite3.connect(str(db_path), isolation_level=None, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")
    conn.executescript(schema)
    return conn


@runtime_checkable
class ProblemStore(Protocol):
    """Problem manifest with vector-based nearest-neighbor dedup."""

    def add(self, raw_question: str, embedding: list[float], spec: dict) -> str: ...

    def nearest(
        self, embedding: list[float]
    ) -> tuple[str, float, dict] | None: ...

    def get(self, problem_id: str) -> dict | None: ...

    def count(self) -> int: ...


@runtime_checkable
class TrajectoryStore(Protocol):
    """Full-text trajectory storage keyed by trajectory id."""

    def upsert(
        self, trajectory_id: str, steps: list[dict], *, source_path: str
    ) -> None: ...

    def get(self, trajectory_id: str) -> dict | None: ...

    def count(self) -> int: ...


@runtime_checkable
class JudgeCache(Protocol):
    """Pairwise judge verdict cache keyed by (sub_problem, trajectory, slice)."""

    def get(
        self, sub_problem_id: str, trajectory_id: str, slice_index: int
    ) -> dict | None: ...

    def put(
        self,
        sub_problem_id: str,
        trajectory_id: str,
        slice_index: int,
        verdict: dict,
    ) -> None: ...


class SqliteProblemStore:
    """Persistent ProblemStore. Implements the ProblemStore Protocol."""

    def __init__(self, db_path: str | pathlib.Path):
        self._conn = _connect(db_path, _PROBLEM_SCHEMA)
        # In-memory mirror for linear cosine scan: pid -> embedding ndarray.
        # spec is re-queried from DB on hit to keep the mirror light.
        self._embeddings: dict[str, np.ndarray] = {}
        self._load()

    def _load(self) -> None:
        rows = self._conn.execute(
            "SELECT problem_id, question_embedding FROM problems"
        ).fetchall()
        for row in rows:
            emb = _blob_to_emb(row["question_embedding"])
            if emb:
                self._embeddings[row["problem_id"]] = np.asarray(emb, dtype=np.float32)

    def add(self, raw_question: str, embedding: list[float], spec: dict) -> str:
        pid = problem_id_of(raw_question)
        spec_json = json.dumps(spec, ensure_ascii=False)
        # INSERT OR REPLACE resets created_at to the DEFAULT — acceptable for V1.
        self._conn.execute(
            """INSERT OR REPLACE INTO problems
               (problem_id, raw_question, question_embedding, spec_json)
               VALUES (?,?,?,?)""",
            (pid, raw_question, _emb_to_blob(embedding), spec_json),
        )
        if embedding:
            self._embeddings[pid] = np.asarray(embedding, dtype=np.float32)
        else:
            self._embeddings.pop(pid, None)
        return pid

    def nearest(self, embedding: list[float]) -> tuple[str, float, dict] | None:
        if not embedding or not self._embeddings:
            return None
        query = np.asarray(embedding, dtype=np.float32)
        if float(np.linalg.norm(query)) == 0.0:
            return None
        best_pid: str | None = None
        best_score = -1.0
        for pid, emb in self._embeddings.items():
            score = cosine(query, emb)
            if score > best_score:
                best_score = score
                best_pid = pid
        if best_pid is None:
            return None
        row = self._conn.execute(
            "SELECT spec_json FROM problems WHERE problem_id=?", (best_pid,)
        ).fetchone()
        if row is None:
            return None
        return (best_pid, best_score, json.loads(row["spec_json"]))

    def get(self, problem_id: str) -> dict | None:
        row = self._conn.execute(
            "SELECT problem_id, raw_question, spec_json FROM problems WHERE problem_id=?",
            (problem_id,),
        ).fetchone()
        if row is None:
            return None
        return {
            "problem_id": row["problem_id"],
            "raw_question": row["raw_question"],
            "spec": json.loads(row["spec_json"]),
        }

    def count(self) -> int:
        return self._conn.execute("SELECT COUNT(*) AS n FROM problems").fetchone()["n"]

    def close(self) -> None:
        self._conn.close()


class SqliteTrajectoryStore:
    """Persistent TrajectoryStore. Implements the TrajectoryStore Protocol."""

    def __init__(self, db_path: str | pathlib.Path):
        self._conn = _connect(db_path, _TRAJECTORY_SCHEMA)

    def upsert(
        self, trajectory_id: str, steps: list[dict], *, source_path: str
    ) -> None:
        self._conn.execute(
            """INSERT OR REPLACE INTO trajectories
               (trajectory_id, steps_json, source_path)
               VALUES (?,?,?)""",
            (trajectory_id, json.dumps(steps, ensure_ascii=False), source_path),
        )

    def get(self, trajectory_id: str) -> dict | None:
        row = self._conn.execute(
            "SELECT trajectory_id, steps_json FROM trajectories WHERE trajectory_id=?",
            (trajectory_id,),
        ).fetchone()
        if row is None:
            return None
        return {
            "trajectory_id": row["trajectory_id"],
            "steps": json.loads(row["steps_json"]),
        }

    def count(self) -> int:
        return self._conn.execute(
            "SELECT COUNT(*) AS n FROM trajectories"
        ).fetchone()["n"]

    def close(self) -> None:
        self._conn.close()


class SqliteJudgeCache:
    """Persistent JudgeCache. Implements the JudgeCache Protocol."""

    def __init__(self, db_path: str | pathlib.Path):
        self._conn = _connect(db_path, _JUDGE_SCHEMA)
        # 旧库升级：CREATE TABLE IF NOT EXISTS 不给已存在的旧库加列，此处幂等 ALTER 补列。
        _migrate_judge_cache(self._conn)

    def get(
        self, sub_problem_id: str, trajectory_id: str, slice_index: int
    ) -> dict | None:
        row = self._conn.execute(
            """SELECT matched, confidence, spans_json, evidence_step, criteria_hit_json
               FROM judge_cache
               WHERE sub_problem_id=? AND trajectory_id=? AND slice_index=?""",
            (sub_problem_id, trajectory_id, slice_index),
        ).fetchone()
        if row is None:
            return None
        criteria_raw = row["criteria_hit_json"]
        return {
            "match": bool(row["matched"]),
            "confidence": row["confidence"],
            "spans": json.loads(row["spans_json"]),
            "evidence_step": row["evidence_step"],
            "criteria_hit": json.loads(criteria_raw) if criteria_raw else [],
        }

    def put(
        self,
        sub_problem_id: str,
        trajectory_id: str,
        slice_index: int,
        verdict: dict,
    ) -> None:
        self._conn.execute(
            """INSERT OR REPLACE INTO judge_cache
               (sub_problem_id, trajectory_id, slice_index, matched, confidence,
                spans_json, evidence_step, criteria_hit_json)
               VALUES (?,?,?,?,?,?,?,?)""",
            (
                sub_problem_id,
                trajectory_id,
                slice_index,
                int(verdict["match"]),
                verdict["confidence"],
                json.dumps(verdict["spans"], ensure_ascii=False),
                verdict.get("evidence_step"),
                json.dumps(verdict.get("criteria_hit") or [], ensure_ascii=False),
            ),
        )

    def close(self) -> None:
        self._conn.close()
