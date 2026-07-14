"""Persistent backfill job queue on SQLite. Crash-recoverable, priority-ordered.

Time is injected (created_at/now) per repo convention: pure/DB layers never
call datetime.now(); callers pass ISO strings.
"""
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 dengxuanliang

from __future__ import annotations

import json
import pathlib
import sqlite3
from dataclasses import dataclass

__all__ = ["SqliteBackfillQueue", "BackfillJob"]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS backfill_jobs (
    job_id INTEGER PRIMARY KEY AUTOINCREMENT,
    label TEXT NOT NULL,
    priority INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'pending',
    attempts INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT,
    result_json TEXT,
    error TEXT
);
CREATE INDEX IF NOT EXISTS idx_jobs_status_priority
    ON backfill_jobs(status, priority DESC, job_id);
"""


@dataclass
class BackfillJob:
    job_id: int
    label: str
    priority: int
    status: str
    attempts: int


class SqliteBackfillQueue:
    def __init__(self, db_path: str | pathlib.Path):
        # isolation_level=None → autocommit; claim() uses explicit BEGIN IMMEDIATE.
        self._conn = sqlite3.connect(str(db_path), isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=30000")
        self._conn.executescript(_SCHEMA)

    def enqueue(self, label: str, *, priority: int, created_at: str) -> int:
        cur = self._conn.execute(
            "INSERT INTO backfill_jobs (label, priority, created_at) VALUES (?,?,?)",
            (label, priority, created_at))
        return cur.lastrowid

    def claim(self, *, now: str) -> BackfillJob | None:
        """Atomically take the highest-priority pending job → running.

        Autocommit mode (isolation_level=None) means we own the transaction
        explicitly: BEGIN IMMEDIATE grabs the write lock up front so two
        claimants can't select the same row.
        """
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            row = self._conn.execute(
                """SELECT * FROM backfill_jobs WHERE status='pending'
                   ORDER BY priority DESC, job_id ASC LIMIT 1""").fetchone()
            if row is None:
                self._conn.execute("ROLLBACK")
                return None
            self._conn.execute(
                "UPDATE backfill_jobs SET status='running', attempts=attempts+1, updated_at=? WHERE job_id=?",
                (now, row["job_id"]))
            self._conn.execute("COMMIT")
            return BackfillJob(job_id=row["job_id"], label=row["label"],
                               priority=row["priority"], status="running",
                               attempts=row["attempts"] + 1)
        except Exception:
            self._conn.execute("ROLLBACK")
            raise

    def complete(self, job: BackfillJob, *, result: dict, now: str) -> None:
        self._conn.execute(
            "UPDATE backfill_jobs SET status='done', result_json=?, updated_at=? WHERE job_id=?",
            (json.dumps(result), now, job.job_id))

    def fail(self, job: BackfillJob, *, error: str, now: str) -> None:
        self._conn.execute(
            "UPDATE backfill_jobs SET status='failed', error=?, updated_at=? WHERE job_id=?",
            (error, now, job.job_id))

    def reset_stale(self, *, now: str) -> int:
        """Requeue running jobs (crash recovery). V1: unconditional (single-worker assumption);
        prod adds a lease timeout. MUST be called at worker/daemon startup."""
        cur = self._conn.execute(
            "UPDATE backfill_jobs SET status='pending', updated_at=? WHERE status='running'",
            (now,))
        return cur.rowcount

    def count_by_status(self, status: str) -> int:
        return self._conn.execute(
            "SELECT COUNT(*) FROM backfill_jobs WHERE status=?", (status,)).fetchone()[0]

    def close(self) -> None:
        self._conn.close()
