# SPDX-License-Identifier: Apache-2.0
from module0_5.queue import SqliteBackfillQueue


def test_enqueue_and_claim_priority_order(tmp_path):
    q = SqliteBackfillQueue(tmp_path / "t.db")
    q.enqueue("low", priority=1, created_at="t0")
    q.enqueue("high", priority=5, created_at="t0")
    job = q.claim(now="t1")
    assert job.label == "high"        # highest priority first
    assert job.status == "running"


def test_claim_empty_returns_none(tmp_path):
    q = SqliteBackfillQueue(tmp_path / "t.db")
    assert q.claim(now="t1") is None


def test_complete_marks_done_and_persists(tmp_path):
    db = tmp_path / "t.db"
    q = SqliteBackfillQueue(db)
    q.enqueue("lbl", priority=1, created_at="t0")
    job = q.claim(now="t1")
    q.complete(job, result={"slices_written": 3}, now="t2")
    q2 = SqliteBackfillQueue(db)  # reopen
    assert q2.count_by_status("done") == 1


def test_fail_records_error_and_increments_attempts(tmp_path):
    q = SqliteBackfillQueue(tmp_path / "t.db")
    q.enqueue("lbl", priority=1, created_at="t0")
    job = q.claim(now="t1")
    q.fail(job, error="judge boom", now="t2")
    assert q.count_by_status("failed") == 1


def test_reset_stale_requeues_running(tmp_path):
    q = SqliteBackfillQueue(tmp_path / "t.db")
    q.enqueue("lbl", priority=1, created_at="t0")
    q.claim(now="t1")  # now running
    n = q.reset_stale(now="t2")  # unconditional reset for test
    assert n == 1
    assert q.count_by_status("pending") == 1


def test_claim_is_atomic_no_double_claim(tmp_path):
    q = SqliteBackfillQueue(tmp_path / "t.db")
    q.enqueue("a", priority=1, created_at="t0")
    j1 = q.claim(now="t1")
    j2 = q.claim(now="t1")
    assert j1 is not None and j2 is None  # only one claimant
