# SPDX-License-Identifier: Apache-2.0
from service.stores import (
    JudgeCache,
    ProblemStore,
    SqliteJudgeCache,
    SqliteProblemStore,
    SqliteTrajectoryStore,
    TrajectoryStore,
)


# ---------------- ProblemStore ----------------

def test_problem_add_get_roundtrip(tmp_path):
    store = SqliteProblemStore(tmp_path / "p.db")
    pid = store.add("What is Bash?", [1.0, 0.0], {"k": "v"})
    got = store.get(pid)
    assert got["problem_id"] == pid
    assert got["raw_question"] == "What is Bash?"
    assert got["spec"] == {"k": "v"}


def test_problem_get_missing_returns_none(tmp_path):
    store = SqliteProblemStore(tmp_path / "p.db")
    assert store.get("deadbeef") is None


def test_problem_nearest_empty_returns_none(tmp_path):
    store = SqliteProblemStore(tmp_path / "p.db")
    assert store.nearest([1.0, 0.0]) is None


def test_problem_nearest_hit_returns_best(tmp_path):
    store = SqliteProblemStore(tmp_path / "p.db")
    pid_a = store.add("alpha", [1.0, 0.0], {"name": "a"})
    store.add("beta", [0.0, 1.0], {"name": "b"})
    result = store.nearest([0.9, 0.1])
    assert result is not None
    pid, score, spec = result
    assert pid == pid_a
    assert spec == {"name": "a"}
    assert score > 0.9


def test_problem_nearest_zero_vector_returns_none(tmp_path):
    store = SqliteProblemStore(tmp_path / "p.db")
    store.add("alpha", [1.0, 0.0], {"name": "a"})
    assert store.nearest([0.0, 0.0]) is None


def test_problem_duplicate_add_same_pid_count_one(tmp_path):
    store = SqliteProblemStore(tmp_path / "p.db")
    p1 = store.add("same question", [1.0, 0.0], {"v": 1})
    p2 = store.add("same question", [1.0, 0.0], {"v": 2})
    assert p1 == p2
    assert store.count() == 1
    assert store.get(p1)["spec"] == {"v": 2}


def test_problem_reopen_persistence(tmp_path):
    db = tmp_path / "p.db"
    s1 = SqliteProblemStore(db)
    pid = s1.add("alpha", [1.0, 0.0], {"name": "a"})
    s1.add("beta", [0.0, 1.0], {"name": "b"})
    s1.close()
    s2 = SqliteProblemStore(db)
    assert s2.count() == 2
    assert s2.get(pid)["spec"] == {"name": "a"}
    result = s2.nearest([0.95, 0.05])
    assert result is not None and result[0] == pid


def test_problem_store_satisfies_protocol(tmp_path):
    store = SqliteProblemStore(tmp_path / "p.db")
    assert isinstance(store, ProblemStore)


# ---------------- TrajectoryStore ----------------

def test_trajectory_upsert_get_roundtrip(tmp_path):
    store = SqliteTrajectoryStore(tmp_path / "t.db")
    steps = [{"index": 0, "role": "user", "content": "hi"}, {"index": 1, "role": "assistant", "content": "yo"}]
    store.upsert("T1", steps, source_path="/x/y.json")
    got = store.get("T1")
    assert got["trajectory_id"] == "T1"
    assert got["steps"] == steps


def test_trajectory_get_missing_returns_none(tmp_path):
    store = SqliteTrajectoryStore(tmp_path / "t.db")
    assert store.get("nope") is None


def test_trajectory_upsert_idempotent(tmp_path):
    store = SqliteTrajectoryStore(tmp_path / "t.db")
    store.upsert("T1", [{"a": 1}], source_path="/p1")
    store.upsert("T1", [{"a": 2}], source_path="/p2")
    assert store.count() == 1
    assert store.get("T1")["steps"] == [{"a": 2}]


def test_trajectory_reopen_persistence(tmp_path):
    db = tmp_path / "t.db"
    s1 = SqliteTrajectoryStore(db)
    s1.upsert("T1", [{"index": 0, "role": "user", "content": "hi"}], source_path="/p")
    s1.close()
    s2 = SqliteTrajectoryStore(db)
    assert s2.count() == 1
    assert s2.get("T1")["steps"][0]["content"] == "hi"


def test_trajectory_store_satisfies_protocol(tmp_path):
    store = SqliteTrajectoryStore(tmp_path / "t.db")
    assert isinstance(store, TrajectoryStore)


# ---------------- JudgeCache ----------------

def test_judge_put_get_roundtrip(tmp_path):
    cache = SqliteJudgeCache(tmp_path / "j.db")
    verdict = {"match": True, "confidence": 0.87, "spans": [{"start": 0, "end": 3}]}
    cache.put("sp1", "T1", 0, verdict)
    got = cache.get("sp1", "T1", 0)
    assert got["match"] is True
    assert got["confidence"] == 0.87
    assert got["spans"] == [{"start": 0, "end": 3}]


def test_judge_get_miss_returns_none(tmp_path):
    cache = SqliteJudgeCache(tmp_path / "j.db")
    assert cache.get("sp1", "T1", 0) is None


def test_judge_match_false_roundtrips(tmp_path):
    cache = SqliteJudgeCache(tmp_path / "j.db")
    cache.put("sp1", "T1", 0, {"match": False, "confidence": 0.1, "spans": []})
    got = cache.get("sp1", "T1", 0)
    assert got["match"] is False
    assert got["spans"] == []


def test_judge_pk_distinguishes_slice_index(tmp_path):
    cache = SqliteJudgeCache(tmp_path / "j.db")
    cache.put("sp1", "T1", 0, {"match": True, "confidence": 0.9, "spans": [{"s": 1}]})
    cache.put("sp1", "T1", 1, {"match": False, "confidence": 0.2, "spans": []})
    a = cache.get("sp1", "T1", 0)
    b = cache.get("sp1", "T1", 1)
    assert a["match"] is True and b["match"] is False


def test_judge_reopen_persistence(tmp_path):
    db = tmp_path / "j.db"
    c1 = SqliteJudgeCache(db)
    c1.put("sp1", "T1", 0, {"match": True, "confidence": 0.5, "spans": [{"s": 0}]})
    c1.close()
    c2 = SqliteJudgeCache(db)
    got = c2.get("sp1", "T1", 0)
    assert got["match"] is True
    assert got["spans"] == [{"s": 0}]


def test_judge_cache_satisfies_protocol(tmp_path):
    cache = SqliteJudgeCache(tmp_path / "j.db")
    assert isinstance(cache, JudgeCache)
