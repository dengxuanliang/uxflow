# SPDX-License-Identifier: Apache-2.0
"""Tests for the write path (ingest_trajectories) and cached read path (search)."""

import pytest

from module1.pipeline import TrajectoryPipeline, PipelineConfig
from module1.sqlite_store import SqliteSliceStore


class FakeGateway:
    """Mock gateway recording every judge call in .calls."""
    def __init__(self, responses=None):
        self._responses = list(responses or [])
        self.calls = []

    async def call(self, messages, model, **kwargs):
        self.calls.append((messages, model))
        if self._responses:
            resp = self._responses.pop(0)
        else:
            resp = '[{"match": false, "confidence": 0.1, "spans": [], "reasoning": "default"}]'
        return resp, {"status_code": 200, "prompt_tokens": 100, "completion_tokens": 50}


class FakeJudgeCache:
    """In-memory JudgeCache recording get/put calls."""
    def __init__(self):
        self._store = {}
        self.get_calls = []
        self.put_calls = []

    def get(self, sub_problem_id, trajectory_id, slice_index):
        key = (sub_problem_id, trajectory_id, slice_index)
        self.get_calls.append(key)
        return self._store.get(key)

    def put(self, sub_problem_id, trajectory_id, slice_index, verdict):
        key = (sub_problem_id, trajectory_id, slice_index)
        self.put_calls.append((key, verdict))
        self._store[key] = verdict


@pytest.fixture
def sub_problem():
    """A sub_problem whose keywords recall all fixture slices."""
    return {
        "id": "p1",
        "target_capability": ["valid_syntax_in_toolcall"],
        "trajectory_signal": "s",
        "hyde_positive": [],
        "keywords": ["python", "SyntaxError", "import", "tool"],
        "structured_filters": {},
    }


@pytest.fixture
def spec(sub_problem):
    return {"raw_input": "q", "sub_problems": [sub_problem]}


def _match_resp(conf):
    return f'[{{"match": true, "confidence": {conf}, "spans": [], "reasoning": "ok"}}]'


def _cfg(**kw):
    """PipelineConfig with batch_size=1 so each 1-element response maps to one slice."""
    kw.setdefault("judge_model", "t")
    kw.setdefault("judge_batch_size", 1)
    return PipelineConfig(**kw)


# ---------------------------------------------------------------------------
# ingest_trajectories
# ---------------------------------------------------------------------------

def test_ingest_incremental_no_reset(trajectories_path, tmp_path):
    """Ingesting the same file twice upserts by key — size stays constant (no reset)."""
    db = tmp_path / "slices.db"
    p = TrajectoryPipeline(config=_cfg(), gateway=FakeGateway(),
                           store_factory=lambda: SqliteSliceStore(db))

    p.ingest_trajectories([trajectories_path])
    size_after_first = p._store.size
    assert size_after_first > 0

    p.ingest_trajectories([trajectories_path])
    size_after_second = p._store.size

    # Same keys re-added → upsert, no doubling; store was not reset.
    assert size_after_second == size_after_first


def test_ingest_on_trajectory_callback(trajectories_path, sample_trajectories):
    """on_trajectory is invoked once per trajectory with (traj, source_path)."""
    p = TrajectoryPipeline(config=_cfg(), gateway=FakeGateway())
    seen = []

    p.ingest_trajectories([trajectories_path],
                          on_trajectory=lambda traj, src: seen.append((traj, src)))

    assert len(seen) == len(sample_trajectories)
    for traj, src in seen:
        assert hasattr(traj, "id")
        assert src == str(trajectories_path)
    # Every trajectory id from the fixture is represented.
    seen_ids = {traj.id for traj, _ in seen}
    assert seen_ids == {t["id"] for t in sample_trajectories}


# ---------------------------------------------------------------------------
# search — cache behavior
# ---------------------------------------------------------------------------

async def test_search_all_miss(trajectories_path, spec):
    """Empty cache → judge runs, cache.put happens, results non-empty."""
    gw = FakeGateway([_match_resp(0.9)] * 20)
    p = TrajectoryPipeline(config=_cfg(recall_top_n=20), gateway=gw)
    p.ingest_trajectories([trajectories_path])

    cache = FakeJudgeCache()
    scored = await p.search(problem_specs=[spec], judge_cache=cache)

    assert scored
    assert len(gw.calls) >= 1          # judge_batch was actually called
    assert len(cache.put_calls) >= 1   # every miss was written back


async def test_search_all_cached_zero_judge(trajectories_path, spec):
    """Fully pre-filled cache → judge_batch never called (gateway.calls == 0)."""
    # First run to discover which (sp_id, traj, slice) get recalled + fill the cache.
    warm_gw = FakeGateway([_match_resp(0.77)] * 20)
    p = TrajectoryPipeline(config=_cfg(recall_top_n=20), gateway=warm_gw)
    p.ingest_trajectories([trajectories_path])
    cache = FakeJudgeCache()
    first = await p.search(problem_specs=[spec], judge_cache=cache)
    assert first

    # Second run with a fresh zero-call gateway and the now-warm cache.
    cold_gw = FakeGateway()
    p2 = TrajectoryPipeline(config=_cfg(recall_top_n=20), gateway=cold_gw)
    p2.ingest_trajectories([trajectories_path])
    second = await p2.search(problem_specs=[spec], judge_cache=cache)

    assert len(cold_gw.calls) == 0  # no judge at all
    # Cached confidences flow through unchanged.
    assert second
    assert all(sc.judge_confidence == 0.77 for sc in second)


async def test_search_order_preserved_mixed_cache(trajectories_path, spec, sub_problem):
    """Mixed cached/miss: each ScoredCandidate's judge_confidence matches its own
    (traj, slice) source — proving cached and miss verdicts are not swapped."""
    sp_id = sub_problem["id"]

    # Discover the exact recall set (keys) via a throwaway search.
    probe_gw = FakeGateway([_match_resp(0.5)] * 20)
    probe = TrajectoryPipeline(config=_cfg(recall_top_n=20), gateway=probe_gw)
    probe.ingest_trajectories([trajectories_path])
    probe_cache = FakeJudgeCache()
    probe_scored = await probe.search(problem_specs=[spec], judge_cache=probe_cache)
    keys = [(sc.trajectory_id, sc.slice_index) for sc in probe_scored]
    assert len(keys) >= 4  # need enough to split cached vs miss meaningfully

    # Pre-fill cache for HALF the keys with a distinctive CACHED sentinel confidence.
    # The other half stay miss and get a distinct MISS sentinel from the gateway.
    CACHED_CONF = 0.11
    MISS_CONF = 0.99
    cached_keys = set(keys[::2])   # every other key is cached
    miss_keys = [k for k in keys if k not in cached_keys]

    cache = FakeJudgeCache()
    for (tid, sidx) in cached_keys:
        cache.put(sp_id, tid, sidx,
                  {"match": True, "confidence": CACHED_CONF, "spans": []})

    # Gateway returns MISS_CONF for every judged (missed) slice.
    gw = FakeGateway([_match_resp(MISS_CONF)] * 20)
    p = TrajectoryPipeline(config=_cfg(recall_top_n=20), gateway=gw)
    p.ingest_trajectories([trajectories_path])

    scored = await p.search(problem_specs=[spec], judge_cache=cache)

    # judge ran only for the misses.
    assert len(gw.calls) >= 1
    by_key = {(sc.trajectory_id, sc.slice_index): sc for sc in scored}
    for key in keys:
        expected = CACHED_CONF if key in cached_keys else MISS_CONF
        assert by_key[key].judge_confidence == pytest.approx(expected), (
            f"{key} got {by_key[key].judge_confidence}, expected {expected} "
            f"(cached={key in cached_keys})"
        )
    # Sanity: the miss slices were written back to cache.
    written = {kv[0][1:] for kv in cache.put_calls}
    for k in miss_keys:
        assert k in written


async def test_search_on_progress(trajectories_path, sub_problem):
    """on_progress called once per sub_problem, done increments to total."""
    spec_multi = {
        "raw_input": "q",
        "sub_problems": [dict(sub_problem, id="p1"), dict(sub_problem, id="p2")],
    }
    gw = FakeGateway([_match_resp(0.9)] * 40)
    p = TrajectoryPipeline(config=_cfg(recall_top_n=20), gateway=gw)
    p.ingest_trajectories([trajectories_path])

    progress = []
    cache = FakeJudgeCache()
    await p.search(problem_specs=[spec_multi], judge_cache=cache,
                   on_progress=lambda done, total: progress.append((done, total)))

    assert progress == [(1, 2), (2, 2)]
