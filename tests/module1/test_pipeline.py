import pytest
from module1.pipeline import TrajectoryPipeline, PipelineConfig
from module1.models import SFTCandidate


class FakeGateway:
    """Mock gateway for pipeline integration test."""
    def __init__(self, responses=None):
        self._responses = list(responses or [])
        self.calls = []

    async def call(self, messages, model, **kwargs):
        self.calls.append((messages, model))
        if self._responses:
            resp = self._responses.pop(0)
        else:
            # Default: no match
            resp = '[{"match": false, "confidence": 0.1, "spans": [], "reasoning": "default"}]'
        return resp, {"status_code": 200, "prompt_tokens": 100, "completion_tokens": 50}


@pytest.fixture
def problem_spec_dict():
    """A minimal ProblemSpec dict for testing."""
    return {
        "raw_input": "写入py文件有语法错误",
        "domain": "agentic_swe",
        "sub_problems": [
            {
                "id": "p1",
                "origin": "original",
                "parent_id": None,
                "raw_text": "写入py文件有语法错误",
                "failure_summary": "写入 py 文件时产生语法错误",
                "target_capability": ["valid_syntax_in_toolcall"],
                "trajectory_signal": "observation 含 SyntaxError 且前序 tool_call 含 python 代码写入",
                "hyde_positive": [
                    "假设: 工具正确写入 python 文件，无语法错误，执行结果 exit code 0",
                    "假设: python 文件包含合法 import 和函数定义，lint 通过",
                ],
                "keywords": ["SyntaxError", "python", "import"],
                "structured_filters": {
                    "languages": ["python"],
                    "tools_used": ["Write", "Edit"],
                    "has_verification_step": None,
                },
                "confidence": 0.92,
                "route": "pass",
            }
        ],
    }


def test_pipeline_config_defaults():
    cfg = PipelineConfig()
    assert cfg.judge_model is not None
    assert cfg.judge_batch_size == 3
    assert cfg.recall_top_n == 20


async def test_pipeline_end_to_end(trajectories_path, problem_spec_dict):
    """Full pipeline: load → slice → sign → index → recall → judge → output."""
    # Judge will match the first slice
    judge_resp = '''[{"match": true, "confidence": 0.88, "spans": [{"start_step": 0, "end_step": 3}], "reasoning": "correct python write"}]'''
    gw = FakeGateway([judge_resp] * 10)

    cfg = PipelineConfig(judge_model="test-model", recall_top_n=5, judge_batch_size=3)
    pipeline = TrajectoryPipeline(config=cfg, gateway=gw)

    candidates = await pipeline.run(
        trajectory_paths=[trajectories_path],
        problem_specs=[problem_spec_dict],
    )

    # Judge mock returns match=true → must produce at least one candidate with spans
    assert len(candidates) >= 1
    assert all(c.matched_problems for c in candidates)
    assert any(
        mp["loss_mask_spans"]
        for c in candidates for mp in c.matched_problems
    )
    assert len(gw.calls) >= 1


async def test_pipeline_no_match(trajectories_path, problem_spec_dict):
    """When judge says no match, no SFTCandidate is produced."""
    judge_resp = '[{"match": false, "confidence": 0.1, "spans": [], "reasoning": "not relevant"}]'
    gw = FakeGateway([judge_resp] * 10)  # enough for any number of recall results

    cfg = PipelineConfig(judge_model="test-model", recall_top_n=5)
    pipeline = TrajectoryPipeline(config=cfg, gateway=gw)

    candidates = await pipeline.run(
        trajectory_paths=[trajectories_path],
        problem_specs=[problem_spec_dict],
    )

    # All judge calls returned no-match → zero candidates
    assert candidates == []


async def test_pipeline_rerun_no_accumulation(trajectories_path, problem_spec_dict):
    """Reusing a pipeline instance across runs must not accumulate index state."""
    judge_resp = '[{"match": true, "confidence": 0.9, "spans": [{"start_step": 0, "end_step": 3}], "reasoning": "ok"}]'
    gw = FakeGateway([judge_resp] * 50)
    cfg = PipelineConfig(judge_model="test-model", recall_top_n=5)
    pipeline = TrajectoryPipeline(config=cfg, gateway=gw)

    first = await pipeline.run(trajectory_paths=[trajectories_path], problem_specs=[problem_spec_dict])
    second = await pipeline.run(trajectory_paths=[trajectories_path], problem_specs=[problem_spec_dict])

    assert len(second) == len(first)
    assert pipeline._store.size == pipeline._store.slice_source_count


async def test_pipeline_multiple_specs(trajectories_path):
    """Pipeline handles multiple ProblemSpecs independently."""
    spec1 = {
        "raw_input": "问题1",
        "domain": "agentic_swe",
        "sub_problems": [{
            "id": "p1", "origin": "original", "parent_id": None,
            "raw_text": "a", "failure_summary": "a",
            "target_capability": ["valid_syntax_in_toolcall"],
            "trajectory_signal": "s",
            "hyde_positive": ["h1", "h2"],
            "keywords": ["python"],
            "structured_filters": {"languages": ["python"]},
            "confidence": 0.9, "route": "pass",
        }],
    }
    spec2 = {
        "raw_input": "问题2",
        "domain": "agentic_swe",
        "sub_problems": [{
            "id": "p2", "origin": "original", "parent_id": None,
            "raw_text": "b", "failure_summary": "b",
            "target_capability": ["wellformed_tool_call"],
            "trajectory_signal": "s",
            "hyde_positive": ["h1", "h2"],
            "keywords": ["tool_call", "JSON"],
            "structured_filters": {},
            "confidence": 0.88, "route": "pass",
        }],
    }

    gw = FakeGateway([
        '[{"match": true, "confidence": 0.9, "spans": [{"start_step": 0, "end_step": 3}], "reasoning": "ok"}]',
        '[{"match": false, "confidence": 0.2, "spans": [], "reasoning": "no"}]',
    ] * 5)

    cfg = PipelineConfig(judge_model="test-model", recall_top_n=3)
    pipeline = TrajectoryPipeline(config=cfg, gateway=gw)

    candidates = await pipeline.run(
        trajectory_paths=[trajectories_path],
        problem_specs=[spec1, spec2],
    )

    # Pipeline ran for both specs without error
    assert isinstance(candidates, list)


async def test_pipeline_empty_trajectories(tmp_path, problem_spec_dict):
    """Empty trajectory file → no candidates, no crash."""
    empty = tmp_path / "empty.jsonl"
    empty.write_text("")

    gw = FakeGateway([])
    cfg = PipelineConfig(judge_model="test-model")
    pipeline = TrajectoryPipeline(config=cfg, gateway=gw)

    candidates = await pipeline.run(
        trajectory_paths=[empty],
        problem_specs=[problem_spec_dict],
    )
    assert candidates == []


async def test_pipeline_output_format(trajectories_path, problem_spec_dict):
    """SFTCandidate output matches expected schema."""
    judge_resp = '[{"match": true, "confidence": 0.92, "spans": [{"start_step": 0, "end_step": 4}], "reasoning": "good"}]'
    gw = FakeGateway([judge_resp] * 10)

    cfg = PipelineConfig(judge_model="test-model", recall_top_n=5)
    pipeline = TrajectoryPipeline(config=cfg, gateway=gw)

    candidates = await pipeline.run(
        trajectory_paths=[trajectories_path],
        problem_specs=[problem_spec_dict],
    )

    for c in candidates:
        assert isinstance(c, SFTCandidate)
        assert c.trajectory_id is not None
        assert c.trajectory_path is not None
        for mp in c.matched_problems:
            assert "problem_spec_id" in mp or "sub_problem_id" in mp
            assert "capability" in mp
            assert "confidence" in mp
            assert "loss_mask_spans" in mp


async def test_run_scored_soft_scoring(trajectories_path, problem_spec_dict):
    """run_scored keeps recalled slices as scored slice-level candidates."""
    from module2.models import ScoredCandidate

    judge_resp = '[{"match": true, "confidence": 0.9, "spans": [{"start_step": 0, "end_step": 3}], "reasoning": "ok"}]'
    gw = FakeGateway([judge_resp] * 50)
    cfg = PipelineConfig(judge_model="test-model", recall_top_n=5)
    pipeline = TrajectoryPipeline(config=cfg, gateway=gw)

    scored = await pipeline.run_scored(
        trajectory_paths=[trajectories_path],
        problem_specs=[problem_spec_dict],
    )
    assert all(isinstance(s, ScoredCandidate) for s in scored)
    assert scored
    assert all(s.relevance_score >= 0.0 for s in scored)
    assert scored == sorted(scored, key=lambda s: s.relevance_score, reverse=True)


# ── PR-3: rubric 透传到 4 处 judge_batch 调用点 ──

_RUBRIC = {
    "positive_criteria": ["写出可运行代码"],
    "negative_criteria": ["产出 SyntaxError"],
    "decisive_evidence": "工具调用后 observation 无 SyntaxError 的步",
    "capability_kind": "avoidance",
}


class SpyJudge:
    """记录每次 judge_batch 收到的 rubric，返回带 evidence_step/criteria_hit 的结果。"""
    def __init__(self):
        self.rubrics = []

    async def judge_batch(self, *, slices, target_capability, trajectory_signal, rubric=None):
        from module1.models import JudgeResult
        self.rubrics.append(rubric)
        return [
            JudgeResult(match=True, confidence=0.9, spans=[{"start_step": 0, "end_step": 1}],
                        evidence_step=1, criteria_hit=["写出可运行代码"])
            for _ in slices
        ]


class FakeCache:
    """内存 judge_cache，用于 search 路径测试。"""
    def __init__(self):
        self.store = {}
        self.puts = []

    def get(self, sp_id, traj_id, slice_index):
        return self.store.get((sp_id, traj_id, slice_index))

    def put(self, sp_id, traj_id, slice_index, verdict):
        self.puts.append(verdict)
        self.store[(sp_id, traj_id, slice_index)] = verdict


def _spec_with_rubric(problem_spec_dict):
    spec = dict(problem_spec_dict)
    spec["sub_problems"] = [dict(sp) for sp in problem_spec_dict["sub_problems"]]
    spec["sub_problems"][0]["rubric"] = _RUBRIC
    return spec


async def test_process_sub_problem_passes_rubric(trajectories_path, problem_spec_dict):
    """legacy run 路径 (_process_sub_problem) 把 rubric 传给 judge。"""
    spec = _spec_with_rubric(problem_spec_dict)
    cfg = PipelineConfig(judge_model="test-model", recall_top_n=5)
    pipeline = TrajectoryPipeline(config=cfg, gateway=FakeGateway())
    pipeline._build_index([trajectories_path])
    spy = SpyJudge()
    pipeline._judge = spy
    await pipeline._process_sub_problem(spec["sub_problems"][0], "spec")
    assert spy.rubrics and all(r == _RUBRIC for r in spy.rubrics)


async def test_score_sub_problem_passes_rubric(trajectories_path, problem_spec_dict):
    """run_scored 生产路径 (_score_sub_problem) 把 rubric 传给 judge。"""
    spec = _spec_with_rubric(problem_spec_dict)
    cfg = PipelineConfig(judge_model="test-model", recall_top_n=5)
    pipeline = TrajectoryPipeline(config=cfg, gateway=FakeGateway())
    pipeline._build_index([trajectories_path])
    spy = SpyJudge()
    pipeline._judge = spy
    scored = await pipeline._score_sub_problem(spec["sub_problems"][0])
    assert spy.rubrics and all(r == _RUBRIC for r in spy.rubrics)
    # 传播链：rerank 把 evidence_step/criteria_hit 搬进 ScoredCandidate
    assert scored
    assert all(sc.evidence_step == 1 for sc in scored)
    assert all(sc.criteria_hit == ["写出可运行代码"] for sc in scored)


async def test_score_sub_problem_cached_search_passes_rubric(trajectories_path, problem_spec_dict):
    """search 生产路径 (_score_sub_problem_cached，最易漏) 把 rubric 传给 judge，
    且 judge_cache.put 带上 evidence_step/criteria_hit。"""
    spec = _spec_with_rubric(problem_spec_dict)
    cfg = PipelineConfig(judge_model="test-model", recall_top_n=5)
    pipeline = TrajectoryPipeline(config=cfg, gateway=FakeGateway())
    pipeline._build_index([trajectories_path])
    spy = SpyJudge()
    pipeline._judge = spy
    cache = FakeCache()
    scored = await pipeline._score_sub_problem_cached(spec["sub_problems"][0], cache)
    assert spy.rubrics and all(r == _RUBRIC for r in spy.rubrics)
    # judge_cache.put 收到的 verdict 带新字段
    assert cache.puts
    assert all(v["evidence_step"] == 1 for v in cache.puts)
    assert all(v["criteria_hit"] == ["写出可运行代码"] for v in cache.puts)
    # 传播链同样在 cached 路径成立
    assert scored and all(sc.evidence_step == 1 for sc in scored)


def test_dict_to_judge_result_reads_new_fields():
    """_dict_to_judge_result 从 cache dict 补读 evidence_step/criteria_hit。"""
    from module1.pipeline import _dict_to_judge_result
    jr = _dict_to_judge_result({
        "match": True, "confidence": 0.8, "spans": [],
        "evidence_step": 4, "criteria_hit": ["c1"],
    })
    assert jr.evidence_step == 4
    assert jr.criteria_hit == ["c1"]


def test_dict_to_judge_result_missing_new_fields_degrade():
    """旧 cache dict（无新字段）→ 降级 None/[]，不崩（向后兼容）。"""
    from module1.pipeline import _dict_to_judge_result
    jr = _dict_to_judge_result({"match": True, "confidence": 0.8, "spans": []})
    assert jr.evidence_step is None
    assert jr.criteria_hit == []
