import json
import pathlib
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
                    "min_turns": None,
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
    assert pipeline._index.size == len(pipeline._slice_map)


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
