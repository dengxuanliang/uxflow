import pathlib

from module1.pipeline import PipelineConfig, TrajectoryPipeline
from module3.compose import GeneralDataConfig
from module3.pipeline import select_final_dataset
from module3.selection import SelectionConfig

FIXTURES = pathlib.Path(__file__).parent.parent.parent / "fixtures"


class FakeGateway:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    async def call(self, messages, model, **kwargs):
        self.calls.append((messages, model))
        if self._responses:
            resp = self._responses.pop(0)
        else:
            resp = (
                '[{"match": false, "confidence": 0.1, "spans": [], '
                '"reasoning": "default"}]'
            )
        return resp, {"status_code": 200, "prompt_tokens": 10, "completion_tokens": 5}


async def test_module_2_3_chain_produces_dataset():
    spec = {
        "raw_input": "写入py文件有语法错误",
        "domain": "agentic_swe",
        "sub_problems": [
            {
                "id": "p1",
                "origin": "original",
                "parent_id": None,
                "raw_text": "x",
                "failure_summary": "y",
                "target_capability": ["valid_syntax_in_toolcall"],
                "trajectory_signal": "SyntaxError",
                "hyde_positive": ["a", "b"],
                "keywords": ["SyntaxError", "python"],
                "structured_filters": {"languages": ["python"]},
                "confidence": 0.9,
                "route": "pass",
            }
        ],
    }
    judge_ok = (
        '[{"match": true, "confidence": 0.9, '
        '"spans": [{"start_step": 0, "end_step": 3}], "reasoning": "ok"}]'
    )
    gw = FakeGateway([judge_ok] * 50)
    cfg = PipelineConfig(judge_model="test-model", recall_top_n=10)
    pipeline = TrajectoryPipeline(config=cfg, gateway=gw)

    scored = await pipeline.run_scored(
        trajectory_paths=[FIXTURES / "trajectories" / "sample_01.jsonl"],
        problem_specs=[spec],
    )
    assert scored

    result = select_final_dataset(
        scored,
        sub_problem_ids=["p1"],
        selection=SelectionConfig(n=3, min_per_problem=1),
        general=GeneralDataConfig(ratio=0.3, source_path=None),
    )
    assert result["manifest"]["targeted_count"] >= 1
    assert result["manifest"]["general_count"] == 0
    assert all(c.bm25_tokens for c in result["targeted"])
    assert all(hasattr(c, "loss_mask_spans") for c in result["targeted"])
