from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass

from module1.pipeline import PipelineConfig, TrajectoryPipeline
from module3.compose import GeneralDataConfig
from module3.pipeline import select_final_dataset
from module3.selection import SelectionConfig

ROOT = pathlib.Path(__file__).resolve().parents[2]
FIXTURE_PATH = ROOT / "fixtures" / "trajectories" / "smoke_20.jsonl"

SPEC = {
    "raw_input": "修复 Python SyntaxError，并通过测试验证修复",
    "domain": "agentic_swe",
    "sub_problems": [
        {
            "id": "p_syntax",
            "origin": "original",
            "parent_id": None,
            "raw_text": "Fix Python SyntaxError and verify with tests",
            "failure_summary": "Python 代码存在 SyntaxError，需要正确编辑并运行测试验证",
            "target_capability": ["valid_syntax_in_toolcall"],
            "trajectory_signal": "tool result contains SyntaxError, then Edit/Write fixes Python syntax, then pytest passes",
            "hyde_positive": [],
            "keywords": ["SyntaxError", "python"],
            "structured_filters": {"languages": ["python"]},
            "confidence": 0.95,
            "route": "pass",
        }
    ],
}

EXPECTED_POSITIVE = {
    "smoke_001",
    "smoke_002",
    "smoke_003",
    "smoke_004",
    "smoke_005",
    "smoke_006",
    "smoke_007",
    "smoke_016",
    "smoke_017",
    "smoke_019",
}
EXPECTED_NEAR_DUPLICATE = {"smoke_007"}
EXPECTED_SELECTED = {
    "smoke_001",
    "smoke_002",
    "smoke_003",
    "smoke_004",
    "smoke_005",
    "smoke_006",
    "smoke_016",
    "smoke_017",
    "smoke_019",
}


@dataclass
class ComplexSmokeResult:
    trajectory_count: int
    recalled_count: int
    scored_count: int
    matched_ids: list[str]
    selected_ids: list[str]
    decayed_ids: list[str]
    manifest: dict
    report: str


class DeterministicJudgeGateway:
    """Judge by trajectory id from the prompt, keeping the rest of the chain real."""

    def __init__(self):
        self.calls = []

    async def call(self, messages, model, **kwargs):
        self.calls.append((messages, model))
        prompt = messages[-1]["content"]
        results = []
        for block in prompt.split("--- 切片 ")[1:]:
            traj_id = _extract_between(block, "trajectory=", ",")
            if traj_id in EXPECTED_POSITIVE:
                results.append(
                    {
                        "match": True,
                        "confidence": 0.94,
                        "spans": [{"start_step": 2, "end_step": 7}],
                        "reasoning": "known positive syntax-fix trajectory",
                    }
                )
            else:
                results.append(
                    {
                        "match": False,
                        "confidence": 0.15,
                        "spans": [],
                        "reasoning": "not a verified Python SyntaxError fix",
                    }
                )
        return json.dumps(results), {
            "status_code": 200,
            "prompt_tokens": 100,
            "completion_tokens": 20,
        }


class DeterministicEmbeddingModel:
    """Give each file-specific slice a stable vector; smoke_006/007 are near dupes."""

    dimension = 64

    def embed(self, text: str) -> list[float]:
        if "src/f.py" in text or "src/g.py" in text:
            bucket = 60
        else:
            bucket = 0
            for idx, path in enumerate(
                [
                    "src/a.py", "src/b.py", "src/c.py", "src/d.py", "src/e.py",
                    "src/n.py", "src/o.py", "src/q.py", "src/h.py", "src/i.py",
                    "src/j.py", "src/k.py", "src/l.py", "src/m.py", "src/p.py",
                    "src/r.py", "src/app.js",
                ],
                start=1,
            ):
                if path in text:
                    bucket = idx
                    break
        vec = [0.0] * self.dimension
        vec[bucket] = 1.0
        return vec

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(text) for text in texts]


def _extract_between(text: str, start: str, end: str) -> str:
    left = text.index(start) + len(start)
    right = text.index(end, left)
    return text[left:right]


async def run_complex_smoke() -> ComplexSmokeResult:
    gw = DeterministicJudgeGateway()
    pipeline = TrajectoryPipeline(
        config=PipelineConfig(
            judge_model="deterministic-judge",
            recall_top_n=20,
            embedding_model=DeterministicEmbeddingModel(),
        ),
        gateway=gw,
    )
    scored = await pipeline.run_scored(
        trajectory_paths=[FIXTURE_PATH],
        problem_specs=[SPEC],
    )
    result = select_final_dataset(
        scored,
        sub_problem_ids=["p_syntax"],
        selection=SelectionConfig(n=12, min_per_problem=1),
        general=GeneralDataConfig(ratio=0.3, source_path=None),
    )

    selected = result["targeted"]
    matched_ids = sorted({c.trajectory_id for c in scored if c.judge_match})
    selected_ids = [c.trajectory_id for c in selected]
    decayed_ids = sorted({c.trajectory_id for c in scored if not c.judge_match})
    report = _render_report(scored, selected, result["manifest"], gw.calls)
    return ComplexSmokeResult(
        trajectory_count=sum(1 for _ in FIXTURE_PATH.open()),
        recalled_count=len(scored),
        scored_count=len(scored),
        matched_ids=matched_ids,
        selected_ids=selected_ids,
        decayed_ids=decayed_ids,
        manifest=result["manifest"],
        report=report,
    )


def write_complex_smoke_report(result: ComplexSmokeResult, path: pathlib.Path) -> pathlib.Path:
    """Write the human-readable report to `path`.

    `path` is required on purpose: a default pointing into the repo let this
    write land in a gitignored directory, where nobody ever saw the output.
    Callers must name the destination (tests pass pytest's tmp_path).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(result.report)
    return path


def _render_report(scored, selected, manifest, calls) -> str:
    selected_ids = {c.trajectory_id for c in selected}
    lines = [
        "# Complex Smoke Report: Modules 1 -> 2 -> 3",
        "",
        "## Scenario",
        "- Fixture: `fixtures/trajectories/smoke_20.jsonl`",
        "- Query: 修复 Python SyntaxError，并通过测试验证修复",
        "- Judge: deterministic FakeGateway keyed by trajectory id",
        "- Embedding: deterministic local fake model; `smoke_006` and `smoke_007` intentionally share a vector",
        "- Selection: `SelectionConfig(n=12, min_per_problem=1)`",
        "- General data: disabled (`source_path=None`)",
        "",
        "## Ground Truth",
        f"- Positive trajectory ids: `{', '.join(sorted(EXPECTED_POSITIVE))}`",
        f"- Near duplicate expected to be removed: `{', '.join(sorted(EXPECTED_NEAR_DUPLICATE))}`",
        f"- Expected selected ids: `{', '.join(sorted(EXPECTED_SELECTED))}`",
        "",
        "## Actual Summary",
        f"- Judge calls: `{len(calls)}`",
        f"- Scored candidates: `{len(scored)}`",
        f"- Selected targeted count: `{len(selected)}`",
        f"- Manifest: `{manifest}`",
        "",
        "## Scored Candidates",
        "| id | relevance | judge_match | spans | bm25_tokens | selected |",
        "|---|---:|---|---|---|---|",
    ]
    for c in scored:
        lines.append(
            "| "
            f"{c.trajectory_id} | {c.relevance_score:.3f} | {c.judge_match} | "
            f"{c.loss_mask_spans} | {', '.join(c.bm25_tokens[:6])} | "
            f"{'yes' if c.trajectory_id in selected_ids else ''} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "- Module 1 recalled Python/SyntaxError-like slices from all 20 trajectories.",
            "- Module 2 gave known positives full score and judge misses a 0.3 decay.",
            "- Module 3 filtered non-trainable misses, deduplicated the near duplicate, and selected the remaining trainable positives.",
        ]
    )
    return "\n".join(lines) + "\n"
