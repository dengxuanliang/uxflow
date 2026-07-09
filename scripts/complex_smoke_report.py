"""Run a real module1 -> module2 -> module3 smoke report on 20 trajectories.

This script uses the local Qwen embedding model and the configured LLM gateway
for judge calls. It is intentionally a manual smoke report, not a pytest test:
LLM outputs can vary, while the generated report records the actual result.

Usage:
    .venv/bin/python scripts/complex_smoke_report.py
"""

from __future__ import annotations

import asyncio
import json
import os
import pathlib

from dotenv import load_dotenv

from llm_gateway import GatewayConfig, LLMGateway
from module0.embedding import EmbeddingModel
from module1.pipeline import PipelineConfig, TrajectoryPipeline
from module3.compose import GeneralDataConfig
from module3.pipeline import select_final_dataset
from module3.selection import SelectionConfig

ROOT = pathlib.Path(__file__).resolve().parents[1]
FIXTURE_PATH = ROOT / "fixtures" / "trajectories" / "smoke_20.jsonl"
REAL_REPORT_PATH = (
    ROOT / "docs" / "superpowers" / "reports" / "2026-07-09-real-complex-smoke-2-3.md"
)
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
    "smoke_001", "smoke_002", "smoke_003", "smoke_004", "smoke_005",
    "smoke_006", "smoke_007", "smoke_016", "smoke_017", "smoke_019",
}
EXPECTED_NEAR_DUPLICATE = {"smoke_007"}
EXPECTED_SELECTED = {
    "smoke_001", "smoke_002", "smoke_003", "smoke_004", "smoke_005",
    "smoke_006", "smoke_016", "smoke_017", "smoke_019",
}


async def main() -> None:
    load_dotenv(ROOT / ".env")
    model = os.environ.get("MODULE1_JUDGE_MODEL") or os.environ.get(
        "MODULE0_TEST_MODEL", "gpt-4o-mini"
    )
    report_path = pathlib.Path(os.environ.get("COMPLEX_SMOKE_REPORT", REAL_REPORT_PATH))

    print("Loading local Qwen embedding model...")
    embedding_model = EmbeddingModel()
    print(f"Embedding model ready: dim={embedding_model.dimension}")

    config = GatewayConfig(
        litellm_base=os.environ.get("LITELLM_BASE", "http://localhost:4000/v1"),
        litellm_key=os.environ.get("LITELLM_KEY", ""),
        transport_stuck_seconds=0,
    )

    async with LLMGateway(config) as gateway:
        pipeline = TrajectoryPipeline(
            config=PipelineConfig(
                judge_model=model,
                recall_top_n=20,
                embedding_model=embedding_model,
            ),
            gateway=gateway,
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

        report = _render_report(
            scored=scored,
            selected=result["targeted"],
            manifest=result["manifest"],
            model=model,
            gateway_stats=gateway.http_stats,
        )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report)
    print(f"Report written: {report_path}")
    print(report)


def _render_report(*, scored, selected, manifest, model: str, gateway_stats: dict) -> str:
    selected_ids = {c.trajectory_id for c in selected}
    matched_ids = {c.trajectory_id for c in scored if c.judge_match}
    decayed_ids = {c.trajectory_id for c in scored if not c.judge_match}
    selected_expected = selected_ids & EXPECTED_SELECTED
    missed_expected = EXPECTED_SELECTED - selected_ids
    unexpected_selected = selected_ids - EXPECTED_SELECTED

    lines = [
        "# Real Complex Smoke Report: Modules 1 -> 2 -> 3",
        "",
        "## Scenario",
        f"- Fixture: `{FIXTURE_PATH.relative_to(ROOT)}`",
        "- Trajectories: `20`",
        "- Query: 修复 Python SyntaxError，并通过测试验证修复",
        f"- Judge model: `{model}`",
        "- Embedding: local `Qwen/Qwen3-Embedding-0.6B` via `EmbeddingModel`",
        "- Selection: `SelectionConfig(n=12, min_per_problem=1)`",
        "- General data: disabled (`source_path=None`)",
        "",
        "## Ground Truth Used For Interpretation",
        f"- Expected positives: `{', '.join(sorted(EXPECTED_POSITIVE))}`",
        f"- Expected near duplicate: `{', '.join(sorted(EXPECTED_NEAR_DUPLICATE))}`",
        f"- Expected selected after dedup: `{', '.join(sorted(EXPECTED_SELECTED))}`",
        "",
        "## Actual Summary",
        f"- Scored candidates: `{len(scored)}`",
        f"- Judge matched ids: `{', '.join(sorted(matched_ids)) or 'none'}`",
        f"- Judge decayed ids: `{', '.join(sorted(decayed_ids)) or 'none'}`",
        f"- Selected ids: `{', '.join(sorted(selected_ids)) or 'none'}`",
        f"- Selected expected ids: `{', '.join(sorted(selected_expected)) or 'none'}`",
        f"- Missed expected ids: `{', '.join(sorted(missed_expected)) or 'none'}`",
        f"- Unexpected selected ids: `{', '.join(sorted(unexpected_selected)) or 'none'}`",
        f"- Manifest: `{manifest}`",
        f"- Gateway stats: `{gateway_stats}`",
        "",
        "## Candidate Table",
        "| id | relevance | judge_match | confidence | spans | tokens | selected |",
        "|---|---:|---|---:|---|---|---|",
    ]

    for c in scored:
        lines.append(
            "| "
            f"{c.trajectory_id} | {c.relevance_score:.4f} | {c.judge_match} | "
            f"{c.judge_confidence:.2f} | {c.loss_mask_spans} | "
            f"{', '.join(c.bm25_tokens[:10])} | "
            f"{'yes' if c.trajectory_id in selected_ids else ''} |"
        )

    lines.extend(
        [
            "",
            "## How To Read This",
            "- Module 1 performs real slicing, signature extraction, BM25/vector recall, and LLM judge calls.",
            "- Module 2 applies soft scoring: judge matches keep full RRF score; misses are decayed.",
            "- Module 3 filters non-trainable misses, deduplicates near duplicates, selects a coverage/diversity set, and leaves general data empty because no source is configured.",
            "- Differences from ground truth are expected to reflect real judge behavior, not deterministic test failure.",
        ]
    )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    asyncio.run(main())
