"""Manual smoke test: Module 0 → LLMGateway → LiteLLM.

Usage:
    1. Fill in .env at project root (LITELLM_BASE, LITELLM_KEY, MODULE0_TEST_MODEL)
    2. python scripts/module0_smoke.py "写入py文件有语法错误，工具调用结构经常出错"
"""

import asyncio
import json
import os
import sys
import pathlib

from dotenv import load_dotenv
load_dotenv(pathlib.Path(__file__).parent.parent / ".env")

from llm_gateway import LLMGateway, GatewayConfig  # noqa: E402  (import after load_dotenv)
from module0 import QueryCompiler, Taxonomy  # noqa: E402


async def main():
    raw_input = sys.argv[1] if len(sys.argv) > 1 else "写入py文件有语法错误"
    model = os.environ.get("MODULE0_TEST_MODEL", "gpt-4o-mini")

    taxonomy_path = pathlib.Path(__file__).parent.parent / "fixtures" / "taxonomy_v0.json"
    taxonomy = Taxonomy.load(taxonomy_path)

    config = GatewayConfig(
        litellm_base=os.environ.get("LITELLM_BASE", "http://localhost:4000/v1"),
        litellm_key=os.environ.get("LITELLM_KEY", ""),
        transport_stuck_seconds=0,
    )
    async with LLMGateway(config) as gw:
        compiler = QueryCompiler(gateway=gw, taxonomy=taxonomy, model=model, embedding_model=None)
        spec = await compiler.compile(raw_input)

        print("=== Problem Spec ===")
        print(json.dumps({
            "raw_input": spec.raw_input,
            "domain": spec.domain,
            "sub_problems": [
                {
                    "id": sp.id, "origin": sp.origin, "parent_id": sp.parent_id,
                    "failure_summary": sp.failure_summary,
                    "target_capability": sp.target_capability,
                    "trajectory_signal": sp.trajectory_signal,
                    "keywords": sp.keywords,
                    "confidence": sp.confidence,
                } for sp in spec.sub_problems
            ],
        }, ensure_ascii=False, indent=2))
        print(f"\n=== Dropped (audit): {len(compiler.dropped_records)} ===")
        for d in compiler.dropped_records:
            print(f"  {d.id}: {d.drop_reason} (conf={d.confidence:.2f})")
        print(f"\n=== Gateway stats: {gw.http_stats} ===")


if __name__ == "__main__":
    asyncio.run(main())
