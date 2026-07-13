"""Launch the Trajectory Inspector service with real LLM + embedding.

Usage:
    .venv/bin/python scripts/inspector_serve.py
    # then open http://localhost:8000
"""
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 dengxuanliang

import os
import pathlib

from dotenv import load_dotenv

load_dotenv(pathlib.Path(__file__).parent.parent / ".env")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import uvicorn  # noqa: E402

from llm_gateway import LLMGateway, GatewayConfig  # noqa: E402
from module0 import QueryCompiler, Taxonomy  # noqa: E402
from module0.embedding import EmbeddingModel  # noqa: E402
from module1.pipeline import TrajectoryPipeline, PipelineConfig  # noqa: E402
from module1.loader import load_trajectories  # noqa: E402
from module3.pipeline import select_final_dataset  # noqa: E402
from service import create_app, MemoryRunStore, PipelineDeps, run_pipeline  # noqa: E402


def build_app():
    root = pathlib.Path(__file__).parent.parent
    model = os.environ.get("MODULE0_TEST_MODEL", "gpt-4o-mini")
    emb = EmbeddingModel()
    taxonomy = Taxonomy.load(root / "fixtures" / "taxonomy_v0.json")

    gw_config = GatewayConfig(
        litellm_base=os.environ.get("LITELLM_BASE", "http://localhost:4000/v1"),
        litellm_key=os.environ.get("LITELLM_KEY", ""),
        transport_stuck_seconds=0,
    )
    gateway = LLMGateway(gw_config)  # long-lived; entered on startup

    compiler = QueryCompiler(
        gateway=gateway, taxonomy=taxonomy, model=model, embedding_model=emb)
    cfg = PipelineConfig(
        judge_model=model, recall_top_n=20, min_confidence=0.7, embedding_model=emb)
    pipeline = TrajectoryPipeline(config=cfg, gateway=gateway)

    deps = PipelineDeps(
        compiler=compiler,
        pipeline=pipeline,
        select_fn=select_final_dataset,
        load_trajectories_fn=load_trajectories,
    )

    store = MemoryRunStore()
    app = create_app(store=store, run_fn=run_pipeline, deps=deps)

    @app.on_event("startup")
    async def _open_gateway():
        await gateway.__aenter__()

    @app.on_event("shutdown")
    async def _close_gateway():
        await gateway.__aexit__(None, None, None)

    return app


if __name__ == "__main__":
    uvicorn.run(build_app(), host="127.0.0.1", port=8000)
