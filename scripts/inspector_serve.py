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
from module1.sqlite_store import SqliteSliceStore  # noqa: E402
from module3.pipeline import select_final_dataset  # noqa: E402
from service import create_app, MemoryRunStore, PipelineDeps  # noqa: E402
from service.stores import (  # noqa: E402
    SqliteProblemStore,
    SqliteTrajectoryStore,
    SqliteJudgeCache,
)
from uxflow_paths import resolve_db_path, ensure_parent  # noqa: E402


def build_app():
    root = pathlib.Path(__file__).parent.parent
    model = os.environ.get("MODULE0_TEST_MODEL", "gpt-4o-mini")
    emb = EmbeddingModel()
    taxonomy = Taxonomy.load(root / "fixtures" / "taxonomy_v0.json")

    # Persistent SQLite backend (path from --db > UXFLOW_DB env > XDG data dir).
    db = resolve_db_path()
    ensure_parent(db)
    # One long-lived slice store: search must recall against the already-ingested
    # slice index, so the same instance must persist across runs.
    slice_store = SqliteSliceStore(db)

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
    # store_factory returns the SAME persistent slice_store singleton on every
    # call (not a fresh MemoryIndex): search recalls against the already-ingested,
    # persisted slice index — a per-call `new` would throw that index away.
    pipeline = TrajectoryPipeline(
        config=cfg, gateway=gateway, store_factory=lambda: slice_store)

    deps = PipelineDeps(
        compiler=compiler,
        pipeline=pipeline,
        select_fn=select_final_dataset,
        load_trajectories_fn=load_trajectories,
        problem_store=SqliteProblemStore(db),
        trajectory_store=SqliteTrajectoryStore(db),
        judge_cache=SqliteJudgeCache(db),
        embedder=emb,
    )

    store = MemoryRunStore()
    tau_q = float(os.environ.get("UXFLOW_QUESTION_DEDUP_THRESHOLD", "0.90"))
    # run_fn/search_fn/ingest_fn default inside create_app (run_pipeline etc.).
    app = create_app(store=store, deps=deps, tau_q=tau_q)

    @app.on_event("startup")
    async def _open_gateway():
        await gateway.__aenter__()

    @app.on_event("shutdown")
    async def _close_gateway():
        await gateway.__aexit__(None, None, None)

    return app


if __name__ == "__main__":
    uvicorn.run(build_app(), host="127.0.0.1", port=8000)
