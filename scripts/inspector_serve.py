"""Launch the Trajectory Inspector service with real LLM + embedding.

Usage:
    .venv/bin/python scripts/inspector_serve.py
    # then open http://localhost:8000
"""
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 dengxuanliang

import asyncio
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
from module1.index import MemoryIndex  # noqa: E402
from module3.pipeline import select_final_dataset  # noqa: E402
from service import create_app, MemoryRunStore, PipelineDeps  # noqa: E402
from service.database import DatabaseManager  # noqa: E402
from uxflow_paths import resolve_db_path, ensure_parent  # noqa: E402


def build_app():
    root = pathlib.Path(__file__).parent.parent
    # Model split: a strong instruction-following model compiles problem specs
    # (high-leverage, low-volume); a separate model judges slices (highest-volume
    # LLM call). Two distinct env keys so compile and judge can use different
    # providers/families.
    # Compile default is gpt-5.5, NOT claude-opus-4-8: Call 1/2 demand strict JSON
    # output, and opus frequently returns empty or drops into an assistant/"memory"
    # mode on agent-behavior-flavored inputs (measured 0/3 vs gpt-5.5 3/3), which
    # surfaces as "Call 1 failed after retries". Override via UXFLOW_COMPILE_MODEL.
    compile_model = os.environ.get("UXFLOW_COMPILE_MODEL", "gpt-5.5")
    judge_model = os.environ.get("UXFLOW_JUDGE_MODEL", "gpt-4o-mini")
    emb = EmbeddingModel()
    taxonomy = Taxonomy.load(root / "fixtures" / "taxonomy_v0.json")

    # Persistent SQLite backend (path from --db > UXFLOW_DB env > XDG data dir).
    db = resolve_db_path()
    ensure_parent(db)

    gw_config = GatewayConfig(
        litellm_base=os.environ.get("LITELLM_BASE", "http://localhost:4000/v1"),
        litellm_key=os.environ.get("LITELLM_KEY", ""),
        transport_stuck_seconds=0,
    )
    gateway = LLMGateway(gw_config)  # long-lived; entered on startup

    compiler = QueryCompiler(
        gateway=gateway, taxonomy=taxonomy, model=compile_model, embedding_model=emb)
    cfg = PipelineConfig(
        judge_model=judge_model, recall_top_n=20, min_confidence=0.7, embedding_model=emb)
    # store_factory 的鸡生蛋问题：TrajectoryPipeline.__init__ 会立即调一次 store_factory()
    # 来设 self._store，但此刻 DatabaseManager 还不存在（它的构造要先拿到 pipeline 对象）。
    # 所以先用一个廉价的 MemoryIndex 占位 bootstrap（不开真 SQLite 连接、无遗留句柄），
    # 待 manager 构造后（它就地把 pipeline._store 换成真 slice_store），再把 factory 重指向
    # manager.slice_store —— 热切库后 factory 自然返回新库实例，且复用已入库的持久 slice 索引。
    pipeline = TrajectoryPipeline(
        config=cfg, gateway=gateway,
        store_factory=MemoryIndex)   # throwaway bootstrap，下方 manager 构造后被替换
    deps = PipelineDeps(
        compiler=compiler,
        pipeline=pipeline,
        select_fn=select_final_dataset,
        load_trajectories_fn=load_trajectories,
        problem_store=None,
        trajectory_store=None,
        judge_cache=None,
        embedder=emb,
    )
    # manager 就地装配 deps 的三个 store + pipeline._store + 自己的 slice_store。
    # run_lock 此处是 bootstrap 占位，create_app 建真锁后经 attach_lock 替换（见下）。
    mgr = DatabaseManager(db, deps, pipeline, run_lock=asyncio.Semaphore(1))
    pipeline._store_factory = lambda: mgr.slice_store

    store = MemoryRunStore()
    tau_q = float(os.environ.get("UXFLOW_QUESTION_DEDUP_THRESHOLD", "0.90"))
    # run_fn/search_fn/ingest_fn default inside create_app (run_pipeline etc.).
    # create_app 建真锁后调用 mgr.attach_lock 替换掉上面的 bootstrap Semaphore。
    app = create_app(store=store, deps=deps, tau_q=tau_q, db_manager=mgr)

    @app.on_event("startup")
    async def _open_gateway():
        await gateway.__aenter__()

    @app.on_event("shutdown")
    async def _close_gateway():
        await gateway.__aexit__(None, None, None)

    return app


if __name__ == "__main__":
    uvicorn.run(build_app(), host="127.0.0.1", port=8000)
