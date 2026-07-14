"""uxflow-evolve: standing module 0.5 orchestration CLI.

Loads persistent stores from the resolved DB path, compiles a query (module 0)
to harvest label_proposals, then runs evolve_once (② ingest → enqueue → ④ worker).

Usage:
    uxflow-evolve "修复 Python 运行时异常" --db /path/uxflow.db --once
"""
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 dengxuanliang

from __future__ import annotations

import argparse
import asyncio
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv  # noqa: E402
load_dotenv()
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from uxflow_paths import resolve_db_path, ensure_parent  # noqa: E402


def _iso_now() -> str:
    # CLI boundary may use real time (repo convention forbids it only in pure/DB layers)
    import datetime
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


async def _amain(args) -> int:
    try:
        from llm_gateway import LLMGateway, GatewayConfig
        from module0 import QueryCompiler, Taxonomy
        from module0.embedding import EmbeddingModel
        from module0.sqlite_taxonomy import SqliteTaxonomyStore
        from module1.sqlite_store import SqliteSliceStore
        from module1.judge import Judge
        from module0_5.queue import SqliteBackfillQueue
        from module0_5.daemon import evolve_once

        db_path = resolve_db_path(args.db)
        ensure_parent(db_path)
        print(f"DB: {db_path}")

        seed = Taxonomy.load(args.taxonomy) if args.taxonomy else None
        slice_store = SqliteSliceStore(db_path)
        taxonomy_store = SqliteTaxonomyStore(db_path, seed=seed)
        queue = SqliteBackfillQueue(db_path)

        emb = EmbeddingModel()
        config = GatewayConfig(
            litellm_base=os.environ.get("LITELLM_BASE", "http://localhost:4000/v1"),
            litellm_key=os.environ.get("LITELLM_KEY", ""),
            transport_stuck_seconds=0)
        async with LLMGateway(config) as gw:
            compiler = QueryCompiler(gateway=gw, taxonomy=taxonomy_store.snapshot(),
                                     model=args.model, embedding_model=emb)
            await compiler.compile(args.query)
            proposals = list(compiler.label_proposals)
            print(f"module0 提议 {len(proposals)} 个新标签")

            judge = Judge(gateway=gw, model=args.model)
            summary = await evolve_once(proposals, slice_store, taxonomy_store, queue, judge,
                                        created_at=_iso_now(), now_fn=_iso_now)
            print(f"演化结果: {summary}")
        return 0
    except Exception as e:  # noqa: BLE001 — CLI boundary: report, don't traceback
        print(f"uxflow-evolve failed: {e}", file=sys.stderr)
        return 1


def main() -> int:
    p = argparse.ArgumentParser(prog="uxflow-evolve")
    p.add_argument("query", help="raw query to compile for label proposals")
    p.add_argument("--db", default=None, help="SQLite DB path (overrides UXFLOW_DB/XDG)")
    p.add_argument("--taxonomy", default=None, help="seed taxonomy JSON (first run only)")
    p.add_argument("--model", default=os.environ.get("MODULE0_TEST_MODEL", "gpt-4o-mini"))
    p.add_argument("--once", action="store_true",
                   help="reserved; the CLI currently always runs a single pass")
    args = p.parse_args()
    return asyncio.run(_amain(args))


if __name__ == "__main__":
    raise SystemExit(main())
