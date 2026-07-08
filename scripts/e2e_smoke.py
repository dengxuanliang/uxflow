"""End-to-end smoke test: Module 0 (QueryCompiler) → Module 1 (TrajectoryPipeline).

Connects both modules with real LLM + real local embedding model, runs on fixture
trajectories, and prints intermediate results at each stage for human inspection.

Usage:
    .venv/bin/python scripts/e2e_smoke.py "写入py文件有语法错误"
    .venv/bin/python scripts/e2e_smoke.py  # uses default query above
"""

import asyncio
import json
import os
import sys
import pathlib

from dotenv import load_dotenv
load_dotenv(pathlib.Path(__file__).parent.parent / ".env")

from llm_gateway import LLMGateway, GatewayConfig
from module0 import QueryCompiler, Taxonomy
from module0.embedding import EmbeddingModel
from module1.pipeline import TrajectoryPipeline, PipelineConfig


# ─── Helpers ────────────────────────────────────────────────────────────────

def _spec_to_dict(spec) -> dict:
    """Serialize ProblemSpec object to contract §1 dict format for Module 1."""
    return {
        "raw_input": spec.raw_input,
        "domain": spec.domain,
        "sub_problems": [
            {
                "id": sp.id,
                "origin": sp.origin,
                "parent_id": sp.parent_id,
                "raw_text": sp.raw_text,
                "failure_summary": sp.failure_summary,
                "target_capability": sp.target_capability,
                "trajectory_signal": sp.trajectory_signal,
                "hyde_positive": sp.hyde_positive,
                "keywords": sp.keywords,
                "structured_filters": {
                    "languages": sp.structured_filters.languages,
                    "tools_used": sp.structured_filters.tools_used,
                    "min_turns": sp.structured_filters.min_turns,
                    "has_verification_step": sp.structured_filters.has_verification_step,
                },
                "confidence": sp.confidence,
                "route": sp.route,
            }
            for sp in spec.sub_problems
        ],
    }


def _print_section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


# ─── Main ───────────────────────────────────────────────────────────────────

async def main():
    raw_input = sys.argv[1] if len(sys.argv) > 1 else "写入py文件有语法错误"
    model = os.environ.get("MODULE0_TEST_MODEL", "gpt-4o-mini")
    root = pathlib.Path(__file__).parent.parent

    taxonomy_path = root / "fixtures" / "taxonomy_v0.json"
    trajectories_path = root / "fixtures" / "trajectories" / "sample_01.jsonl"

    # ── 1. Load embedding model ──────────────────────────────────────────
    print("⏳ 加载 embedding 模型 (Qwen3-Embedding-0.6B)...")
    emb = EmbeddingModel()
    print(f"✓ Embedding 模型就绪 (dim={emb.dimension})")

    # ── 2. Gateway + Module 0 ────────────────────────────────────────────
    taxonomy = Taxonomy.load(taxonomy_path)

    config = GatewayConfig(
        litellm_base=os.environ.get("LITELLM_BASE", "http://localhost:4000/v1"),
        litellm_key=os.environ.get("LITELLM_KEY", ""),
        transport_stuck_seconds=0,
    )

    async with LLMGateway(config) as gw:
        # ── Module 0: Compile ProblemSpec ─────────────────────────────────
        _print_section("模块0: 编译 ProblemSpec")
        print(f"输入: \"{raw_input}\"")
        print(f"模型: {model}\n")

        compiler = QueryCompiler(
            gateway=gw, taxonomy=taxonomy, model=model, embedding_model=emb
        )
        spec = await compiler.compile(raw_input)

        print(f"产出 {len(spec.sub_problems)} 条子问题:")
        for sp in spec.sub_problems:
            print(f"\n  [{sp.id}] confidence={sp.confidence:.2f}")
            print(f"    target_capability: {sp.target_capability}")
            print(f"    trajectory_signal: {sp.trajectory_signal[:80]}...")
            print(f"    keywords: {sp.keywords}")
            sf = sp.structured_filters
            print(f"    structured_filters: languages={sf.languages}, "
                  f"tools_used={sf.tools_used}, "
                  f"min_turns={sf.min_turns}, "
                  f"has_verification_step={sf.has_verification_step}")

        if compiler.dropped_records:
            print(f"\n  Dropped ({len(compiler.dropped_records)}):")
            for d in compiler.dropped_records:
                print(f"    {d.id}: {d.drop_reason} (conf={d.confidence:.2f})")
        else:
            print("\n  Dropped: 无")

        # ── Serialize for Module 1 ───────────────────────────────────────
        spec_dict = _spec_to_dict(spec)

        if not spec.sub_problems:
            print("\n⚠️  模块0 未产出任何通过的子问题，无法继续模块1。")
            print(f"\n=== Gateway stats: {gw.http_stats} ===")
            return

        # ── Module 1: Build Index ────────────────────────────────────────
        _print_section("模块1: 切片 + 签名 + 建索引")
        print(f"轨迹文件: {trajectories_path.name}")

        cfg = PipelineConfig(
            judge_model=model,
            recall_top_n=20,
            min_confidence=0.7,
            embedding_model=emb,
        )
        pipeline = TrajectoryPipeline(config=cfg, gateway=gw)
        pipeline._build_index([trajectories_path])

        print(f"索引大小: {pipeline._index.size} 条签名\n")
        print("切片明细:")
        for (traj_id, slice_idx), sl in sorted(pipeline._slice_map.items()):
            # Find matching signature
            sig = None
            for s in pipeline._index._signatures:
                if s.trajectory_id == traj_id and s.slice_index == slice_idx:
                    sig = s
                    break
            print(f"  {traj_id} / slice {slice_idx}: "
                  f"steps [{sl.start_step}-{sl.end_step}] "
                  f"({sl.step_count} steps)")
            if sig:
                print(f"    languages={sig.languages}, "
                      f"tools_used={sig.tools_used}")
                print(f"    has_error={sig.has_error_pattern}, "
                      f"has_success={sig.has_success_pattern}, "
                      f"has_verify={sig.has_verification_step}")
                print(f"    bm25_tokens (top 10): {sig.bm25_tokens[:10]}")
                print(f"    embedding: {'有' if sig.embedding else '无'} "
                      f"({len(sig.embedding)}d)" if sig.embedding else
                      "    embedding: 无")

        # ── Module 1: Recall + Judge per sub_problem ─────────────────────
        _print_section("模块1: 召回 + 精判")

        all_candidates = []
        spec_id = spec_dict.get("raw_input", "unknown")[:50]

        for sp_dict in spec_dict["sub_problems"]:
            sp_id = sp_dict["id"]
            print(f"── 子问题 [{sp_id}] ──")
            print(f"  target_capability: {sp_dict['target_capability']}")
            print(f"  keywords: {sp_dict['keywords']}")
            print(f"  structured_filters: {sp_dict['structured_filters']}")
            print(f"  hyde_positive 段数: {len(sp_dict['hyde_positive'])}")
            print()

            # Diagnose recall funnel before calling _process_sub_problem
            sf = sp_dict.get("structured_filters", {})
            kw = sp_dict.get("keywords", [])
            hp = sp_dict.get("hyde_positive", [])
            query_embs = cfg.embedding_model.embed_batch(hp) if cfg.embedding_model and hp else []

            filter_passed = pipeline._index._apply_filters(sf)
            print(f"  过滤后候选: {len(filter_passed)} 条 "
                  f"(共 {pipeline._index.size} 条签名)")
            if not filter_passed:
                # Show which filter eliminated everything
                for fname, fval in sf.items():
                    if fval is None:
                        continue
                    test_filter = {fname: fval}
                    n = len(pipeline._index._apply_filters(test_filter))
                    print(f"    单独 {fname}={fval} → {n}/{pipeline._index.size} 通过")
                # Relax: drop min_turns filter and retry to exercise full pipeline
                sf_relaxed = {k: (None if k == "min_turns" else v) for k, v in sf.items()}
                filter_relaxed = pipeline._index._apply_filters(sf_relaxed)
                if filter_relaxed:
                    print(f"  → 放宽 min_turns 后: {len(filter_relaxed)} 条通过，"
                          "继续跑 BM25/向量/judge（仅为观测）")
                    sf = sf_relaxed
                    sp_dict = {**sp_dict, "structured_filters": sf}
                    filter_passed = filter_relaxed
                else:
                    print("  ⚠️  即使放宽 min_turns 仍无候选，跳过")
                    print()
                    continue

            # Show BM25 + vector scores for transparency
            bm25_scores = pipeline._index._bm25_score(filter_passed, kw) if kw else {}
            vector_scores = pipeline._index._vector_score(filter_passed, query_embs) if query_embs else {}
            print(f"  BM25 命中: {len(bm25_scores)} 条, "
                  f"向量命中: {len(vector_scores)} 条")

            recalled = pipeline._index.recall(
                structured_filters=sf, keywords=kw,
                query_embeddings=query_embs, top_n=cfg.recall_top_n)
            print(f"  RRF 召回 top-{cfg.recall_top_n}: {len(recalled)} 条")
            for r in recalled[:5]:
                print(f"    {r.trajectory_id}/slice{r.slice_index}")
            print()

            candidates = await pipeline._process_sub_problem(sp_dict, spec_id)

            if candidates:
                print(f"  ✓ Judge 命中 {len(candidates)} 条轨迹:")
                for c in candidates:
                    for mp in c.matched_problems:
                        print(f"    {c.trajectory_id}: "
                              f"confidence={mp['confidence']:.2f}, "
                              f"spans={mp['loss_mask_spans']}")
            else:
                if recalled:
                    print("  ✗ 召回有结果但 Judge 全部 miss "
                          f"(min_confidence={cfg.min_confidence})")
                else:
                    print("  ✗ 召回为空，无法进入 Judge")
            print()

            all_candidates.extend(candidates)

        # ── Final Output ─────────────────────────────────────────────────
        _print_section("模块1: 最终 SFTCandidate")

        if all_candidates:
            # Merge by trajectory_id (same logic as pipeline.run)
            merged: dict = {}
            for c in all_candidates:
                if c.trajectory_id in merged:
                    merged[c.trajectory_id].matched_problems.extend(
                        c.matched_problems
                    )
                else:
                    merged[c.trajectory_id] = c

            print(f"共 {len(merged)} 条唯一轨迹命中:\n")
            for c in merged.values():
                print(f"  trajectory_id: {c.trajectory_id}")
                print(f"  trajectory_path: {c.trajectory_path}")
                for i, mp in enumerate(c.matched_problems):
                    print(f"    match[{i}]: capability={mp['capability']}, "
                          f"confidence={mp['confidence']:.2f}")
                    print(f"             loss_mask_spans={mp['loss_mask_spans']}")
                print()
        else:
            print("⚠️  无任何 SFTCandidate 产出。")
            print("   排查方向: 检查上方召回结果是否为空 → "
                  "若空则检查 filters/keywords 对齐；"
                  "若有召回但 judge 全 miss 则检查 confidence 阈值。")

        # ── Stats ────────────────────────────────────────────────────────
        _print_section("Gateway 统计")
        print(f"  {gw.http_stats}")


if __name__ == "__main__":
    asyncio.run(main())
