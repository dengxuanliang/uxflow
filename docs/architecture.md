# UXFlow Architecture

Public overview for contributors and cloners. For the frozen module 0 ↔ module 1 schema, see [`superpowers/specs/interface-contract.md`](superpowers/specs/interface-contract.md). For internal design history, see `superpowers/specs/` and `superpowers/plans/`.

## Module map

| Package | Role | Key types |
|---------|------|-----------|
| `llm_gateway` | Adaptive LLM call gateway (retries, transport watchdog, truncation recovery). Shared across all LLM stages. | `LLMGateway`, `GatewayConfig` |
| `module0` | Query compilation. Turns a natural-language complaint into a structured `ProblemSpec` (sub-problems with confidence routing). | `QueryCompiler`, `Taxonomy`, `ProblemSpec` |
| `module1` | Trajectory pipeline. Slices trajectories, builds free-layer signatures, RRF-recalls per sub-problem, LLM-judges. | `TrajectoryPipeline`, `PipelineConfig`, `Signature` |
| `module2` | Relevance rerank. Soft-scores recalled slices against each sub-problem (BM25 + vector + judge signals). | scored candidates |
| `module3` | Dedup + submodular selection. Cross-problem dedup, then coverage-optimized final pick with a general-data ratio. | `select_final_dataset`, `SelectionConfig` |
| `module0_5` | Label self-evolution. Proposes new capability labels from module0 output, dedups against the taxonomy, backfills via real judging. In-memory library (no persistence on this version). | `ingest_proposal`, `run_backfill`, `rerank_with_inheritance` |
| `uxflow_embed` | Pluggable `Embedder` protocol: `FakeEmbedder`, `LocalEmbedder` (Qwen, needs `local-embed` extra), `ApiEmbedder`. | `Embedder` |
| `service` | Optional FastAPI Inspector UI. Exposes run / search / ingest / cancel / SSE events over the same pipeline. Web frontend in `service/web/`. | `create_app`, `MemoryRunStore`, `PipelineDeps` |

## Data flow

```
complaint (text)
   │
   ▼
module0 ── ProblemSpec ──┐
   │                     │
   │                     ▼
   │                  module1 (slice + sign + index + recall + judge)
   │                     │
   │                     ▼
   │                  module2 (soft-score)
   │                     │
   │                     ▼
   │                  module3 (dedup + submodular pick) ── final SFT dataset
   │
   └── (module0_5 back-loop) ── label proposals ──► shared taxonomy
                                       ▲
                                       │
                            run_backfill (real judge)
```

- **module 0 → module 1 contract** is the only hard cross-module coupling. It is frozen in [`interface-contract.md`](superpowers/specs/interface-contract.md): `ProblemSpec` schema, `StructuredFilters` field alignment, BM25 keyword alignment, and embedding-model alignment. Changing it is a cross-module change requiring both sides to sign off.
- **module 0.5** is a side loop: it consumes module0's `LabelProposal`s and module1's `judge` callable to backfill new labels into the shared taxonomy. It never blocks the main 0→1→2→3 path.
- **service** wraps the same modules behind a FastAPI app with SSE streaming; it adds no new pipeline logic, only orchestration + a web UI.

## Entry points

- `scripts/e2e_smoke.py` — end-to-end smoke on bundled fixtures (module 0 → 3 → 0.5).
- `scripts/inspector_serve.py` — launch the Inspector web UI on `127.0.0.1:8000`.

## Configuration

All runtime config is env-driven; see [`.env.example`](../../.env.example) for the full list with defaults. The two required variables are `LITELLM_BASE` and `LITELLM_KEY` — UXFlow routes every LLM call through a LiteLLM proxy.
