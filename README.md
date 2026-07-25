[English](README.md) | [中文](README.zh-CN.md)

# UXFlow

SWE trajectory selector for SFT dataset curation.

## What is this

UXFlow curates SFT (supervised fine-tuning) training data by selecting high-quality SWE agent trajectory slices. It runs a five-stage pipeline:

- **module0** — query compilation (LLM compiles a `ProblemSpec` from natural-language complaints)
- **module1** — recall + judge (slice, sign, index trajectories; RRF recall + LLM judging per sub-problem)
- **module2** — relevance rerank (soft-score recalled slices against each sub-problem)
- **module3** — dedup + submodular selection (cross-problem dedup, coverage-optimized final pick)
- **module0_5** — label self-evolution (proposes/backfills new capability labels into the shared taxonomy; CLI: `uxflow-evolve`)

`llm_gateway` provides the adaptive LLM call gateway used across stages; `uxflow_embed` pluggably backs embedding (Fake / Local Qwen / API). The optional `service` package exposes the same pipeline over a FastAPI Inspector UI. See [`docs/architecture.md`](docs/architecture.md) for the module map and data flow.

## Requirements

- Python 3.11–3.13
- Linux or macOS (Windows untested)
- A [LiteLLM](https://docs.litellm.ai/) proxy endpoint (OpenAI-compatible) for LLM calls

## Install (from source)

UXFlow is installed from source (it is not published to PyPI). We recommend [uv](https://github.com/astral-sh/uv).

```bash
git clone https://github.com/dengxuanliang/uxflow
cd uxflow
uv sync --extra dev --extra service   # or: pip install -e ".[dev,service]"
cp .env.example .env                 # then edit .env to fill in LITELLM_BASE / LITELLM_KEY
```

The core install is lightweight (httpx, datasketch, numpy) and needs no ML dependencies. The `service` extra pulls FastAPI + uvicorn for the Inspector UI. To enable the optional local embedding backend (Qwen via sentence-transformers + torch):

```bash
uv sync --extra local-embed           # or: pip install -e ".[local-embed]"
```

## Quickstart

```bash
# 1. Configure (one-time)
cp .env.example .env
${EDITOR:-vi} .env                    # set LITELLM_BASE + LITELLM_KEY

# 2. Run the end-to-end smoke on bundled fixtures (needs LLM only, no torch):
uv run python scripts/e2e_smoke.py "写入py文件有语法错误"

# 3. Or start the Inspector web UI:
uv run python scripts/inspector_serve.py
#    then open http://127.0.0.1:8000
```

The smoke script loads `fixtures/taxonomy_v0.json` + `fixtures/trajectories/sample_01.jsonl`, runs module 0 → 1 → 2 → 3 → 0.5 end-to-end, and prints intermediate results at each stage for human inspection.

## Data directory

Module 0.5 (label self-evolution) persists to a single SQLite file. The default location follows the XDG spec: `~/.local/share/uxflow/uxflow.db`. Override it with the `UXFLOW_DB` environment variable or the `--db` flag of `uxflow-evolve`. The database is never tracked by git.

## Embedding backends

Embedding is pluggable behind a single `Embedder` protocol in the `uxflow_embed` package:

- **FakeEmbedder** — deterministic, no ML dependencies. Used in tests.
- **LocalEmbedder** — local Qwen embedding model, requires the `local-embed` extra.
- **ApiEmbedder** — calls an OpenAI-compatible embedding endpoint.

The core install carries no ML dependencies: use `ApiEmbedder` (remote endpoint) or `FakeEmbedder`. Install the `local-embed` extra only if you want to run Qwen locally.

## Test

```bash
uv run pytest -m "not requires_model"   # pure-logic, no ML deps

uv sync --extra local-embed             # needed for the full suite
uv run pytest                           # full suite (loads the Qwen model)
```

The `-m "not requires_model"` subset runs pure logic with zero ML dependencies. The full suite requires the `local-embed` extra and downloads the Qwen model.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). For a module map and data-flow diagram, see [`docs/architecture.md`](docs/architecture.md).

## License

Apache-2.0. See [LICENSE](LICENSE).
