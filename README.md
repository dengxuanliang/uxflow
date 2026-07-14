[English](README.md) | [中文](README.zh-CN.md)

# UXFlow

SWE trajectory selector for SFT dataset curation.

## What is this

UXFlow curates SFT (supervised fine-tuning) training data by selecting high-quality SWE agent trajectory slices. It runs a four-stage pipeline:

- **module0** — query compilation
- **module1** — recall + judge
- **module2** — relevance rerank
- **module3** — dedup + submodular selection

(`llm_gateway` provides the adaptive LLM call gateway used across stages.) See [`docs/`](docs/) for architecture details.

## Requirements

- Python 3.11–3.13
- Linux or macOS (Windows untested)

## Install (from source)

UXFlow is installed from source (it is not published to PyPI). We recommend [uv](https://github.com/astral-sh/uv).

```bash
git clone https://github.com/dengxuanliang/uxflow
cd uxflow
uv sync --extra dev            # or: pip install -e ".[dev]"
```

The core install is lightweight (httpx, datasketch, numpy) and needs no ML dependencies. To enable the optional local embedding backend (Qwen via sentence-transformers + torch):

```bash
uv sync --extra local-embed    # or: pip install -e ".[local-embed]"
```

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

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

Apache-2.0. See [LICENSE](LICENSE).
