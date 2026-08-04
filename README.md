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
- For real runs: an OpenAI-compatible LLM endpoint, reached through a [LiteLLM](https://docs.litellm.ai/) proxy ([step 1](#step-1--start-an-llm-proxy) walks you through starting one)

## Install

UXFlow is installed from source (it is not published to PyPI). We recommend [uv](https://github.com/astral-sh/uv).

```bash
git clone https://github.com/dengxuanliang/uxflow
cd uxflow
uv sync --extra dev --extra service
```

The core install is lightweight (httpx, datasketch, numpy) and pulls **no ML dependencies**. The `service` extra adds FastAPI + uvicorn for the Inspector UI.

---

## Quickstart — run the whole pipeline right now

No proxy, no API key, no ML dependencies. One command:

```bash
UXFLOW_LLM_BACKEND=fake UXFLOW_EMBED_BACKEND=fake \
  uv run python scripts/e2e_smoke.py "写入py文件有语法错误"
```

### What you should see

It starts by naming the active backends, then walks all five stages:

```
⏳ 构建 embedding backend (fake)...
✓ Embedding 就绪 (dim=1024)
╔════════════════════════════════════════════════════════════════╗
║  ⚠️  FAKE 后端 — 本次运行的结果不可用于真实数据筛选             ║
╠════════════════════════════════════════════════════════════════╣
║  UXFLOW_LLM_BACKEND=fake                                        ║
║  UXFLOW_EMBED_BACKEND=fake                                      ║
...
╚════════════════════════════════════════════════════════════════╝

  模块0: 编译 ProblemSpec          → 产出 2 条子问题
  模块1: 切片 + 签名 + 建索引
  模块1: 召回 + 精判
  模块2/3: 软加分 + 去重覆盖优选   → 最终 targeted: 6
  模块0.5: ①提议捕获 → ②入库 → ③继承 → ④回填 → ③升权
  Gateway 统计                     → fake backend: no HTTP traffic
```

Exit code `0`, and the same warning banner is printed again at the end.

### What this proves — and what it doesn't

✅ Your install works, all five stages wire together, SQLite persistence survives a restart.

❌ **Nothing about result quality.** LLM answers are fixed replays and embedding vectors are deterministic hashes. `最终 targeted: 6` is a real code path producing meaningless picks. Use this to verify the plumbing, never to judge the output.

That is why every fake run prints the banner twice, names *both* backends, and the Inspector UI shows a standing warning: so a fake result is never mistaken for a real one.

---

## Going real

Four things change between the run above and a real one. Do them in order — each step has a check so you find problems where they happen, not three steps later.

> **Two independent switches.** `UXFLOW_LLM_BACKEND` and `UXFLOW_EMBED_BACKEND` are separate. Flipping only one gives you a half-real run: correct judging over meaningless recall ordering, or the reverse. Both must be real before results mean anything — the startup banner always prints both so you can check at a glance.

### Step 1 — start an LLM proxy

UXFlow sends every LLM call through a LiteLLM proxy. Skip this step if you already run one.

**Option A — via uv (no Docker):**

```bash
cp litellm.config.example.yaml litellm.config.yaml
${EDITOR:-vi} litellm.config.yaml          # point litellm_params.model at your upstream

export OPENAI_API_KEY=sk-...               # your provider key
export LITELLM_MASTER_KEY=sk-local-dev

uvx --from 'litellm[proxy]==1.95.0' --with 'fastapi<0.140.7' \
  litellm --config litellm.config.yaml --port 4000
```

> The `fastapi<0.140.7` pin is required: 0.140.7 removed `get_flat_dependant`, which litellm's proxy still imports, and litellm declares no upper bound. Without the pin the proxy dies at startup — and misleadingly, as `ModuleNotFoundError: No module named 'proxy_server'`, because its CLI swallows the real ImportError. To see the actual cause, run `python -c "import litellm.proxy.proxy_server"`.
>
> A `failed to fetch remote model cost map ... falling back to local backup` warning is harmless — litellm could not reach its pricing table and used the bundled copy. Silence it with `export LITELLM_LOCAL_MODEL_COST_MAP=True`.

**Option B — via Docker:**

```bash
cp litellm.config.example.yaml litellm.config.yaml
export OPENAI_API_KEY=sk-...
docker compose -f docker-compose.litellm.yml up -d
```

**Expected:** uvicorn startup logs, listening on port 4000. Leave this running and open a second terminal for the remaining steps.

> ⚠️ **The two `model_name` values must match what UXFlow asks for** — `gpt-5.5` for compilation, `gpt-4o-mini` for judging. A mismatch surfaces much later as the misleading `Call 1 failed after retries`. To use different names, set `UXFLOW_COMPILE_MODEL` / `UXFLOW_JUDGE_MODEL` to match.

### Step 2 — configure `.env`

```bash
cp .env.example .env
```

Set these three:

| Variable | Value | Note |
|---|---|---|
| `LITELLM_BASE` | `http://localhost:4000/v1` | already the default |
| `LITELLM_KEY` | `sk-local-dev` | must equal `LITELLM_MASTER_KEY` from step 1 |
| `UXFLOW_LLM_BACKEND` | `real` | already the default |

`real` is the default on purpose: with credentials missing it **fails loudly** rather than silently falling back to replays.

### Step 3 — verify the proxy answers

```bash
uv run python scripts/check_model_split.py
```

This isolates one question — do both model names route through and answer? — so a failure here is definitively connectivity or naming, never business logic.

**Expected:**

```
=== result ===
  compile (gpt-5.5): PASS
  judge   (gpt-4o-mini): PASS

✓ both models answered — split is live.
```

Exit code `0`. **If it fails**, the script tells you how to read it: `404-ish / empty` → that model name is not registered in your `litellm.config.yaml`; `timeout / conn` → the proxy at `LITELLM_BASE` is not reachable.

Do not continue until this passes.

### Step 4 — install a real embedding backend

```bash
uv sync --extra local-embed
```

This pulls `torch` + `sentence-transformers` (a large download), and on first run fetches `Qwen/Qwen3-Embedding-0.6B` from HuggingFace. If HuggingFace is slow or blocked for you, see [Behind a restricted network](#behind-a-restricted-network).

Prefer not to run a model locally? Use a hosted embedding endpoint instead — set `UXFLOW_EMBED_BACKEND=api` and see [Embedding backends](#embedding-backends).

### Step 5 — run it for real

```bash
UXFLOW_EMBED_BACKEND=local UXFLOW_DB=~/.local/share/uxflow/real.db \
  uv run python scripts/e2e_smoke.py "写入py文件有语法错误"
```

**Expected — three differences from the fake run:**

1. **No warning box.** Instead, a single line:
   ```
   ✓ LLM: real   ✓ Embedding: local
   ```
   Seeing no warning *is* the confirmation. If the box still appears, one backend didn't switch — the box names which.

2. **Real HTTP traffic** in the closing stats:
   ```
   {'total_http_requests': 12, 'success': 12, 'errors': {}, 'success_rate': 100.0}
   ```
   The fake run says `fake backend: no HTTP traffic` here. This is the hardest evidence that calls actually went out.

3. **Results that mean something** — relevance scores now reflect semantic similarity rather than hash noise.

> ⚠️ **Point `UXFLOW_DB` at a fresh database when switching backends.** Vectors persisted in SQLite are backend-specific; reusing the database from your fake run silently compares hash vectors against real ones.

**If module0 reports `产出 0 条子问题` with everything dropped**, your compile model isn't returning strict JSON. That is a model-capability issue, not a bug — switch `UXFLOW_COMPILE_MODEL` to a stronger instruction-following model.

---

## Inspector web UI

The same pipeline behind a browser UI, with live progress over SSE:

```bash
uv run python scripts/inspector_serve.py     # then open http://127.0.0.1:8000
```

It honours the same backend switches. Under a fake backend it shows a standing warning banner in the page header — the terminal banner is invisible to anyone using the browser.

## Configuration reference

All configuration is environment-driven; [`.env.example`](.env.example) documents every variable. The ones that matter most:

| Variable | Default | Purpose |
|---|---|---|
| `LITELLM_BASE` | `http://localhost:4000/v1` | LiteLLM proxy endpoint |
| `LITELLM_KEY` | *(empty)* | Proxy API key — required for real runs |
| `UXFLOW_LLM_BACKEND` | `real` | `real` \| `fake` |
| `UXFLOW_EMBED_BACKEND` | `fake` | `fake` \| `local` \| `api` |
| `UXFLOW_COMPILE_MODEL` | `gpt-5.5` | Compiles the ProblemSpec; needs strict JSON |
| `UXFLOW_JUDGE_MODEL` | `gpt-4o-mini` | Judges slices; highest-volume call |
| `UXFLOW_DB` | `~/.local/share/uxflow/uxflow.db` | SQLite path (XDG) |

### Embedding backends

Embedding is pluggable behind a single `Embedder` protocol in `uxflow_embed`:

- **`fake`** (default) — deterministic, no ML dependencies. Vectors carry no semantics.
- **`local`** — Qwen3-Embedding-0.6B via sentence-transformers. Needs the `local-embed` extra; downloads the model on first use.
- **`api`** — any OpenAI-compatible embedding endpoint. Needs `OPENAI_API_KEY`; tune with `UXFLOW_EMBED_API_MODEL` / `_DIM` / `_BASE`.

### Data directory

Module 0.5 persists to a single SQLite file, defaulting to the XDG path `~/.local/share/uxflow/uxflow.db`. Override with `UXFLOW_DB` or `uxflow-evolve --db`. It is never tracked by git.

## Behind a restricted network

UXFlow itself needs no external services beyond your LLM endpoint, but three of its *tools* fetch from infrastructure that may be slow or blocked in some regions. Each has a mirror:

| Step | Fetches from | Workaround |
|---|---|---|
| `uv sync` / `uvx` | PyPI | `export UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple` |
| `uv` provisioning Python | GitHub Releases | `export UV_PYTHON_INSTALL_MIRROR=<a GitHub release proxy>`, or install a Python ≥3.11 yourself |
| `UXFLOW_EMBED_BACKEND=local` | HuggingFace | `export HF_ENDPOINT=https://hf-mirror.com` |
| `docker compose ... litellm` | Docker Hub | use **Option A** in [step 1](#step-1--start-an-llm-proxy) — it goes through PyPI instead |

If a TLS-intercepting proxy sits in front of you, Docker fails with `x509: certificate signed by unknown authority` even when `curl` to the same host succeeds — curl trusts the proxy's CA from the system store, the Docker daemon does not. Either install that CA for the daemon and restart it, or take Option A.

Leave `HF_HUB_OFFLINE` / `TRANSFORMERS_OFFLINE` commented out until the Qwen model is actually on disk — setting them early blocks the very download they're meant to skip.

## Test

```bash
uv run pytest -m "not requires_model"   # pure logic, no ML deps

uv sync --extra local-embed             # needed for the full suite
uv run pytest                           # full suite (loads the Qwen model)
```

The `-m "not requires_model"` subset is what CI runs across Linux/macOS × Python 3.11–3.13.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). For a module map and data-flow diagram, see [`docs/architecture.md`](docs/architecture.md).

## License

Apache-2.0. See [LICENSE](LICENSE).
