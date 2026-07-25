# OSS "Clone-and-Run" Readiness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make UXFlow open out-of-the-box after `git clone` — a cloner can install, configure, run the demo, and start the Inspector service without reading any private/internal docs.

**Architecture:** No new code modules. This is a packaging + documentation pass: ship a tracked `.env.example`, surface the already-existing `service` package and `scripts/` entry points in README, add a public architecture doc, and fill standard GitHub OSS hygiene files. All changes are additive except one pyproject cleanup (`onnx-embed` extra has zero references and is removed). Per-concern test files under `tests/` guard the invariants so they don't regress — one file per concern, matching the existing top-level `tests/test_*.py` convention.

**Tech Stack:** Markdown docs, `python-dotenv` (already a dev dep) for `.env.example` parsing, GitHub issue/PR templates, `uv`/`ruff`/`pytest` for verification.

**Scope note:** All items are small, cohesive, and cross-reference each other (README links to `.env.example`, architecture.md, and `scripts/`). One plan with three phases is cleaner than splitting. If you'd rather ship in slices, Phase A alone makes the project runnable.

**Reference skills:** @superpowers:test-driven-development (light tests for invariants), @superpowers:verification-before-completion (Phase D before claiming done).

---

## Resolved decisions (from discussion before execution)

1. **No `.python-version` pin.** CI already tests 3.11/3.12/3.13; pinning only adds friction for contributors whose local Python differs. Local default stays the contributor's choice.
2. **Drop `onnx-embed` extra** (confirmed destructive — zero `.py` references).
3. **`docs/architecture.md` stays an ~80-line overview** that links the frozen `interface-contract.md`; do not duplicate the 54KB internal design doc.
4. **Bilingual issue templates.** Ship an EN + zh-CN pair for both bug and feature templates (disambiguated via the `name:` field in the chooser).
5. **Tests split by concern** — one `tests/test_<concern>.py` per concern, matching the existing top-level `tests/test_*.py` convention (e.g., `test_gateway.py`, `test_uxflow_paths.py`).

---

## File Structure

**Create:**
- `.env.example` — tracked env template (LITELLM_*, UXFLOW_*, optional flags). NOT matched by `.gitignore` `.env` pattern.
- `SECURITY.md` — vulnerability reporting policy + `.env`-never-tracked reminder.
- `docs/architecture.md` — public module overview (replaces the gitignored `docs/architecture-*.md` as the cloner-facing entry).
- `.github/ISSUE_TEMPLATE/bug_report.md`, `bug_report.zh-CN.md`, `feature_request.md`, `feature_request.zh-CN.md`, `config.yml`
- `.github/PULL_REQUEST_TEMPLATE.md` — bilingual inline (GitHub only honors one PR template file, no chooser).
- `tests/test_env_example.py`, `tests/test_readme.py`, `tests/test_architecture_doc.py`, `tests/test_security_policy.py`, `tests/test_github_templates.py`, `tests/test_pyproject_extras.py` — one invariant test file per concern.

**Modify:**
- `README.md` + `README.zh-CN.md` — fold `module0_5` into stage list; add `service` extra to Install; add "Quickstart" + "Run the Inspector" sections; fix `docs/` link to point at `docs/architecture.md`.
- `CONTRIBUTING.md` + `CONTRIBUTING.zh-CN.md` — link the new architecture doc + `.env.example`.
- `pyproject.toml` — drop dead `onnx-embed` extra (line 19); the only destructive change, called out in Task C5.
- `.gitignore` — no change needed (`.env` already ignores only real `.env`, not `.env.example`); Task A1 verifies this.

---

## Phase A — First-run enablement (blocks running; ship this first)

### Task A1: Create `.env.example` and a guard test

**Files:**
- Create: `.env.example`
- Test: `tests/test_env_example.py`

**Context:** env vars consumed across `scripts/*.py` and `src/uxflow_embed/api.py` (verified via grep):
- **Required:** `LITELLM_BASE`, `LITELLM_KEY` (every script reads these; no default for `LITELLM_KEY`).
- **Optional with defaults:** `UXFLOW_COMPILE_MODEL` (gpt-5.5), `UXFLOW_JUDGE_MODEL` (gpt-4o-mini), `MODULE0_TEST_MODEL` (gpt-4o-mini), `UXFLOW_DB` (XDG), `UXFLOW_QUESTION_DEDUP_THRESHOLD` (0.90), `HF_HUB_OFFLINE` (1), `TRANSFORMERS_OFFLINE`, `OPENAI_API_KEY` (ApiEmbedder only), `XDG_DATA_HOME`.
- **Test flag:** `MODULE0_INTEGRATION=1` (gates integration tests).

- [ ] **Step 1: Write the failing test**

`tests/test_env_example.py`:

```python
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 dengxuanliang
"""Guard: .env.example exists, is tracked, lists required keys, has no real secrets."""
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parent.parent
ENV_EXAMPLE = ROOT / ".env.example"

REQUIRED_ENV_KEYS = {"LITELLM_BASE", "LITELLM_KEY"}


def test_env_example_exists():
    assert ENV_EXAMPLE.is_file(), ".env.example missing — cloners can't configure"


def test_env_example_lists_required_keys():
    text = ENV_EXAMPLE.read_text(encoding="utf-8")
    for key in REQUIRED_ENV_KEYS:
        assert re.search(rf"^#?\s*{key}\s*=", text, re.MULTILINE), (
            f".env.example missing {key}")


def test_env_example_has_no_real_secrets():
    text = ENV_EXAMPLE.read_text(encoding="utf-8")
    assert not re.search(r"sk-[A-Za-z0-9]{20,}", text), (
        ".env.example must not contain a real-looking API key")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_env_example.py -v`
Expected: FAIL — `.env.example missing — cloners can't configure`.

- [ ] **Step 3: Write `.env.example`**

`.env.example`:

```dotenv
# UXFlow environment template. Copy to `.env` and fill in real values.
#   cp .env.example .env
# `.env` is gitignored; never commit real keys. See SECURITY.md.
#
# UXFlow routes all LLM calls through a LiteLLM proxy (one base URL, one key,
# many models). Stand up LiteLLM separately — see https://docs.litellm.ai/.
# The proxy must expose the models named below on your provider.

# ─── Required ────────────────────────────────────────────────────────────
# LiteLLM proxy base URL (OpenAI-compatible /v1 endpoint).
LITELLM_BASE="http://localhost:4000/v1"
# LiteLLM proxy API key.
LITELLM_KEY=""

# ─── Optional: model selection (defaults shown) ──────────────────────────
# Model that compiles ProblemSpec in module0 (high-leverage, low-volume).
# Default: gpt-5.5. Must follow strict JSON output.
UXFLOW_COMPILE_MODEL=gpt-5.5
# Model that judges trajectory slices in module1 (highest-volume LLM call).
UXFLOW_JUDGE_MODEL=gpt-4o-mini
# Legacy alias used by smoke scripts; defaults to gpt-4o-mini.
MODULE0_TEST_MODEL=gpt-4o-mini

# ─── Optional: runtime paths & tuning ────────────────────────────────────
# SQLite DB for module0.5 label self-evolution. Default: XDG data dir
# (~/.local/share/uxflow/uxflow.db). Override or use `uxflow-evolve --db`.
# UXFLOW_DB=
# Question-level dedup threshold for the Inspector search path. Default 0.90.
UXFLOW_QUESTION_DEDUP_THRESHOLD=0.90

# ─── Optional: local embedding backend ───────────────────────────────────
# Force HuggingFace offline: use local Qwen3-Embedding cache, skip the
# network revision check (a flaky network returns RST and breaks it).
# Only relevant if you installed the `local-embed` extra.
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
# ApiEmbedder (OpenAI-compatible endpoint) only. Not needed for Local/Fake.
# OPENAI_API_KEY=

# ─── Test flag (leave unset for normal runs) ──────────────────────────────
# Set to 1 to enable module0 integration tests against a live LiteLLM proxy.
# MODULE0_INTEGRATION=1
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_env_example.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add .env.example tests/test_env_example.py
git commit -m "docs: add .env.example and env-example invariant test"
```

---

### Task A2: Update README (EN + zh-CN) — stage list, Install, Quickstart, Inspector

**Files:**
- Modify: `README.md`
- Modify: `README.zh-CN.md`
- Test: `tests/test_readme.py`

**Context:** README currently omits `module0_5` from the stage list, never mentions the `service` package or `scripts/inspector_serve.py`, and points at a bare `docs/` that is otherwise empty for cloners. CI already installs `--extra service` (`.github/workflows/ci.yml:23`), so the extra is real and tested.

- [ ] **Step 1: Write the failing test**

`tests/test_readme.py`:

```python
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 dengxuanliang
"""Guard: README surfaces the service extra, Quickstart, module0_5, and architecture link."""
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
README = ROOT / "README.md"
README_CN = ROOT / "README.zh-CN.md"


def test_readme_mentions_service_extra_and_inspector():
    text = README.read_text(encoding="utf-8")
    assert "--extra service" in text, "README Install section must list the service extra"
    assert "inspector_serve" in text, "README must show how to start the Inspector"
    assert "Quickstart" in text or "quickstart" in text.lower()


def test_readme_stage_list_includes_module0_5():
    text = README.read_text(encoding="utf-8")
    assert "module0_5" in text or "module 0.5" in text.lower()
    assert "uxflow-evolve" in text


def test_readme_docs_link_targets_architecture_md():
    for readme in (README, README_CN):
        text = readme.read_text(encoding="utf-8")
        assert "docs/architecture.md" in text, (
            f"{readme.name} should link docs/architecture.md, not bare docs/")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_readme.py -v`
Expected: 3 FAIL (service/quickstart/module0_5/architecture link all missing).

- [ ] **Step 3: Rewrite `README.md`**

Replace the full file with:

```markdown
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
$EDITOR .env                          # set LITELLM_BASE + LITELLM_KEY

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
```

- [ ] **Step 4: Mirror the same structural changes in `README.zh-CN.md`**

Apply the same edits to the Chinese README: fold `module0_5` into the stage list with the same one-liners, add `--extra service` to the Install block, add `.env.example` copy step, add a `## 快速上手` Quickstart section, change the bare `docs/` link to `docs/architecture.md`, and add the LiteLLM requirement line. Keep the existing Chinese prose style.

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_readme.py -v`
Expected: PASS (3 tests).

- [ ] **Step 6: Lint + commit**

Run: `uv run ruff check`
Expected: clean.

```bash
git add README.md README.zh-CN.md tests/test_readme.py
git commit -m "docs: surface service extra, Quickstart, module0_5 in README"
```

---

## Phase B — Public architecture doc

### Task B1: Write `docs/architecture.md` and wire links

**Files:**
- Create: `docs/architecture.md`
- Modify: `CONTRIBUTING.md`, `CONTRIBUTING.zh-CN.md`
- Test: `tests/test_architecture_doc.py`

**Context:** `docs/` currently contains only `superpowers/plans|specs/` (internal dev artifacts). The cloner-facing `architecture-talk-3min.md`, `architecture-uml.md`, and `diagrams/` are gitignored as personal artifacts. README now points cloners at `docs/architecture.md`, so it must exist and be a clean public overview — NOT a dump of the 54KB internal design doc. The canonical interface contract is already tracked at `docs/superpowers/specs/interface-contract.md`; reference it, don't duplicate it.

- [ ] **Step 1: Write the failing test**

`tests/test_architecture_doc.py`:

```python
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 dengxuanliang
"""Guard: docs/architecture.md exists, covers every module, links the interface contract."""
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
ARCH_DOC = ROOT / "docs" / "architecture.md"

EXPECTED_MODULES = (
    "module0", "module1", "module2", "module3",
    "module0_5", "llm_gateway", "uxflow_embed", "service",
)


def test_public_architecture_doc_exists():
    assert ARCH_DOC.is_file(), "docs/architecture.md missing — README links it"


def test_architecture_doc_covers_all_modules():
    text = ARCH_DOC.read_text(encoding="utf-8")
    for name in EXPECTED_MODULES:
        assert name in text, f"architecture.md missing module {name}"


def test_architecture_doc_links_interface_contract():
    text = ARCH_DOC.read_text(encoding="utf-8")
    assert "interface-contract.md" in text
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_architecture_doc.py -v`
Expected: FAIL — `docs/architecture.md missing`.

- [ ] **Step 3: Write `docs/architecture.md`**

```markdown
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
| `module0_5` | Label self-evolution. Proposes new capability labels from module0 output, dedups against the taxonomy, backfills via real judging, persists to SQLite. CLI: `uxflow-evolve`. | `ingest_proposal`, `run_backfill`, `SqliteTaxonomyStore` |
| `uxflow_embed` | Pluggable `Embedder` protocol: `FakeEmbedder`, `LocalEmbedder` (Qwen, needs `local-embed` extra), `ApiEmbedder`. | `Embedder` |
| `service` | Optional FastAPI Inspector UI. Exposes run / search / ingest / cancel / SSE events over the same pipeline. Web frontend in `service/web/`. | `create_app`, `MemoryRunStore`, `PipelineDeps` |
| `uxflow_paths` | Resolves the SQLite DB path: `--db` > `UXFLOW_DB` env > XDG data dir. | `resolve_db_path` |

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
   └── (module0_5 back-loop) ── label proposals ──► taxonomy (SQLite)
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
- `uxflow-evolve` (installed console script → `scripts/uxflow_evolve.py`) — module0.5 label evolution CLI.

## Configuration

All runtime config is env-driven; see [`.env.example`](../../.env.example) for the full list with defaults. The two required variables are `LITELLM_BASE` and `LITELLM_KEY` — UXFlow routes every LLM call through a LiteLLM proxy.
```

- [ ] **Step 4: Wire links in CONTRIBUTING**

In `CONTRIBUTING.md`, after the "Dev setup" section, add:

```markdown
## Architecture

For a module map and data-flow diagram, see [`docs/architecture.md`](docs/architecture.md). The module 0 ↔ module 1 contract is frozen in [`docs/superpowers/specs/interface-contract.md`](docs/superpowers/specs/interface-contract.md).
```

Mirror the same paragraph in `CONTRIBUTING.zh-CN.md` after "开发环境搭建":

```markdown
## 架构

模块地图与数据流图见 [`docs/architecture.md`](docs/architecture.md)。模块 0 ↔ 模块 1 的接口契约冻结于 [`docs/superpowers/specs/interface-contract.md`](docs/superpowers/specs/interface-contract.md)。
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_architecture_doc.py -v`
Expected: PASS (3 tests).

- [ ] **Step 6: Commit**

```bash
git add docs/architecture.md CONTRIBUTING.md CONTRIBUTING.zh-CN.md tests/test_architecture_doc.py
git commit -m "docs: add public architecture.md and link it from CONTRIBUTING"
```

---

## Phase C — OSS hygiene + cleanup

### Task C2: Add `SECURITY.md`

**Files:**
- Create: `SECURITY.md`
- Test: `tests/test_security_policy.py`

- [ ] **Step 1: Write the failing test**

`tests/test_security_policy.py`:

```python
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 dengxuanliang
"""Guard: SECURITY.md exists and documents private reporting + the .env rule."""
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
SECURITY = ROOT / "SECURITY.md"


def test_security_policy_exists():
    assert SECURITY.is_file(), "SECURITY.md missing"


def test_security_policy_documents_private_reporting():
    text = SECURITY.read_text(encoding="utf-8")
    assert "GitHub Security Advisories" in text or "security advisory" in text.lower()


def test_security_policy_mentions_env_rule():
    text = SECURITY.read_text(encoding="utf-8")
    assert ".env" in text  # must remind that real keys never get committed
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_security_policy.py -v`
Expected: FAIL.

- [ ] **Step 3: Write `SECURITY.md`**

```markdown
# Security Policy

## Reporting a vulnerability

Please **do not** open a public GitHub issue for security vulnerabilities.

Instead, use **GitHub Security Advisories** ("Report a vulnerability" on the Security tab of this repo) so the maintainers are notified privately. Include:

- a description of the issue and its impact,
- steps to reproduce,
- affected versions / commits,
- any suggested fix.

We will acknowledge receipt within 5 business days and aim for a fix or mitigation within 30 days.

## Secrets

- `.env` is gitignored. **Never commit real API keys**, LiteLLM keys, or credentials.
- The tracked `.env.example` carries only placeholder values; if you find a real-looking key (`sk-…`) committed anywhere, treat it as a credential leak and report it as above.
- If you accidentally commit a secret, rotate it immediately — do not rely on `git rm` alone, since the blob remains in history.
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_security_policy.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add SECURITY.md tests/test_security_policy.py
git commit -m "docs: add SECURITY.md policy"
```

---

### Task C3: Add bilingual issue templates

**Files:**
- Create: `.github/ISSUE_TEMPLATE/bug_report.md`, `bug_report.zh-CN.md`, `feature_request.md`, `feature_request.zh-CN.md`, `config.yml`
- Test: `tests/test_github_templates.py`

- [ ] **Step 1: Write the failing test**

`tests/test_github_templates.py`:

```python
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 dengxuanliang
"""Guard: issue templates have valid YAML frontmatter; PR template exists."""
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parent.parent
ISSUE_DIR = ROOT / ".github" / "ISSUE_TEMPLATE"

EXPECTED_ISSUE_TEMPLATES = {
    "bug_report.md",
    "bug_report.zh-CN.md",
    "feature_request.md",
    "feature_request.zh-CN.md",
}


def _frontmatter(text: str) -> str:
    m = re.match(r"^---\n(.*?\n)---\n", text, re.DOTALL)
    assert m, "missing YAML frontmatter"
    return m.group(1)


def test_issue_templates_present():
    files = {p.name for p in ISSUE_DIR.glob("*.md")}
    missing = EXPECTED_ISSUE_TEMPLATES - files
    assert not missing, f"missing issue templates: {missing}"


def test_issue_templates_have_valid_frontmatter():
    for name in EXPECTED_ISSUE_TEMPLATES:
        text = (ISSUE_DIR / name).read_text(encoding="utf-8")
        fm = _frontmatter(text)
        assert re.search(r"^name:\s*\S", fm, re.MULTILINE), (
            f"{name} frontmatter missing 'name:'")


def test_issue_template_config_yaml_exists():
    cfg = ISSUE_DIR / "config.yml"
    assert cfg.is_file(), ".github/ISSUE_TEMPLATE/config.yml missing"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_github_templates.py -v`
Expected: FAIL.

- [ ] **Step 3: Write the templates**

`.github/ISSUE_TEMPLATE/bug_report.md`:

```markdown
---
name: Bug report
about: Report something that's broken
title: "[bug] "
labels: bug
---
## Summary
<!-- One or two sentences. -->

## Reproduce
```bash
# commands
```

## Expected
<!-- What you expected. -->

## Actual
<!-- What happened. Include tracebacks / Inspector event stream. -->

## Environment
- OS:
- Python: (`uv run python -V`)
- UXFlow commit:
- Embedding backend: Fake / Local / API
- `.env` relevant vars (redact keys):
```

`.github/ISSUE_TEMPLATE/bug_report.zh-CN.md`:

```markdown
---
name: 问题报告
about: 报告一个故障
title: "[bug] "
labels: bug
---
## 概要
<!-- 一两句话描述。 -->

## 复现步骤
```bash
# 命令
```

## 预期
<!-- 你期望发生什么。 -->

## 实际
<!-- 实际发生了什么。附上 traceback / Inspector 事件流。 -->

## 环境
- 操作系统:
- Python:（`uv run python -V`）
- UXFlow commit:
- Embedding 后端: Fake / Local / API
- `.env` 相关变量（请抹掉密钥）:
```

`.github/ISSUE_TEMPLATE/feature_request.md`:

```markdown
---
name: Feature request
about: Suggest a new capability
title: "[feat] "
labels: enhancement
---
## Problem
<!-- What's hard today? -->

## Proposal
<!-- What would make it easier? -->

## Alternatives considered
```

`.github/ISSUE_TEMPLATE/feature_request.zh-CN.md`:

```markdown
---
name: 功能请求
about: 建议一个新能力
title: "[feat] "
labels: enhancement
---
## 现状痛点
<!-- 今天哪里不好用？ -->

## 建议
<!-- 怎样会更顺手？ -->

## 已考虑的替代方案
```

`.github/ISSUE_TEMPLATE/config.yml`:

```yaml
blank_issues_enabled: false
contact_links:
  - name: Questions & discussions / 问答讨论
    url: https://github.com/dengxuanliang/uxflow/discussions
    about: Use Discussions for usage questions that aren't bugs or feature requests. / 非 bug、非功能请求的使用问题请走 Discussions。
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_github_templates.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add .github/ISSUE_TEMPLATE/ tests/test_github_templates.py
git commit -m "docs: add bilingual bug/feature issue templates + config"
```

---

### Task C4: Add bilingual PR template

**Files:**
- Create: `.github/PULL_REQUEST_TEMPLATE.md`
- Test: `tests/test_github_templates.py` (extend — same concern, same file)

**Context:** GitHub honors only one `PULL_REQUEST_TEMPLATE.md` (no chooser), so the bilingual content is inline rather than two separate files.

- [ ] **Step 1: Append failing guard to `tests/test_github_templates.py`**

```python
def test_pull_request_template_exists():
    prt = ROOT / ".github" / "PULL_REQUEST_TEMPLATE.md"
    assert prt.is_file(), "PULL_REQUEST_TEMPLATE.md missing"
    text = prt.read_text(encoding="utf-8")
    assert "pytest" in text.lower() and "ruff" in text.lower()
    assert "ruff" in text  # both EN checklist and zh-CN checklist reference it
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_github_templates.py::test_pull_request_template_exists -v`
Expected: FAIL.

- [ ] **Step 3: Write `.github/PULL_REQUEST_TEMPLATE.md`**

```markdown
## Summary / 概要
<!-- What & why. Reference any issue: Closes #N / 是什么与为什么。引用相关 issue: Closes #N -->

## Checklist / 自检清单
- [ ] `uv run ruff check` is clean / ruff 通过
- [ ] `uv run pytest -m "not requires_model"` is green / 纯逻辑测试通过
- [ ] New tests added for any new behavior; `@pytest.mark.requires_model` only when genuinely model-bound / 新行为已加测试; 仅在确需真实模型时才打 requires_model 标记
- [ ] No secrets, real keys, or `.env` content committed / 未提交密钥、真实 key 或 `.env` 内容
- [ ] SPDX header on new source files (`# SPDX-License-Identifier: Apache-2.0`) / 新源文件带 SPDX 头

## Notes / 备注
<!-- Anything reviewers should know. / 评审者需要知道的事项。 -->
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_github_templates.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add .github/PULL_REQUEST_TEMPLATE.md tests/test_github_templates.py
git commit -m "docs: add bilingual pull request template"
```

---

### Task C5: Drop the dead `onnx-embed` extra

**Files:**
- Modify: `pyproject.toml:19`
- Test: `tests/test_pyproject_extras.py`

**Context:** `grep -r "onnx\|Onnx" --include=*.py` over `src/` + `scripts/` + `tests/` returns **zero matches** (verified pre-plan). The extra is self-documented as "reserved: no OnnxEmbedder backend yet". A cloner-facing `pyproject.toml` should not advertise a non-existent option. **This is the only destructive change in the plan** (confirmed in discussion).

- [ ] **Step 1: Write the failing test**

`tests/test_pyproject_extras.py`:

```python
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 dengxuanliang
"""Guard: pyproject.toml advertises only extras that actually have backends."""
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
PYPROJECT = ROOT / "pyproject.toml"

LIVE_EXTRAS = {"dev", "service", "local-embed"}


def test_no_dead_extras():
    text = PYPROJECT.read_text(encoding="utf-8")
    assert "onnx-embed" not in text, (
        "onnx-embed extra has zero backends — drop it or implement it")


def test_live_extras_present():
    text = PYPROJECT.read_text(encoding="utf-8")
    for extra in LIVE_EXTRAS:
        assert extra in text, f"pyproject.toml missing live extra: {extra}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_pyproject_extras.py::test_no_dead_extras -v`
Expected: FAIL — `onnx-embed extra has zero backends`.

- [ ] **Step 3: Remove the line**

In `pyproject.toml`, delete line 19:

```toml
onnx-embed = ["onnxruntime>=1.17"]  # reserved: no OnnxEmbedder backend yet
```

Leave the surrounding `[project.optional-dependencies]` block otherwise unchanged:

```toml
[project.optional-dependencies]
local-embed = ["sentence-transformers>=3.0", "torch>=2.0"]
dev = ["pytest>=7.0", "pytest-asyncio>=0.23", "python-dotenv", "ruff"]
service = ["fastapi>=0.110", "uvicorn>=0.29", "python-multipart>=0.0.9"]
```

- [ ] **Step 4: Re-sync and run full guard suite + ruff + pytest**

Run: `uv sync --extra dev --extra service`
Run: `uv run ruff check`
Run: `uv run pytest -m "not requires_model" -q`
Expected: all green; `tests/test_pyproject_extras.py` fully passes.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml uv.lock tests/test_pyproject_extras.py
git commit -m "build: drop dead onnx-embed extra (no backend exists)"
```

---

## Phase D — Verification before claiming done

> **REQUIRED SUB-SKILL:** @superpowers:verification-before-completion — run every command below and confirm output before reporting complete.

- [ ] **Step 1: Full invariant suite (all per-concern files)**

Run: `uv run pytest tests/test_env_example.py tests/test_readme.py tests/test_architecture_doc.py tests/test_security_policy.py tests/test_github_templates.py tests/test_pyproject_extras.py -v`
Expected: all guards PASS.

- [ ] **Step 2: Pure-logic suite still green**

Run: `uv run pytest -m "not requires_model" -q`
Expected: green, no new failures vs. baseline.

- [ ] **Step 3: Lint clean**

Run: `uv run ruff check`
Expected: clean.

- [ ] **Step 4: CI matrix locally (one OS, all Pythons if installed)**

Run: `uv run --python 3.12 pytest -m "not requires_model" -q` (and 3.11/3.13 if available)
Expected: green.

- [ ] **Step 5: Cold-clone simulation**

```bash
# In a temp dir, simulate what a cloner sees:
git clone . /tmp/uxflow-cold-test
cd /tmp/uxflow-cold-test
uv sync --extra dev --extra service
cp .env.example .env            # confirm template is tracked + parseable
uv run pytest -m "not requires_model" -q
```
Expected: green; `.env.example` present; no missing-file errors.

- [ ] **Step 6: Final commit (if cold-clone surfaced any fix)**

Only if Step 5 surfaced something. Otherwise nothing to commit here.
