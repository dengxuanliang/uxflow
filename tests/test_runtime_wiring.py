# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 dengxuanliang
"""Guard: the documented default path needs no ML deps and fails loudly when unconfigured.

These tests exist because keyword-only README guards let three "clone and run"
regressions through: hardcoded LocalEmbedder in the entry scripts, forced
HF offline mode, and the legacy model alias driving compile.
"""
import pathlib
import re

import pytest

from uxflow_runtime import make_embedder, require_llm_config, resolve_models

ROOT = pathlib.Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
ENV_EXAMPLE = ROOT / ".env.example"

# Scripts that must run on the documented zero-ML default path.
ZERO_ML_ENTRY_SCRIPTS = ("e2e_smoke.py", "inspector_serve.py")


def test_default_backend_is_fake_and_ml_free(monkeypatch):
    monkeypatch.delenv("UXFLOW_EMBED_BACKEND", raising=False)
    emb = make_embedder()
    assert type(emb).__name__ == "FakeEmbedder"
    assert len(emb.embed("hello")) == emb.dimension


def test_explicit_fake_backend_is_case_insensitive(monkeypatch):
    monkeypatch.setenv("UXFLOW_EMBED_BACKEND", "  FAKE  ")
    assert type(make_embedder()).__name__ == "FakeEmbedder"


def test_unknown_backend_exits_with_guidance(monkeypatch):
    monkeypatch.setenv("UXFLOW_EMBED_BACKEND", "bogus")
    with pytest.raises(SystemExit, match="bogus"):
        make_embedder()


@pytest.mark.parametrize("missing", ["LITELLM_BASE", "LITELLM_KEY"])
def test_require_llm_config_names_the_missing_key(monkeypatch, missing):
    monkeypatch.setenv("LITELLM_BASE", "http://x/v1")
    monkeypatch.setenv("LITELLM_KEY", "k")
    monkeypatch.setenv(missing, "")
    with pytest.raises(SystemExit, match=missing):
        require_llm_config()


def test_require_llm_config_returns_values_when_set(monkeypatch):
    monkeypatch.setenv("LITELLM_BASE", "http://x/v1")
    monkeypatch.setenv("LITELLM_KEY", "secret")
    assert require_llm_config() == ("http://x/v1", "secret")


def test_legacy_alias_drives_judge_not_compile(monkeypatch):
    """MODULE0_TEST_MODEL must never select the compile model (needs strict JSON)."""
    monkeypatch.delenv("UXFLOW_COMPILE_MODEL", raising=False)
    monkeypatch.delenv("UXFLOW_JUDGE_MODEL", raising=False)
    monkeypatch.setenv("MODULE0_TEST_MODEL", "gpt-4o-mini")
    compile_model, judge_model = resolve_models()
    assert compile_model == "gpt-5.5"
    assert judge_model == "gpt-4o-mini"


def test_explicit_model_env_wins(monkeypatch):
    monkeypatch.setenv("UXFLOW_COMPILE_MODEL", "c-model")
    monkeypatch.setenv("UXFLOW_JUDGE_MODEL", "j-model")
    monkeypatch.setenv("MODULE0_TEST_MODEL", "legacy")
    assert resolve_models() == ("c-model", "j-model")


def test_entry_scripts_do_not_force_offline():
    """HF_HUB_OFFLINE=1 blocks the first model download for fresh cloners."""
    for script in SCRIPTS.glob("*.py"):
        text = script.read_text(encoding="utf-8")
        assert not re.search(r'setdefault\(\s*["\']HF_HUB_OFFLINE', text), (
            f"{script.name} forces offline mode — breaks first-run model download")


def test_env_example_does_not_enable_offline_by_default():
    text = ENV_EXAMPLE.read_text(encoding="utf-8")
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        assert not re.search(rf"^\s*{key}\s*=", text, re.MULTILINE), (
            f".env.example must keep {key} commented out — it blocks the "
            "first model download")


def test_env_example_documents_embed_backend():
    text = ENV_EXAMPLE.read_text(encoding="utf-8")
    assert re.search(r"^\s*UXFLOW_EMBED_BACKEND\s*=\s*fake", text, re.MULTILINE), (
        ".env.example must default UXFLOW_EMBED_BACKEND to fake")


def test_entry_scripts_do_not_hardcode_local_embedder():
    for name in ZERO_ML_ENTRY_SCRIPTS:
        text = (SCRIPTS / name).read_text(encoding="utf-8")
        assert "from module0.embedding import EmbeddingModel" not in text, (
            f"{name} must use make_embedder(), not a hardcoded LocalEmbedder")
        assert "make_embedder" in text, f"{name} must build its embedder via make_embedder()"


def test_api_backend_prefers_litellm_credentials(monkeypatch):
    """LITELLM_KEY/BASE win, so switching to api needs no new .env entries."""
    monkeypatch.setenv("UXFLOW_EMBED_BACKEND", "api")
    monkeypatch.setenv("LITELLM_BASE", "http://lit.local/v1")
    monkeypatch.setenv("LITELLM_KEY", "lit-key")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")

    emb = make_embedder()
    assert "lit.local" in emb._url
    assert "lit-key" in emb._client.headers["Authorization"]


def test_api_backend_falls_back_to_openai_key(monkeypatch):
    monkeypatch.setenv("UXFLOW_EMBED_BACKEND", "api")
    monkeypatch.delenv("LITELLM_BASE", raising=False)
    monkeypatch.delenv("LITELLM_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "openai-fallback")

    emb = make_embedder()
    assert "openai-fallback" in emb._client.headers["Authorization"]


def test_api_backend_defaults(monkeypatch):
    """Defaults are large/3072/batch-32 -- the gateway-safe combination."""
    monkeypatch.setenv("UXFLOW_EMBED_BACKEND", "api")
    monkeypatch.setenv("LITELLM_KEY", "k")
    monkeypatch.delenv("UXFLOW_EMBED_API_MODEL", raising=False)
    monkeypatch.delenv("UXFLOW_EMBED_API_DIM", raising=False)
    monkeypatch.delenv("UXFLOW_EMBED_BATCH", raising=False)
    monkeypatch.delenv("UXFLOW_EMBED_RATELIMIT_BASE", raising=False)
    monkeypatch.delenv("UXFLOW_EMBED_RATELIMIT_CAP", raising=False)

    emb = make_embedder()
    assert emb.dimension == 3072
    assert emb.preferred_batch_size == 32
    # 429 backoff defaults: an Azure tier quota window is ~60s, so 2s doubling
    # up to 32s gives the window a real chance to reopen within max_tries.
    assert emb._rate_limit_base == 2.0
    assert emb._rate_limit_cap == 32.0


def test_api_backend_respects_tunable_env_vars(monkeypatch):
    monkeypatch.setenv("UXFLOW_EMBED_BACKEND", "api")
    monkeypatch.setenv("LITELLM_KEY", "k")
    monkeypatch.setenv("UXFLOW_EMBED_BATCH", "64")
    monkeypatch.setenv("UXFLOW_EMBED_TIMEOUT", "10")
    monkeypatch.setenv("UXFLOW_EMBED_MAX_TRIES", "3")
    monkeypatch.setenv("UXFLOW_EMBED_RATELIMIT_BASE", "4.0")
    monkeypatch.setenv("UXFLOW_EMBED_RATELIMIT_CAP", "16.0")

    emb = make_embedder()
    assert emb.preferred_batch_size == 64
    assert emb._timeout == 10.0
    assert emb._max_tries == 3
    assert emb._rate_limit_base == 4.0
    assert emb._rate_limit_cap == 16.0


def test_api_backend_without_any_key_exits_with_guidance(monkeypatch):
    """Missing credentials must name BOTH accepted variables."""
    monkeypatch.setenv("UXFLOW_EMBED_BACKEND", "api")
    monkeypatch.delenv("LITELLM_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(SystemExit) as exc:
        make_embedder()
    msg = str(exc.value)
    assert "LITELLM_KEY" in msg
    assert "OPENAI_API_KEY" in msg


def test_entry_scripts_guard_llm_config():
    """An empty key must fail up front, not as 'Call 1 failed after retries'.

    Validation now lives inside make_gateway()'s real path, so entry scripts
    satisfy this by routing through it rather than calling require_llm_config
    themselves — building an LLMGateway directly would bypass the check.
    """
    for name in ZERO_ML_ENTRY_SCRIPTS:
        text = (SCRIPTS / name).read_text(encoding="utf-8")
        assert ("make_gateway" in text or "require_llm_config" in text), (
            f"{name} must validate LLM config before building the gateway")
        assert "LLMGateway(" not in text, (
            f"{name} must build its gateway via make_gateway(), which enforces "
            "config validation and honours UXFLOW_LLM_BACKEND")
        assert 'os.environ.get("LITELLM_KEY"' not in text, (
            f"{name} should read the key via require_llm_config(), not directly")
