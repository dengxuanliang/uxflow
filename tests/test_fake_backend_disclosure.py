# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 dengxuanliang
"""Guard: selecting a fake backend is always disclosed, and never the default.

The whole value of the fake backend is that it lets a fresh clone run with no
proxy. The whole risk is a user believing those results came from a real model.
These tests pin down both halves so the disclosure can't be quietly dropped.
"""
import pathlib
import subprocess
import sys

import pytest

from uxflow_runtime import backend_banner, llm_backend, make_gateway

ROOT = pathlib.Path(__file__).resolve().parent.parent


@pytest.fixture
def clean_env(monkeypatch):
    monkeypatch.delenv("UXFLOW_LLM_BACKEND", raising=False)
    monkeypatch.delenv("UXFLOW_EMBED_BACKEND", raising=False)


# ── The default must be real ────────────────────────────────────────────────

def test_llm_backend_defaults_to_real(clean_env):
    """Unset env must not silently downgrade to scripted replies."""
    assert llm_backend() == "real"


def test_default_gateway_requires_config(clean_env, monkeypatch):
    """With no LLM config, the default path exits — it does not fall back to fake."""
    monkeypatch.setenv("LITELLM_BASE", "")
    monkeypatch.setenv("LITELLM_KEY", "")
    with pytest.raises(SystemExit):
        make_gateway()


def test_unknown_backend_is_rejected(monkeypatch):
    monkeypatch.setenv("UXFLOW_LLM_BACKEND", "sortof")
    with pytest.raises(SystemExit):
        llm_backend()


# ── Fake must be disclosed ──────────────────────────────────────────────────

def test_fake_llm_is_announced(monkeypatch):
    monkeypatch.setenv("UXFLOW_LLM_BACKEND", "fake")
    monkeypatch.setenv("UXFLOW_EMBED_BACKEND", "local")
    banner = backend_banner()
    assert "FAKE" in banner
    assert "UXFLOW_LLM_BACKEND=fake" in banner


def test_fake_embed_is_announced(monkeypatch):
    monkeypatch.setenv("UXFLOW_LLM_BACKEND", "real")
    monkeypatch.setenv("UXFLOW_EMBED_BACKEND", "fake")
    banner = backend_banner()
    assert "FAKE" in banner
    assert "UXFLOW_EMBED_BACKEND=fake" in banner


def test_banner_always_reports_both_backends(monkeypatch):
    """Half-fake runs are the easy mistake; naming one backend would hide them."""
    monkeypatch.setenv("UXFLOW_LLM_BACKEND", "fake")
    monkeypatch.setenv("UXFLOW_EMBED_BACKEND", "local")
    banner = backend_banner()
    assert "UXFLOW_LLM_BACKEND=" in banner
    assert "UXFLOW_EMBED_BACKEND=" in banner


def test_all_real_banner_carries_no_warning(monkeypatch):
    monkeypatch.setenv("UXFLOW_LLM_BACKEND", "real")
    monkeypatch.setenv("UXFLOW_EMBED_BACKEND", "local")
    banner = backend_banner()
    assert "FAKE" not in banner
    assert "✓" in banner


# ── The disclosure reaches actual stdout ────────────────────────────────────

def test_smoke_script_prints_banner_to_stdout(tmp_path, monkeypatch):
    """End-to-end: the banner must survive into the real script's output."""
    env = {
        **dict(__import__("os").environ),
        "UXFLOW_LLM_BACKEND": "fake",
        "UXFLOW_EMBED_BACKEND": "fake",
        "UXFLOW_DB": str(tmp_path / "smoke.db"),
    }
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "e2e_smoke.py"), "写入py文件有语法错误"],
        capture_output=True, text=True, env=env, timeout=300,
    )
    assert result.returncode == 0, f"fake run failed:\n{result.stdout}\n{result.stderr}"
    # Printed at startup AND after the results, which is where it actually gets read.
    assert result.stdout.count("FAKE 后端") >= 2, (
        "fake disclosure missing from smoke output")


# ── The service exposes backends so the web UI can warn ─────────────────────

def test_stats_endpoint_reports_backends(monkeypatch):
    monkeypatch.setenv("UXFLOW_LLM_BACKEND", "fake")
    monkeypatch.setenv("UXFLOW_EMBED_BACKEND", "fake")

    from fastapi.testclient import TestClient

    from service import MemoryRunStore, PipelineDeps, create_app

    deps = PipelineDeps(
        compiler=None, pipeline=None, select_fn=None, load_trajectories_fn=None,
        problem_store=None, trajectory_store=None, judge_cache=None, embedder=None,
    )
    app = create_app(store=MemoryRunStore(), deps=deps)
    with TestClient(app) as client:
        body = client.get("/stats").json()
    assert body["llm_backend"] == "fake"
    assert body["embed_backend"] == "fake"


def test_web_ui_renders_the_backend_warning():
    """The terminal banner is invisible in a browser; the UI must warn too."""
    app_js = (ROOT / "src" / "service" / "web" / "app.js").read_text(encoding="utf-8")
    index = (ROOT / "src" / "service" / "web" / "index.html").read_text(encoding="utf-8")
    assert "fake-banner" in index, "no fake-backend banner element in the UI"
    assert "llm_backend" in app_js and "embed_backend" in app_js, (
        "app.js does not read the backend fields from /stats")
