# SPDX-License-Identifier: Apache-2.0
import pathlib

from uxflow_paths import resolve_db_path


def test_cli_arg_wins(monkeypatch):
    monkeypatch.setenv("UXFLOW_DB", "/env/x.db")
    assert resolve_db_path("/cli/y.db") == pathlib.Path("/cli/y.db")


def test_env_over_xdg(monkeypatch):
    monkeypatch.setenv("UXFLOW_DB", "/env/x.db")
    monkeypatch.setenv("XDG_DATA_HOME", "/xdg")
    assert resolve_db_path(None) == pathlib.Path("/env/x.db")


def test_xdg_default(monkeypatch):
    monkeypatch.delenv("UXFLOW_DB", raising=False)
    monkeypatch.setenv("XDG_DATA_HOME", "/xdg")
    assert resolve_db_path(None) == pathlib.Path("/xdg/uxflow/uxflow.db")


def test_home_fallback(monkeypatch):
    monkeypatch.delenv("UXFLOW_DB", raising=False)
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)
    monkeypatch.setenv("HOME", "/home/u")
    assert resolve_db_path(None) == pathlib.Path("/home/u/.local/share/uxflow/uxflow.db")


def test_ensure_parent_creates_dir(tmp_path):
    from uxflow_paths import ensure_parent
    target = tmp_path / "sub" / "uxflow.db"
    ensure_parent(target)
    assert target.parent.is_dir()
