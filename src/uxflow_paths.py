"""DB path resolution: --db arg > UXFLOW_DB env > XDG data dir > ~/.local/share."""
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 dengxuanliang

from __future__ import annotations

import os
import pathlib

__all__ = ["resolve_db_path", "ensure_parent"]

_APP = "uxflow"
_DB_NAME = "uxflow.db"


def resolve_db_path(cli_arg: str | None = None) -> pathlib.Path:
    """Resolve the SQLite DB path by precedence: cli_arg > env > XDG > home."""
    if cli_arg:
        return pathlib.Path(cli_arg)
    env = os.environ.get("UXFLOW_DB")
    if env:
        return pathlib.Path(env)
    xdg = os.environ.get("XDG_DATA_HOME")
    if xdg:
        return pathlib.Path(xdg) / _APP / _DB_NAME
    home = pathlib.Path(os.environ.get("HOME", str(pathlib.Path.home())))
    return home / ".local" / "share" / _APP / _DB_NAME


def ensure_parent(path: pathlib.Path) -> None:
    """Create the DB's parent directory if missing (mkdir -p)."""
    path.parent.mkdir(parents=True, exist_ok=True)
