[English](CONTRIBUTING.md) | [中文](CONTRIBUTING.zh-CN.md)

# Contributing to UXFlow

Thanks for your interest in contributing!

## Dev setup

UXFlow is installed from source. We recommend [uv](https://github.com/astral-sh/uv).

```bash
uv sync --extra dev            # or: pip install -e ".[dev]"
```

## Running tests

Tests are split by whether they need the real embedding model, using the `requires_model` pytest marker.

- **Pure-logic tests** (recommended default) need no ML dependencies and run in CI across Linux/macOS × Python 3.11–3.13:

  ```bash
  uv run pytest -m "not requires_model"
  ```

- **Full suite** requires the `local-embed` extra and downloads the Qwen model:

  ```bash
  uv run pytest
  ```

Contributors without a GPU or the model should run the `-m "not requires_model"` subset.

## Test conventions

- Pure-logic tests should inject `FakeEmbedder` (from `uxflow_embed`) rather than a real model, so they stay fast and dependency-free.
- Tests that genuinely need the real model must be marked `@pytest.mark.requires_model`.

## Code style

Lint must pass:

```bash
uv run ruff check
```

## PR process

1. Fork the repo and create a feature branch.
2. Make your change with tests.
3. Ensure both are green:
   ```bash
   uv run pytest -m "not requires_model"
   uv run ruff check
   ```
4. Open a PR against `main`. CI must pass.

## License

Contributions are licensed under Apache-2.0. New source files should carry the SPDX header:

```python
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 dengxuanliang
```
