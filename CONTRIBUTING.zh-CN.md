[English](CONTRIBUTING.md) | [中文](CONTRIBUTING.zh-CN.md)

# 为 UXFlow 贡献

感谢你有兴趣参与贡献!

## 开发环境搭建

UXFlow 从源码安装。推荐使用 [uv](https://github.com/astral-sh/uv)。

```bash
uv sync --extra dev            # or: pip install -e ".[dev]"
```

### 修改顶层模块时的注意事项

绝大部分代码是可编辑安装：改动 `src/module0/`、`src/llm_gateway/`、
`src/service/` 等目录下的文件会立即生效。

但有三个文件是例外。`src/uxflow_runtime.py`、`src/uxflow_paths.py` 和
`scripts/uxflow_evolve.py` 在 `pyproject.toml` 里是 `force-include`，会被
**拷贝**进 `site-packages` 而非链接。改完其中任何一个后，普通的 `uv sync`
不会同步 —— 它看到 `pyproject.toml` 没变就什么都不做，于是你导入的仍是旧副本：

```bash
uv sync --extra dev --extra service --reinstall-package uxflow
```

典型症状是：刚加的符号报 `ImportError`，或者改动看起来完全没生效。

## 架构

模块地图与数据流图见 [`docs/architecture.md`](docs/architecture.md)。模块 0 ↔ 模块 1 的接口契约冻结于 [`docs/superpowers/specs/interface-contract.md`](docs/superpowers/specs/interface-contract.md)。

## 运行测试

测试通过 `requires_model` 这个 pytest marker,按是否需要真实嵌入模型进行划分。

- **纯逻辑测试**(推荐默认):无需任何 ML 依赖,在 CI 中覆盖 Linux/macOS × Python 3.11–3.13:

  ```bash
  uv run pytest -m "not requires_model"
  ```

- **完整套件**:需要 `local-embed` extra 并会下载 Qwen 模型:

  ```bash
  uv sync --extra local-embed
  uv run pytest
  ```

没有 GPU 或模型的贡献者应运行 `-m "not requires_model"` 子集。

## 测试约定

- 纯逻辑测试应注入 `FakeEmbedder`(来自 `uxflow_embed`),而不是真实模型,以保持快速且无依赖。
- 确实需要真实模型的测试必须标记 `@pytest.mark.requires_model`。

## 代码风格

Lint 必须通过:

```bash
uv run ruff check
```

## PR 流程

1. Fork 仓库并创建特性分支。
2. 完成改动并附带测试。
3. 确保两者都为绿:
   ```bash
   uv run pytest -m "not requires_model"
   uv run ruff check
   ```
4. 向 `main` 提交 PR。CI 必须通过。

## 许可证

贡献以 Apache-2.0 授权。新增源文件应携带 SPDX 头:

```python
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 dengxuanliang
```
