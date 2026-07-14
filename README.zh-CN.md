[English](README.md) | [中文](README.zh-CN.md)

# UXFlow

面向 SFT 数据集治理的 SWE 轨迹优选器。

## 这是什么

UXFlow 通过筛选高质量的 SWE agent 轨迹切片来治理 SFT(监督微调)训练数据。它运行一条四阶段流水线:

- **module0** —— 查询编译
- **module1** —— 召回 + 评判
- **module2** —— 相关性重排
- **module3** —— 去重 + 子模选择

(`llm_gateway` 提供各阶段共用的自适应 LLM 调用网关。)架构细节参见 [`docs/`](docs/)。

## 环境要求

- Python 3.11–3.13
- Linux 或 macOS(Windows 未经测试)

## 安装(从源码)

UXFlow 从源码安装(未发布到 PyPI)。推荐使用 [uv](https://github.com/astral-sh/uv)。

```bash
git clone https://github.com/dengxuanliang/uxflow
cd uxflow
uv sync --extra dev            # or: pip install -e ".[dev]"
```

核心安装非常轻量(httpx、datasketch、numpy),无需任何 ML 依赖。如需启用可选的本地嵌入后端(通过 sentence-transformers + torch 运行 Qwen):

```bash
uv sync --extra local-embed    # or: pip install -e ".[local-embed]"
```

## 数据目录

模块 0.5(标签自演化)持久化到单个 SQLite 文件。默认位置遵循 XDG 规范:`~/.local/share/uxflow/uxflow.db`。可通过 `UXFLOW_DB` 环境变量或 `uxflow-evolve` 的 `--db` 参数覆盖。该数据库从不纳入 git 跟踪。

## 嵌入后端

嵌入能力通过 `uxflow_embed` 包中统一的 `Embedder` 协议实现可插拔:

- **FakeEmbedder** —— 确定性输出,无 ML 依赖。用于测试。
- **LocalEmbedder** —— 本地 Qwen 嵌入模型,需要 `local-embed` extra。
- **ApiEmbedder** —— 调用 OpenAI 兼容的嵌入端点。

核心安装不携带任何 ML 依赖:请使用 `ApiEmbedder`(远程端点)或 `FakeEmbedder`。仅当你想在本地运行 Qwen 时才安装 `local-embed` extra。

## 测试

```bash
uv run pytest -m "not requires_model"   # pure-logic, no ML deps

uv sync --extra local-embed             # 运行完整套件前需先安装
uv run pytest                           # full suite (loads the Qwen model)
```

`-m "not requires_model"` 子集运行纯逻辑,零 ML 依赖。完整套件需要 `local-embed` extra 并会下载 Qwen 模型。

## 贡献

参见 [CONTRIBUTING.zh-CN.md](CONTRIBUTING.zh-CN.md)。

## 许可证

Apache-2.0。参见 [LICENSE](LICENSE)。
