[English](README.md) | [中文](README.zh-CN.md)

# UXFlow

面向 SFT 数据集治理的 SWE 轨迹优选器。

## 这是什么

UXFlow 通过筛选高质量的 SWE agent 轨迹切片来治理 SFT(监督微调)训练数据。它运行一条五阶段流水线:

- **module0** —— 查询编译(LLM 从自然语言投诉编译出 `ProblemSpec`)
- **module1** —— 召回 + 评判(切片、签名、索引轨迹;按子问题 RRF 召回 + LLM 评判)
- **module2** —— 相关性重排(对召回切片按各子问题做 soft-score)
- **module3** —— 去重 + 子模选择(跨问题去重、覆盖优化的最终挑选)
- **module0_5** —— 标签自演化(向共享 taxonomy 提议/回填新的能力标签;CLI:`uxflow-evolve`)

`llm_gateway` 提供各阶段共用的自适应 LLM 调用网关;`uxflow_embed` 以可插拔方式支撑嵌入(Fake / 本地 Qwen / API)。可选的 `service` 包通过 FastAPI Inspector UI 暴露同一条流水线。模块地图与数据流参见 [`docs/architecture.md`](docs/architecture.md)。

## 环境要求

- Python 3.11–3.13
- Linux 或 macOS(Windows 未经测试)
- 一个 [LiteLLM](https://docs.litellm.ai/) 代理端点(OpenAI 兼容),用于 LLM 调用

## 安装(从源码)

UXFlow 从源码安装(未发布到 PyPI)。推荐使用 [uv](https://github.com/astral-sh/uv)。

```bash
git clone https://github.com/dengxuanliang/uxflow
cd uxflow
uv sync --extra dev --extra service   # 或:pip install -e ".[dev,service]"
cp .env.example .env                 # 然后编辑 .env,填入 LITELLM_BASE / LITELLM_KEY
```

核心安装非常轻量(httpx、datasketch、numpy),无需任何 ML 依赖。`service` extra 拉取 FastAPI + uvicorn 以运行 Inspector UI。如需启用可选的本地嵌入后端(通过 sentence-transformers + torch 运行 Qwen):

```bash
uv sync --extra local-embed           # 或:pip install -e ".[local-embed]"
```

## 快速上手

```bash
# 1. 配置(一次性)
cp .env.example .env
${EDITOR:-vi} .env                    # 设置 LITELLM_BASE + LITELLM_KEY

# 2. 在捆绑的 fixtures 上跑端到端 smoke(只需 LLM,无需 torch):
uv run python scripts/e2e_smoke.py "写入py文件有语法错误"

# 3. 或启动 Inspector web UI:
uv run python scripts/inspector_serve.py
#    然后打开 http://127.0.0.1:8000
```

该 smoke 脚本加载 `fixtures/taxonomy_v0.json` + `fixtures/trajectories/sample_01.jsonl`,端到端运行 module 0 → 1 → 2 → 3 → 0.5,并在每个阶段打印中间结果供人工检查。

## 数据目录

module 0.5(标签自演化)持久化到单个 SQLite 文件。默认位置遵循 XDG 规范:`~/.local/share/uxflow/uxflow.db`。可用 `UXFLOW_DB` 环境变量或 `uxflow-evolve` 的 `--db` 参数覆盖。该数据库从不纳入 git。

## 嵌入后端

嵌入能力通过 `uxflow_embed` 包中统一的 `Embedder` 协议实现可插拔:

- **FakeEmbedder** —— 确定性输出,无 ML 依赖。用于测试。
- **LocalEmbedder** —— 本地 Qwen 嵌入模型,需要 `local-embed` extra。
- **ApiEmbedder** —— 调用 OpenAI 兼容的嵌入端点。

核心安装不携带任何 ML 依赖:请使用 `ApiEmbedder`(远程端点)或 `FakeEmbedder`。仅当你想在本地运行 Qwen 时才安装 `local-embed` extra。

## 测试

```bash
uv run pytest -m "not requires_model"   # 纯逻辑,无 ML 依赖

uv sync --extra local-embed             # 运行完整套件前需先安装
uv run pytest                           # full suite (loads the Qwen model)
```

`-m "not requires_model"` 子集运行纯逻辑,零 ML 依赖。完整套件需要 `local-embed` extra 并会下载 Qwen 模型。

## 贡献

参见 [CONTRIBUTING.zh-CN.md](CONTRIBUTING.zh-CN.md)。模块地图与数据流图参见 [`docs/architecture.md`](docs/architecture.md)。

## 许可证

Apache-2.0。参见 [LICENSE](LICENSE)。
