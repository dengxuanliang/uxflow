# UXFlow 开源 + 跨平台 落地设计规格

> **状态**：设计定稿，待评审。
> **范围**：把 UXFlow 从"本机可跑的私有项目"改造为"能在 GitHub 上被找到、clone、跨 Linux/macOS 跑起来、接受外部 PR"的开源项目。**不上 PyPI**（仅源码安装）。
> **上游依据**：`interface-contract.md`（§6 Embedding 配置一致性约束）、现有 `pyproject.toml`、`src/module0/embedding.py`。
> **前置事实**：模块 0/1/2/3 已实现并全部通过（209 passed）。当前包名 `llm-gateway`，仓库名 `UXFlow`，二者不一致。`sentence-transformers`（拖入 torch）是核心依赖。无 README/LICENSE/CI。

---

## 0. 目标与非目标

**目标**：别人能在 GitHub 上 ① 找到（README/许可证/元数据齐全）、② clone、③ 跨 Linux/macOS × Python 3.11-3.13 跑起来（含测试）、④ 提 PR（贡献流程清晰、CI 自动验证）。

**非目标（本次不做）**：
- **不上 PyPI**——仅源码安装（`git clone` + `uv sync` / `pip install -e .`）。
- 不支持 Windows（CI 不测，README 不承诺）。
- 不追求覆盖所有 embedding provider（API 后端给一个参考实现）。
- 不实现 ONNX 后端（仅协议预留）。

---

## 1. 贯穿性设计原则

沿用项目既定原则：V1 实现为最终产品级架构预留可扩展接口余量，把会变化的部分收敛到接口边界。本设计新增的 `Embedder` 协议与既有 `SliceStore` / `TaxonomyStore` 同构。

---

## 2. 已确认决策

| # | 决策点 | 选定 |
|---|--------|------|
| 许可证 | 开源许可 | **Apache-2.0**（含专利授权，AI/ML 主流） |
| 包名 | 品牌统一 | **uxflow**（`llm_gateway` 降为子模块，物理目录不改） |
| 核心策略 | 跨平台难点（本地模型分发） | 抽象 **`Embedder` 协议**（embed / embed_batch / dimension） |
| 后端 | Embedder 实现档位 | V1：**假 embedder + 本地模型（可选依赖）+ API 参考实现**；ONNX 仅协议预留 |
| 依赖 | 分层 | torch/sentence-transformers 降为可选组 `[local-embed]`；numpy 提为核心 |
| Python | 版本范围 | **3.11 - 3.13** |
| CI | 平台矩阵 | **macOS + Linux** × 3.11/3.12/3.13，只跑不依赖 ML 的纯逻辑测试 |
| 分发 | 安装方式 | **仅源码安装（clone）**，不上 PyPI |
| 文档 | 设计文档可见性 | `docs/superpowers/specs` 与 `plans` **随仓库公开**（开源透明加分） |
| 锁定 | 依赖复现 | 提交 **`uv.lock`**（应用型项目标准做法） |

---

## 3. `Embedder` 协议 + 后端实现

### 3.1 协议（事实接口极小，探查确认全库只用这些）

```python
# src/uxflow_embed/protocol.py（命名实现时可调）
@runtime_checkable
class Embedder(Protocol):
    @property
    def dimension(self) -> int: ...
    def embed(self, text: str) -> list[float]: ...
    def embed_batch(self, texts: list[str]) -> list[list[float]]: ...
```

### 3.2 后端实现（分档）

| 实现 | 依赖 | 状态 | 说明 |
|------|------|------|------|
| `FakeEmbedder` | 无 | **V1（复用现有）** | **提升现有 `tests/integration/complex_smoke_runner.py:154` 的 `DeterministicEmbeddingModel`** 为包内正式类，供全测试注入。确定性 hash→归一化 1024-d 向量 |
| `LocalEmbedder` | `[local-embed]`（torch+ST） | V1（迁移） | 现有 `src/module0/embedding.py::EmbeddingModel` 迁移，**去掉 `local_files_only=True`**（改为自动下载）；懒加载 import |
| `ApiEmbedder` | `httpx`（核心已有） | V1（参考实现） | OpenAI 兼容格式（Qwen API 亦支持此格式）；key 从环境变量读 |
| `OnnxEmbedder` | `[onnx-embed]` | 仅协议预留 | YAGNI，等有明确需求再实现 |

### 3.3 关键设计点

- **下游零改动**：探查确认 `pipeline.py:31` 已是 `object | None`，signature/compiler 全按鸭子类型调用，无 import 具体类型。抽协议只需新建 + 让实现声明 `Embedder`。
- **契约 §6 约束升级为协议层保证**：同一 `Embedder` 实例注入，`dimension` 可校验；语义（1024-d、L2 归一化）不变。
- **`FakeEmbedder` 是开源协作地基**：让模块 1/2/3/0.5 全部纯逻辑测试脱离 torch（详见 §5）。
- **`local_files_only=True` 是开源分发头号障碍**（现 `embedding.py:35`）：clone 者本机无模型文件会直接失败。改为默认自动下载，离线用户可另配本地路径参数。

---

## 4. 依赖分层 + `pyproject.toml` 重构 + 命名

### 4.1 命名统一

```toml
[project]
name = "uxflow"
description = "SWE trajectory selector for SFT dataset curation"
requires-python = ">=3.11"
license = "Apache-2.0"
readme = "README.md"
authors = [{name = "dengxuanliang"}]
urls = {Repository = "https://github.com/dengxuanliang/uxflow", Issues = "https://github.com/dengxuanliang/uxflow/issues"}
```

`llm_gateway` 作为子模块保留（`src/llm_gateway/` 物理目录不改，避免大量 import 改动）；仅改分发包名与品牌。

### 4.2 依赖分层（核心改动）

```toml
dependencies = [          # 核心：轻量，纯逻辑 + API embedder + 假 embedder 可跑
    "httpx>=0.27",
    "datasketch>=1.6",    # 模块3 dedup 必需，纯 Python，保留核心
    "numpy",              # ⚠ 必须显式提核心：见 §4.3
]

[project.optional-dependencies]
local-embed = ["sentence-transformers>=3.0", "torch>=2.0"]
onnx-embed  = ["onnxruntime>=1.17"]      # 预留，V1 不实现
dev = ["pytest>=7.0", "pytest-asyncio>=0.23", "python-dotenv", "ruff"]
```

**安装体验分层**：
- `pip install -e .` / `uv sync` → 轻装：纯逻辑（模块1/2/3/0.5）+ API embedder + 假 embedder
- `uv sync --extra local-embed` → 加本地 Qwen 推理
- `uv sync --extra dev` → 贡献者开发环境

### 4.3 关键设计点

- **`numpy` 必须显式提核心**：现在它是 sentence-transformers 的传递依赖；一旦 ST 移到可选组，轻装用户就没 numpy，而 `index.py` 直接 `import numpy` 会崩。这是"移动依赖"最易漏的坑。
- **`datasketch` 留核心**：模块3 dedup 的 MinHash 层刚需，纯 Python 无重依赖。
- **`torch>=2.0` 不锁变体**：让 pip/uv 按平台解析（macOS 拿 MPS/CPU）。
- **懒加载守卫**：`LocalEmbedder` 的 `import torch`/`sentence_transformers` 保持在方法内（现 `embedding.py:24-25` 已是）；未装 `[local-embed]` 的用户 import uxflow 不报错，仅实例化本地后端时才要求依赖，缺失则给清晰报错（提示 `pip install -e ".[local-embed]"`）。

---

## 5. 测试脱离 ML（让 clone 者能跑、CI 能跑）

### 5.1 当前是已破状态（升级为核心动机）

探查确认这些测试**直接实例化真实 Qwen**（`local_files_only=True`）：
- `tests/module0/test_embedding.py:23` → `EmbeddingModel()`
- `tests/module0/test_integration.py:76` → `EmbeddingModel()`
- `scripts/e2e_smoke.py:79`、`scripts/complex_smoke_report.py:71`

**后果**：别人 clone 后第一次 `pytest` 就红一片（本机无模型文件），误以为自己弄坏了——这是"提 PR"的头号劝退点。这不是锦上添花，是当前就破的准入门槛。

### 5.2 改造

1. **纯逻辑测试注入 `FakeEmbedder`**：模块1/2/3/0.5 的测试用假 embedder 替代真实模型（它们只关心向量如何排序/去重/召回，不关心向量语义）。改造后零 ML 依赖，全平台可跑。
2. **pytest marker 分层**：真正测试 Qwen 行为的（`test_embedding.py` 本身）打 `@pytest.mark.requires_model`；CI 用 `-m "not requires_model"` 跳过，装了 `[local-embed]` 的开发者可全跑。
3. **凡 `EmbeddingModel()` 直接调用点**全部改为注入（测试注入 Fake，脚本可选真实）。

### 5.3 CI 不测本地模型（刻意）

真实 Qwen 推理慢、要下模型、设备差异大，放 CI 脆弱且是负担。CI 保证"纯逻辑跨 macOS/Linux × 3 版本正确"；本地模型正确性靠开发者本地 + `requires_model` 手动全量跑。

---

## 6. uv 锁定 + CI 矩阵

### 6.1 uv + `uv.lock`

- 引入 `uv`（快速、可复现）；产出 **`uv.lock` 提交进仓库**。
- **⚠ 修正内部冲突**：现有 `.gitignore` 有一行 `uv.lock` 忽略项；本设计要求**从 `.gitignore` 移除该行**——应用型开源项目应锁定并提交 `uv.lock`，供他人复现。
- hatchling 仍作 build backend（uv 管环境/依赖，hatchling 管打包）。

### 6.2 CI（`.github/workflows/ci.yml`）

```yaml
strategy:
  matrix:
    os: [ubuntu-latest, macos-latest]     # 不含 Windows
    python: ["3.11", "3.12", "3.13"]      # 6 组合
steps:
  - uv sync --extra dev                     # 轻装，无 torch
  - uv run pytest -m "not requires_model"   # 纯逻辑测试
  - uv run ruff check
```

---

## 7. 开源工程文件 + 仓库卫生

### 7.1 必备文件（仓库根）

| 文件 | 内容 |
|------|------|
| `LICENSE` | Apache-2.0 全文 |
| `README.md` | 项目定位、**源码安装指引**（clone + `uv sync` / `pip install -e .`，含 `[local-embed]` 分层）、快速上手、跨平台矩阵（Linux/macOS × 3.11-3.13；Windows 未测）、架构图链接。英文为主 |
| `CONTRIBUTING.md` | 开发环境（`uv sync --extra dev`）、测试约定（`requires_model` marker、纯逻辑测试注入 Fake）、代码风格（ruff）、PR 流程 |
| `pyproject.toml` 元数据 | `authors` / `urls` / `readme`（GitHub 场景无需 PyPI 专用 classifiers，精简即可） |
| `.github/workflows/ci.yml` | §6.2 矩阵 |

- 每个源文件加 Apache-2.0 许可证头（Apache 惯例，实现时批量加）。

### 7.2 公开前泄密体检（原设计缺失，补入）

开源到公开仓库前必做的一次性检查（本次探查结果：**当前干净**，但须作为显式关卡防实现阶段误提交）：
- `.env` / `.DS_Store` / `.claude/` 均**未被 git 跟踪**（`.gitignore` 已覆盖）✓
- git 历史无 `.env` 提交、跟踪文件无明文密钥 ✓
- **实现阶段守则**：改 `.gitignore`、`git add` 时勿纳入 `.env`；发布前再跑一次 `git ls-files | grep -iE "env|secret|key"` 复核。

### 7.3 设计文档可见性（已定：公开）

- `docs/superpowers/specs` 与 `plans` 随仓库公开（透明加分）。
- `.gitignore` 现忽略 `docs/superpowers/reports/` 与 `SWE轨迹优选器设计.md` → 保持忽略（reports 是过程性/含预期对照，中文原始设计稿不公开）。

---

## 8. 落地顺序（分批增量验证）

1. **第 1 批 — Embedder 抽象 + 测试解耦**（纯代码）：建协议 + `FakeEmbedder`（复用现有 Deterministic）+ `LocalEmbedder`（迁移，去 local_files_only）+ `ApiEmbedder`；纯逻辑测试注入 Fake + 打 `requires_model` marker。**完成后全部纯逻辑测试脱离 torch。**
2. **第 2 批 — 依赖分层 + 命名**：`pyproject.toml` 重构（核心/可选组、numpy 提核心、uxflow）、懒加载守卫、`uv.lock`（并从 .gitignore 移除）。
3. **第 3 批 — CI + 工程文件**：CI 矩阵、LICENSE、README、CONTRIBUTING、许可证头、泄密体检。

（原第 4 批"发布 PyPI"已删除。）

---

## 9. 端到端验证

- **测试解耦**：在无 torch 的干净环境 `uv sync --extra dev`（不带 local-embed）→ `pytest -m "not requires_model"` 全绿。
- **本地后端仍可用**：装 `[local-embed]` → `pytest`（含 requires_model）全绿。
- **CI 矩阵**：macOS + Linux × 3.11/3.12/3.13 六格全绿。
- **clone 冒烟**：模拟他人——干净目录 clone → `uv sync` → 跑纯逻辑测试通过（验证"clone 即可跑"）。
- **泄密体检**：`git ls-files` 无 `.env`/密钥/session 文件。

---

## 10. 开放点处置（已确认）

1. **`ApiEmbedder` provider 格式**：**OpenAI 兼容**（`/v1/embeddings` 格式；Qwen API / vLLM / Ollama 等大多兼容，一个实现覆盖多数 provider）。
2. **README/CONTRIBUTING 语言**：**中英双语**（`README.md` 英文 + `README.zh-CN.md` 中文；CONTRIBUTING 同理）。
3. **`llm_gateway` 物理目录**：**不重命名**，仅改分发包名为 uxflow（零 import 改动）。
4. **项目元数据（真实值）**：
   - 作者署名：`dengxuanliang`
   - 仓库 URL：`https://github.com/dengxuanliang/uxflow`
   - Apache 许可证头版权行：`Copyright 2026 dengxuanliang`
   - `pyproject.toml`：`authors = [{name = "dengxuanliang"}]`，`urls = {Repository = "https://github.com/dengxuanliang/uxflow", Issues = "https://github.com/dengxuanliang/uxflow/issues"}`
