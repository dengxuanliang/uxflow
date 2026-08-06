[English](README.md) | [中文](README.zh-CN.md)

# UXFlow

面向 SFT 数据集治理的 SWE 轨迹优选器。

## 这是什么

UXFlow 通过筛选高质量的 SWE agent 轨迹切片来治理 SFT（监督微调）训练数据。它运行一条五阶段流水线:

- **module0** —— 查询编译（LLM 把自然语言抱怨编译成 `ProblemSpec`）
- **module1** —— 召回 + 评判（切片、签名、建索引；RRF 召回 + LLM 逐子问题精判）
- **module2** —— 相关性重排（对召回切片按子问题软加分）
- **module3** —— 去重 + 子模选择（跨子问题去重，覆盖最优最终挑选）
- **module0_5** —— 标签自演化（向共享词表提议/回填新能力标签；CLI: `uxflow-evolve`）

`llm_gateway` 提供各阶段共用的自适应 LLM 调用网关；`uxflow_embed` 可插拔嵌入（Fake / Local Qwen / API）。可选的 `service` 包通过 FastAPI Inspector UI 暴露同一条流水线。模块地图与数据流见 [`docs/architecture.md`](docs/architecture.md)。

## 环境要求

- Python 3.11–3.13
- Linux 或 macOS（Windows 未经测试）
- 跑 real 模式需要：一个 OpenAI 兼容的 LLM 端点，经由 [LiteLLM](https://docs.litellm.ai/) 代理接入（[步骤 1](#步骤-1--起一个-llm-代理) 会带你起一个）

## 安装

UXFlow 从源码安装（未发布到 PyPI）。推荐使用 [uv](https://github.com/astral-sh/uv)。

```bash
git clone https://github.com/dengxuanliang/uxflow
cd uxflow
uv sync --extra dev --extra service
```

核心安装非常轻量（httpx、datasketch、numpy、python-dotenv），**不带任何 ML 依赖**。`service` extra 额外拉取 FastAPI + uvicorn 用于 Inspector UI。

---

## 快速上手 —— 现在就跑通整条流水线

无需代理、无需密钥、无需 ML 依赖。一条命令:

```bash
UXFLOW_LLM_BACKEND=fake UXFLOW_EMBED_BACKEND=fake \
  uv run python scripts/e2e_smoke.py "写入py文件有语法错误"
```

### 你应该看到什么

开头会声明当前生效的后端，然后依次走完五个阶段:

```
⏳ 构建 embedding backend (fake)...
✓ Embedding 就绪 (dim=1024)
╔════════════════════════════════════════════════════════════════╗
║  ⚠️  FAKE 后端 — 本次运行的结果不可用于真实数据筛选             ║
╠════════════════════════════════════════════════════════════════╣
║  UXFLOW_LLM_BACKEND=fake                                        ║
║  UXFLOW_EMBED_BACKEND=fake                                      ║
...
╚════════════════════════════════════════════════════════════════╝

  模块0: 编译 ProblemSpec          → 产出 2 条子问题
  模块1: 切片 + 签名 + 建索引
  模块1: 召回 + 精判
  模块2/3: 软加分 + 去重覆盖优选   → 最终 targeted: 6
  模块0.5: ①提议捕获 → ②入库 → ③继承 → ④回填 → ③升权
  Gateway 统计                     → fake backend: no HTTP traffic
```

退出码 `0`，且结束时会**再打印一次**同样的告警横幅。

### 这证明了什么、没证明什么

✅ 安装正确、五个阶段能连通、SQLite 持久化经得起重启。

❌ **完全不能说明结果质量。** LLM 回答是固定回放，嵌入向量是确定性哈希。`最终 targeted: 6` 走的是真实代码路径，但选出来的东西没有意义。用它验证管道是否通，绝不要用它判断产出好坏。

这也是为什么每次 fake 运行都会**打两遍**横幅、**同时列出两个**后端，Inspector UI 还会常驻告警 —— 就是为了让 fake 结果永远不会被误当成真实结果。

---

## 切到 real 模式

从上面那次运行到真实运行，有四件事要改。请**按顺序**执行 —— 每一步都带校验，让问题暴露在它发生的地方，而不是三步之后。

> **两个独立开关。** `UXFLOW_LLM_BACKEND` 和 `UXFLOW_EMBED_BACKEND` 是分开的。只切一个会得到"半真"运行：判得对但召回排序是噪声，或者反过来。**两个都切成真的，结果才有意义** —— 启动横幅永远同时打印两者，方便你一眼核对。

### 步骤 1 —— 起一个 LLM 代理

UXFlow 的 LLM 调用走标准 OpenAI 兼容的 `/chat/completions` 端点。可经 LiteLLM 代理接入（方式 B/C），也可直连供应商（方式 A）。若你已有 LiteLLM 代理，跳过本步。

**方式 A —— 直连供应商（不跑 LiteLLM 代理）:**

若你的供应商本身是 OpenAI 兼容（OpenAI / DeepSeek / Moonshot / OpenRouter 等），跳过本步的代理，直接让 `.env` 指向供应商——无需 docker/uvx。此方式下 `LITELLM_KEY` 就填供应商的 key（不是 `sk-local-dev`），并必须用 `UXFLOW_COMPILE_MODEL` / `UXFLOW_JUDGE_MODEL` 覆盖默认别名 `gpt-5.5` / `gpt-4o-mini`（多数供应商没有 `gpt-5.5`）。具体见[步骤 2](#步骤-2--配置-env) 的直连列。**注意：跳过的只是 litellm 代理，不是后续步骤** —— Step 2–5 对直连同用户同样适用，只是把「代理」读作「你的供应商端点」。

**方式 B —— 用 uv（不依赖 Docker）:**

```bash
cp litellm.config.example.yaml litellm.config.yaml
${EDITOR:-vi} litellm.config.yaml          # 把 litellm_params.model 指向你的真实上游

export OPENAI_API_KEY=sk-...               # 你的 provider 密钥
export LITELLM_MASTER_KEY=sk-local-dev

uvx --from 'litellm[proxy]==1.95.0' --with 'fastapi<0.140.7' \
  litellm --config litellm.config.yaml --port 4000
```

> `fastapi<0.140.7` 这个约束是**必需的**：0.140.7 移除了 `get_flat_dependant`，而 litellm 的 proxy 仍在 import 它，且 litellm 自己没声明上限。不加约束的话代理会在启动时直接挂掉 —— 而且报错具有误导性，显示为 `ModuleNotFoundError: No module named 'proxy_server'`，因为它的 CLI 把真正的 ImportError 吞掉了。想看真实原因，执行 `python -c "import litellm.proxy.proxy_server"`。
>
> 若看到 `failed to fetch remote model cost map ... falling back to local backup` 警告，**无害** —— litellm 没能联网取到价格表，改用了内置副本。想消掉它：`export LITELLM_LOCAL_MODEL_COST_MAP=True`。

**方式 C —— 用 Docker:**

```bash
cp litellm.config.example.yaml litellm.config.yaml
export OPENAI_API_KEY=sk-...
docker compose -f docker-compose.litellm.yml up -d
```

**预期结果:** 出现 uvicorn 启动日志，监听 4000 端口。**保持它运行**，另开一个终端做后续步骤。

> ⚠️ **两个 `model_name` 必须与 UXFlow 请求的名字一致** —— 编译用 `gpt-5.5`、裁决用 `gpt-4o-mini`。对不上会在很久之后以误导性的 `Call 1 failed after retries` 暴露。若要用别的名字，请把 `UXFLOW_COMPILE_MODEL` / `UXFLOW_JUDGE_MODEL` 设成对应值。

### 步骤 2 —— 配置 `.env`

```bash
cp .env.example .env
```

设置这三项:

| 变量 | 值 | 说明 |
|---|---|---|
| `LITELLM_BASE` | `http://localhost:4000/v1` | 已是默认值 |
| `LITELLM_KEY` | `sk-local-dev` | 必须等于步骤 1 的 `LITELLM_MASTER_KEY` |
| `UXFLOW_LLM_BACKEND` | `real` | 已是默认值 |

默认值取 `real` 是刻意的：缺少凭据时它会**直接报错**，而不是静默退化成回放。

**方式 A（直连供应商）改填:**

| 变量 | 值 | 说明 |
|---|---|---|
| `LITELLM_BASE` | `https://你的供应商/v1` | 供应商的 base URL |
| `LITELLM_KEY` | 你的供应商 key | 即供应商 API key，非 `sk-local-dev` |
| `UXFLOW_COMPILE_MODEL` | 供应商真实模型名 | 覆盖别名 `gpt-5.5`（编译，需严格 JSON） |
| `UXFLOW_JUDGE_MODEL` | 供应商真实模型名 | 覆盖别名 `gpt-4o-mini`（裁决） |

### 步骤 3 —— 验证你的 LLM 端点确实能答

```bash
uv run python scripts/check_model_split.py
```

这一步只隔离一个问题：两个模型名能否在端点处给出应答。因此这里失败一定是连通性或命名问题，绝不会是业务逻辑问题。

**预期结果:**

```
=== result ===
  compile (gpt-5.5): PASS
  judge   (gpt-4o-mini): PASS

✓ both models answered — split is live.
```

退出码 `0`。**失败时脚本会告诉你怎么读**：`404-ish / empty` → 端点没认这个模型名（直连=供应商没有该模型；代理=该模型名没在 `litellm.config.yaml` 里注册）；`timeout / conn` → `LITELLM_BASE` 指向的端点（供应商 / 代理）不可达。

**这一步不通过，不要往下走。**

### 步骤 4 —— 安装真实嵌入后端

```bash
uv sync --extra local-embed
```

这会拉取 `torch` + `sentence-transformers`（体积很大），并在首次运行时从 HuggingFace 下载 `Qwen/Qwen3-Embedding-0.6B`。如果 HuggingFace 对你很慢或不通，见[受限网络环境](#受限网络环境)。

若你还想开 Inspector UI，同时 `uv sync --extra service`（见[安装](#安装)）。

不想在本地跑模型？改用托管的嵌入端点：设 `UXFLOW_EMBED_BACKEND=api`，详见[嵌入后端](#嵌入后端)。

### 步骤 5 —— 真实运行

```bash
UXFLOW_EMBED_BACKEND=local UXFLOW_DB=~/.local/share/uxflow/real.db \
  uv run python scripts/e2e_smoke.py "写入py文件有语法错误"
```

**预期结果 —— 与 fake 运行的三处差异:**

1. **没有告警框**。取而代之是一行:
   ```
   ✓ LLM: real   ✓ Embedding: local
   ```
   **看不到告警本身就是确认。** 如果框还在，说明有一个后端没切过来 —— 框里会写明是哪个。

2. **结尾统计里有真实 HTTP 流量:**
   ```
   {'total_http_requests': 12, 'success': 12, 'errors': {}, 'success_rate': 100.0}
   ```
   fake 运行这里显示的是 `fake backend: no HTTP traffic`。**这是"确实发出了网络请求"最硬的证据。**

3. **结果开始有意义** —— relevance 分数反映的是语义相似度，而不是哈希噪声。

> ⚠️ **切换后端时，请把 `UXFLOW_DB` 指向一个全新的数据库。** SQLite 里持久化的向量与后端绑定；复用 fake 那次的库会让哈希向量和真实向量被静默地放在一起比较。

**如果模块0 报 `产出 0 条子问题` 且全部被 Dropped**，说明你的 compile 模型没能输出严格 JSON。这是模型能力问题而非 bug —— 把 `UXFLOW_COMPILE_MODEL` 换成指令遵循更强的模型。

---

## Inspector web UI

同一条流水线的浏览器界面，通过 SSE 实时展示进度:

**需先 `uv sync --extra service`**（拉取 FastAPI + uvicorn，见[安装](#安装)）；否则会报 `No module named uvicorn`。

```bash
uv run python scripts/inspector_serve.py     # 然后打开 http://127.0.0.1:8000
```

它遵循同样的后端开关。在 fake 后端下，页头会常驻一条告警横幅 —— 因为终端里的横幅对使用浏览器的人完全不可见。

## 配置参考

所有配置都由环境变量驱动，[`.env.example`](.env.example) 里记录了全部变量。最关键的几个:

| 变量 | 默认值 | 用途 |
|---|---|---|
| `LITELLM_BASE` | `http://localhost:4000/v1` | LiteLLM 代理端点 |
| `LITELLM_KEY` | *(空)* | 代理 API key —— real 模式必填 |
| `UXFLOW_LLM_BACKEND` | `real` | `real` \| `fake` |
| `UXFLOW_EMBED_BACKEND` | `fake` | `fake` \| `local` \| `api` |
| `UXFLOW_COMPILE_MODEL` | `gpt-5.5` | 编译 ProblemSpec，需严格 JSON 输出 |
| `UXFLOW_JUDGE_MODEL` | `gpt-4o-mini` | 裁决切片，调用量最大 |
| `UXFLOW_DB` | `~/.local/share/uxflow/uxflow.db` | SQLite 路径（XDG） |

### 嵌入后端

嵌入能力通过 `uxflow_embed` 中统一的 `Embedder` 协议实现可插拔:

- **`fake`**（默认）—— 确定性输出，无 ML 依赖。向量不带语义。
- **`local`** —— 经 sentence-transformers 跑 Qwen3-Embedding-0.6B。需要 `local-embed` extra，首次使用时下载模型。
- **`api`** —— 任何 OpenAI 兼容的嵌入端点。需要 `OPENAI_API_KEY`，可用 `UXFLOW_EMBED_API_MODEL` / `_DIM` / `_BASE` 调整。

### 数据目录

模块 0.5 持久化到单个 SQLite 文件，默认走 XDG 路径 `~/.local/share/uxflow/uxflow.db`。可用 `UXFLOW_DB` 或 `uxflow-evolve --db` 覆盖。该文件从不纳入 git 跟踪。

## 受限网络环境

UXFlow 本身除了你的 LLM 端点之外不依赖任何外部服务，但它用到的三个**工具**会从境外基础设施拉取内容，在某些地区可能很慢或不通。每一个都有镜像方案:

| 步骤 | 拉取来源 | 应对方式 |
|---|---|---|
| `uv sync` / `uvx` | PyPI | `export UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple` |
| `uv` 准备 Python 解释器 | GitHub Releases | `export UV_PYTHON_INSTALL_MIRROR=<GitHub release 代理>`，或自行装一个 ≥3.11 的 Python |
| `UXFLOW_EMBED_BACKEND=local` | HuggingFace | `export HF_ENDPOINT=https://hf-mirror.com` |
| `docker compose ... litellm` | Docker Hub | 改用[步骤 1](#步骤-1--起一个-llm-代理) 的**方式 B** —— 它走 PyPI |

如果你前面有一层做 TLS 拦截的代理，Docker 会报 `x509: certificate signed by unknown authority`，**即使 `curl` 访问同一个域名是通的** —— curl 从系统信任库里认得那个代理的 CA，Docker daemon 认不得。要么把该 CA 装进 daemon 的信任库并**重启 daemon**（这一步最容易漏），要么直接走方式 B。

在 Qwen 模型真正落盘之前，请保持 `HF_HUB_OFFLINE` / `TRANSFORMERS_OFFLINE` 处于注释状态 —— 过早开启会阻断它们本想跳过的那次下载。

### 嵌入模型下载卡住：让 Python 信任拦截代理的 CA

`model.safetensors` 卡在 0% 不动、但 `curl https://huggingface.co` 是通的——多半是前面有一层 TLS 拦截代理：它对**元数据域**（huggingface.co）放行，却对**下载域**（`*.cdn.hf.co`，权重文件走 CDN）解密重签。Python 的 `ssl` 不认代理的私有 CA，于是到 CDN 的握手失败、下载根本起不来（0%，没有 `.incomplete` 文件）。先诊断，确认后再修复。

**诊断**

```bash
# 1. 先确认是"零字节"，不是下到一半断了：没有 *.incomplete 也没有 safetensors
find ~/.cache/huggingface/hub/models--Qwen--Qwen3-Embedding-0.6B \
  -name '*.safetensors' -o -name '*.incomplete'
# （输出为空 = 下载从未开始，不是被中断）

# 2. 取下载域主机名（resolve 会 302 到 CDN）+ 代理 host:port
CDN_HOST=$(curl -sI -o /dev/null -w '%{redirect_url}\n' \
  https://huggingface.co/Qwen/Qwen3-Embedding-0.6B/resolve/main/model.safetensors \
  | sed -E 's#https?://([^/]+).*#\1#')
echo "下载域: $CDN_HOST"   # 形如 us.aws.cdn.hf.co / *.hf.co
PROXY=${https_proxy:-$HTTPS_PROXY}; PROXY=${PROXY#http://}; PROXY=${PROXY#https://}; PROXY=${PROXY%%/*}
[ -n "$PROXY" ] && PROXY_FLAG=(-proxy "$PROXY") || PROXY_FLAG=()

# 3. 元数据域 = huggingface.co。issuer 是公共 CA（DigiCert / Amazon / Let's Encrypt）→ 代理对它透传
openssl s_client -connect huggingface.co:443 -servername huggingface.co "${PROXY_FLAG[@]}" </dev/null 2>/dev/null \
  | openssl x509 -noout -issuer

# 4. 下载域 = $CDN_HOST。issuer 不是公共 CA（自签 / 厂商 CA）→ 代理在解密下载域
openssl s_client -connect "$CDN_HOST":443 -servername "$CDN_HOST" "${PROXY_FLAG[@]}" </dev/null 2>/dev/null \
  | openssl x509 -noout -issuer
```

第 3 步公共 CA + 第 4 步私有厂商 CA → 确认是只对下载域解密的 TLS 拦截代理，按下面修复。（两步都是公共 CA → 不是证书问题，改用 `export HF_ENDPOINT=https://hf-mirror.com` 走镜像。）

**修复：让 Python 信任代理的 CA**

```bash
# 1. 导出下载域的完整证书链（叶子证书 + 代理的自签根 CA）
#    必须连下载域 $CDN_HOST，不是 huggingface.co（元数据域没被解密，链它拿不到代理 CA）
mkdir -p ~/.local/share/uxflow
CDN_HOST=$(curl -sI -o /dev/null -w '%{redirect_url}\n' \
  https://huggingface.co/Qwen/Qwen3-Embedding-0.6B/resolve/main/model.safetensors \
  | sed -E 's#https?://([^/]+).*#\1#')
PROXY=${https_proxy:-$HTTPS_PROXY}; PROXY=${PROXY#http://}; PROXY=${PROXY#https://}; PROXY=${PROXY%%/*}
[ -n "$PROXY" ] && PROXY_FLAG=(-proxy "$PROXY") || PROXY_FLAG=()
openssl s_client -showcerts -connect "$CDN_HOST":443 -servername "$CDN_HOST" "${PROXY_FLAG[@]}" </dev/null 2>/dev/null \
  | sed -n '/-----BEGIN CERTIFICATE-----/,/-----END CERTIFICATE-----/p' \
  > ~/.local/share/uxflow/proxy-ca.pem

# 2. 组合 CA 包 = 系统 CA + 代理证书链
#    不要单独用 proxy-ca.pem —— 它只有代理的 CA，单独用会连不上普通 HTTPS 站点
#    系统 CA：macOS 从 keychain 导，Linux 用发行版的 ca-certificates
case "$(uname -s)" in
  Darwin)
    SYS_CA=~/.local/share/uxflow/sys-ca.pem
    security find-certificate -a -p /Library/Keychains/System.keychain > "$SYS_CA"
    security find-certificate -a -p /System/Library/Keychains/SystemRootCertificates.keychain >> "$SYS_CA"
    ;;
  Linux)
    SYS_CA=/etc/ssl/certs/ca-certificates.crt                              # Debian/Ubuntu/Fedora
    [ -f /etc/pki/tls/certs/ca-bundle.crt ] && SYS_CA=/etc/pki/tls/certs/ca-bundle.crt   # RHEL/CentOS
    ;;
esac
cat "$SYS_CA" ~/.local/share/uxflow/proxy-ca.pem > ~/.local/share/uxflow/combined-ca.pem

# 3. 让 Python 指向组合 CA 包（真正起作用的是 SSL_CERT_FILE）
#    huggingface_hub 1.x 用 `httpx` 下载（不是 `requests`）；httpx 在 trust_env=True（默认）时
#    读 SSL_CERT_FILE / SSL_CERT_DIR，不读 REQUESTS_CA_BUNDLE / CURL_CA_BUNDLE。
#    实测：只设后两个仍报 [SSL: CERTIFICATE_VERIFY_FAILED]。保留 REQUESTS_CA_BUNDLE 只为照顾其他用 `requests`/`curl` 的工具。
export SSL_CERT_FILE=~/.local/share/uxflow/combined-ca.pem
export REQUESTS_CA_BUNDLE=~/.local/share/uxflow/combined-ca.pem

# 4. 带上 CA 包重跑第 5 步
UXFLOW_EMBED_BACKEND=local UXFLOW_DB=~/.local/share/uxflow/real.db \
  uv run python scripts/e2e_smoke.py "写入py文件有语法错误"
```

下完一次后这套环境变量可以撤掉（模型已落盘到 `~/.cache/huggingface/hub`）；之后想避网可再开 `HF_HUB_OFFLINE=1`。

## 测试

```bash
uv run pytest -m "not requires_model"   # 纯逻辑，无 ML 依赖

uv sync --extra local-embed             # 运行完整套件前需先安装
uv run pytest                           # 完整套件（加载 Qwen 模型）
```

`-m "not requires_model"` 子集就是 CI 在 Linux/macOS × Python 3.11–3.13 上跑的那套。

## 贡献

参见 [CONTRIBUTING.zh-CN.md](CONTRIBUTING.zh-CN.md)。模块地图与数据流图见 [`docs/architecture.md`](docs/architecture.md)。

## 许可证

Apache-2.0。参见 [LICENSE](LICENSE)。
