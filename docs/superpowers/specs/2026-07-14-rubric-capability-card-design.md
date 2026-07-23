# Rubric 判据卡 + 失败轨迹证据接入 + 双模型接力 —— 设计文档

日期：2026-07-14（2026-07-21 增补失败轨迹证据接入节）
状态：设计定稿，待实现（分 PR-1/2/3 落地）
作者：dengxuanliang

---

## Context

### 要解决的真实问题

由问题清单编译出的 `capability_label`（如 `avoid_redundant_repetition`）与最终选入 SFT
数据集的 loss mask span 片段之间**关系不紧密**。典型症状：召回到的片段"含有相关上下文"，
但 mask 圈住的是工具输出 dump、环境返回、平淡过渡步，而非真正演示该能力的决策 token。

拿去做 SFT，模型学到的是"看到超时就说 Found it"这类**伪相关**，而非能力本身。

### 根因（方案层，非调参可救）

现在这条鸿沟由：`hyde_positive`（一两句理想行为）嵌入 → 相似度召回 → 一个**宽松的
judge**（"判是否演示 + 圈 start/end 范围"）弥合。三个通用缺陷：

1. **判定太糊**：judge 圈的是"步骤范围"，天然把中间的工具输出、过渡步全包进来。
2. **门槛太低**："含有相关上下文"就算 match，擦边的也进来。
3. **无落点约束**：没有机制区分"要学的决策 token"和"只作背景的环境输出"。

这三点与"是哪个能力"无关 → 所以着力点是**通用**的，不针对某一种 capability 手写检测器
（明确否决 `has_repetition_risk` 这类能力特定信号：N 种能力要写 N 个检测器，不 scale，
本质是把 judge 的工作硬编码进管线）。

### 核心思路

把"相似度一步匹配"拆成职责清晰的三段，每段能力无关：

| 环节 | 职责 | 手段 | 通用性来源 |
|---|---|---|---|
| 召回 | 别漏（宽进） | 现有 hyde+BM25，**不动** | 本来就通用 |
| 判定 | 严判 + 精定位 | judge 对着 **rubric** 核对，指不出决定性证据就 false | rubric 是 LLM 生成的 |
| 落点 | 只留要学的 | mask 只圈 assistant 决策/推理，排除工具输出 | 通用原则 |

唯一"因能力而异"的东西是一张 **LLM 自动生成的行为判据卡（rubric）**，不是手写规则。
**管线代码里零能力特定分支** —— 这是通用性的关键保证。

---

## 设计决策与前置讨论结论（冻结，勿推翻）

这些是经过多轮质疑后拍定的结论，实现时必须遵守：

1. **rubric 锚定人写的原始描述，不从标签名凭空生成**
   rubric 必须从 `raw_text` / `failure_summary`（用户原始问题清单那条具体描述）蒸出来，
   而非从 `avoid_redundant_repetition` 这种抽象标签名自由联想。原始描述是非 LLM 生成的锚，
   打破"从标签名凭空生成判据"的循环。

2. **rubric 的价值是"把隐藏假设外置成可审计文本"，不是"保证判据正确"**
   不存在能力的 ground truth，无法验证判据"绝对准确"。正确目标是：让判据错的时候**能被
   发现**，而非静默自洽。显式写出判据不增加错误，只是把本就存在的错误暴露出来供人抽审。

3. **双模型接力打断自洽环，但接力 ≠ 验证**
   Opus 生成 rubric、GPT-5.5 执行 judge —— 判据作者与判官分权，打断"同一个脑子自说自话"
   的自洽环。但这仍是**异任务接力**，产生不了"分歧"信号，故**不构成验证**。它买到的是
   "更可信的单次判决 + 落点更准"，不买"质量被验证"。

4. **diff 交叉验证（真正的验证）本次不做**
   真正的交叉验证是独立的审计活动：让第二个势均力敌、异家族的强模型对同一批片重判，
   与生产 judge 结果 diff，分歧率 = rubric 欠规范程度。此为**未来**工作，本次系统尚未
   完善，不引入以免"整崩"。

5. **人审判据卡，而非人审海量片段**
   一个能力一张卡（几十张），片段成千上万。人工审判据卡的边际成本极低，是引入 rubric
   真正省下的东西。

---

## 数据结构改动（契约层，最先定）

### CapabilityRubric（放 `src/module0/schema.py`，与 SubProblem 同源）

```
@dataclass(frozen=True)
CapabilityRubric:
    positive_criteria:  list[str]   # 1-3 条可观测的"正向展示"判据
    negative_criteria:  list[str]   # 1-3 条反例/失败模式
    decisive_evidence:  str         # 决定性证据的"形态"描述（在哪类步骤、什么信号）
    capability_kind:    str         # "presence" | "avoidance" | "recovery"
```

字段职责（对应三个不满）：
- `positive_criteria` / `negative_criteria`：给 judge 明文对照物 → 解决"擦边也 match"。
- `decisive_evidence`：描述证据**形态**而非硬编码检测器 → 解决"落点是机械中段"。
  **禁止**写成"检测 X 工具连续调用 N 次"这类能力特定规则。
- `capability_kind`：让 judge 知道该找哪类证据。**唯一的分型信号，只影响 judge 判读
  视角，不进任何 if-else 管线分支**。三型定义：
  - `presence`（有痕）：有正向文本/结构标记，如 writes_test_first。找正向痕迹本身。
  - `avoidance`（无痕）：正例=坏模式的缺席，如 avoid_redundant_repetition。找"本可犯错
    的锚点 + 其后没犯错的枢轴步"，mask 圈枢轴。
  - `recovery`（转折）：error → 正确处置，如 recover_from_test_failure。mask 圈处置步。

新增枚举（冻结）：`CAPABILITY_KINDS = frozenset({"presence", "avoidance", "recovery"})`

### SubProblem 加字段（`schema.py:58`）

```
rubric: CapabilityRubric | None = None
```

- **可空**：符合"留产品级接口余量"；保证向后兼容 —— 老 ProblemSpec 与三个 smoke 脚本
  走的老路径，`rubric=None` 时 judge 自动 fallback 到现有 prompt，不炸。
- **契约同步（硬要求）**：`schema.py:1-4` 明写"do NOT add/rename without updating the
  contract document"；SubProblem docstring 说"12 required fields"。加 rubric 前**必须
  同步改** `docs/superpowers/specs/interface-contract.md` §1.2，否则契约漂移。

### JudgeResult 加字段（`src/module1/models.py:62`）

```
match: bool
confidence: float
spans: list[dict]
reasoning: str = ""
evidence_step: int | None = None            # 新：决定性证据在第几步
criteria_hit: list[str] = field(default_factory=list)  # 新：命中 rubric 哪几条 positive_criteria
```

这两字段是可回溯性关键：每个 match 都能追溯"命中哪条判据 + 证据在第几步"，供人工抽查、
并让下游 mask 收窄有依据。

### 可回溯字段的传播链（比 judge_cache 更前置的断流点，勿漏）

**⚠️ 最易漏、影响最大**：judge 产出的 `evidence_step` / `criteria_hit` 要真正到达人审界面，
先过的不是 cache，而是 **`rerank()` 这道映射**。两条**生产路径**（`_score_sub_problem` 与
`_score_sub_problem_cached`）最后都调 `rerank()` 把 `JudgeResult` 摊平成 `ScoredCandidate`，
而现状**只搬 `confidence / spans / match` 三样，新字段在这里当场被丢**。cache 那 5 个点补得
再全，水管下游（rerank）堵着，字段照样到不了 `/runs`。故这条链**必须与 judge_cache 平级**
纳入 PR-3：

- `module2/models.py:12-26` `ScoredCandidate`：**加 `evidence_step` / `criteria_hit` 成员**
  （带默认值，向后兼容）。现状无此二字段 → 没有落脚点。
- `module2/rerank.py:38-49` `rerank()` 构造 `ScoredCandidate` 处：现只映射
  `judge_confidence=jr.confidence / loss_mask_spans=jr.spans / judge_match=jr.match` →
  **补搬 `jr.evidence_step` / `jr.criteria_hit`**。这是 /runs 与 search **两条路径共用**的
  瓶颈,改一处两路都通。
- `service/viewmodel.py:64-77`：InspectorView 从 `ScoredCandidate` 只读
  `judge_match / loss_mask_spans` → **surface 两个新字段**，否则人审界面看不到证据步，
  成功判据 #3 交付不出来。

**范围校正**：judge_cache 那节（下文）只覆盖 **search 路径**（唯一走 cache 的路径）；
`/runs` 生产路径（`run_scored → _score_sub_problem`）**根本不碰 judge_cache**，它的可回溯性
**完全**靠上面这条 rerank 链。两条链缺一，对应路径的可回溯性即为零。

### 序列化透传

rubric 从 module0 流到 module1 的唯一通道，漏一处就断在半路：

- `orchestrator.py:33` `_spec_to_dict`（函数定义行；手工 `return {` 字段列表在 `:38`）：
  **加 rubric 序列化**（None → null）。
- `parsing.py` `parse_call2_response`（required 元组在 `parsing.py:119`）：rubric 作为
  **可选**字段解析，**不进 required**（避免老响应解析失败）。

### JudgeResult 新字段的持久化（judge cache，仅 search 路径，勿漏）

`evidence_step` / `criteria_hit` 不仅要在**新判**时产出，还要在**缓存命中**时保留，
否则 **search 路径**（唯一走 cache 的路径）二次查询会丢掉可回溯性。注意作用域：这节只管
search；`/runs` 生产路径不走 cache，它的可回溯性靠上文 rerank 传播链，别把两者混为一谈。

**关键：judge_cache 是列式存储（不是 JSON blob），字段要显式穿过下面 5 个点，缺任一处新
字段就在那一层静默丢。尤其：只给 `CREATE TABLE` 加列毫无作用 —— 真正读写的 SQL 在
`get()/put()` 方法里，DDL 改了 DML 没改，等于加了两列永远没人写、没人读。** 按数据流列全：

写路径：
- `pipeline.py:373` `judge_cache.put(...)` 调用点：**构造**的 verdict dict 只放
  `match/confidence/spans` → 补进两个新字段（否则 store 层根本收不到值可写）。
- `stores.py:279-291` `JudgeCacheStore.put()`：`INSERT` 列清单 +`VALUES` 元组只取
  `verdict["match"/"confidence"/"spans"]` → **加列 + 加取值**。**这才是"补写"的主战场**，
  上面 `pipeline.py:373` 只是备好 dict。
- `stores.py:62-68` `CREATE TABLE judge_cache`：仅 `matched / confidence / spans_json` 三列
  → 加两列（或复用一个 json 列存扩展字段）。**迁移解法必须写死，别只靠 `CREATE TABLE IF
  NOT EXISTS`** —— 该语句对**已存在的旧库不会加列**，新 `SELECT` 会抛 `no such column`。
  确定方案：启动时对旧库跑幂等 `ALTER TABLE judge_cache ADD COLUMN ...`（`try/except
  OperationalError` 吞"duplicate column"），或引入 schema 版本号做一次性迁移。这正是本节
  警惕的"DDL 没生效"在**旧库**上的翻版。

读路径：
- `stores.py:259-269` `JudgeCacheStore.get()`：`SELECT matched, confidence, spans_json` +
  只把这三个 key 映射成 dict → **SELECT 补列 + 返回 dict 补 key**（否则读出来永远没有新字段）。
- `pipeline.py:28` `_dict_to_judge_result`：从 cache dict 重建 JudgeResult，只读
  `match/confidence/spans` → 补读两个新字段。

此项归入 PR-3（与 judge 产出新字段同批），否则新字段一过 cache 就蒸发 —— 而"蒸发"最隐蔽的
形态正是"DDL 加了列、DML 没动"：看着改了，其实没生效。

---

## 生成侧改动（Opus / Call 2）

### `module0/prompts.py` — Call 2 prompt 增段

在现有"label + self-eval"任务后追加：对每个 route==pass 的 sub_problem 产出 rubric。
prompt 内硬约束：

- rubric **必须从 `raw_text` / `failure_summary` 蒸**，禁止从 capability 标签名凭空发挥
  （决策 1）。
- `decisive_evidence` 描述证据**形态**（哪类步骤、什么信号），**禁止**硬规则。
- `capability_kind` 三选一，prompt 内给三型定义。

### `compiler.py` — 采集 + 组装

- `_build_sub_problem`（`compiler.py:258`）把解析出的 rubric 塞进 SubProblem。
- **rubric 失败降级为 `rubric=None`，不 drop 整个 sub_problem** —— 判据卡是增强，不是
  准入门槛。**注意：不要套用 `compiler.py:134-138` 那段外层模式** —— 那段是 schema 非法
  就 `_record_dropped`（**丢弃整个 sub_problem**），与本意相反（会把带坏 rubric 的合格
  问题一起丢掉）。正确做法：在 `_build_sub_problem` **内部对 rubric 部分单独 try**，rubric
  非法只置 None，sub_problem 其余字段照常构建。精神相似但行为相反，勿照抄。
- **成本注意**：Call 2 多产 rubric，输出 token 增加，可能撞 `max_tokens=4000`
  （`compiler.py:65`）。小样本先观察是否被截断，必要时调高。

---

## 执行侧改动（GPT-5.5 / Judge）—— 收益最大处

### `judge.py` — system prompt 重写（`judge.py:22-32`）

从"判是否演示 + 圈 start/end"改为**三步走**：

1. **对照**：逐条核对 slice 是否满足 `positive_criteria`、是否踩中 `negative_criteria`。
2. **定位**：指出决定性证据在第几步（`evidence_step`）。**指不出明确决定性步 →
   match=false**。此条直接杀掉"平淡 Found it"类擦边片。
3. **收窄 mask**：`spans` **只圈决定性步的 assistant 决策/推理那 1-2 步**，工具输出 dump
   排除在 mask 外（它是 context 不是 target）。

按 `capability_kind` 给判读视角（**prompt 内文字引导，非代码分支**）：见上文三型定义。

### `judge.py` — prompt 组装 + 解析

- **rubric 到这里是 dict，不是 `CapabilityRubric` 对象**：`CapabilityRubric` dataclass 只
  活在 module0；经 `_spec_to_dict` 序列化 → problem_store 存成 JSON blob（`stores.py`
  `spec_json`）/ 进程 LRU → module1 pipeline 全程把 sub_problem 当**普通 dict** 处理
  （`pipeline.py:217-222` 全是 `sub_problem.get(...)`）。所以取值是
  `sub_problem.get("rubric")`（**dict 或 None**），`_build_judge_prompt` 内部按 **dict 形状**
  消费（`rubric["positive_criteria"]`，**不是** `rubric.positive_criteria`）—— 在 module1
  写属性访问会 `AttributeError`。
- `_build_judge_prompt`（`judge.py:83`）注入该 rubric dict（None 时走旧格式，向后兼容）。
  现签名只收 `target_capability` + `trajectory_signal`，**需扩参**把 rubric 传入；
  `judge_batch` 签名跟着扩。
- **`judge_batch` 调用点有 4 处，全部要覆盖（勿只改前两处）**：
  - `pipeline.py:253` `_process_sub_problem`（legacy `run()` 路径）
  - `pipeline.py:309` `_score_sub_problem`（`run_scored`，**生产**）
  - `pipeline.py:367` `_score_sub_problem_cached`（`search`，**生产**，易漏）—— 只改前两处
    会导致 **search 路径静默走旧 prompt、拿不到 rubric 判据**，却看不出报错。
  - `backfill.py:60`（module0.5 回填）—— 见"双模型接力"节：该路径无 sub_problem、无 rubric
    源，传 None 走旧路即可，**本次不增强**，但新参数默认值必须让它不炸。
- `_parse_judge_response`（`judge.py:105`）解析 `evidence_step` / `criteria_hit`，沿用现有
  容错降级风格（`judge.py:141-148`），字段缺失不崩、退回 no-match。

### `summarizer.py` — 按需保调用序列（先不动）

现三个截断常量在 `summarizer.py:20-22`：assistant 150 / args 200 / tool_result 300。两处与
本方案相关，**别只盯 args**：

- **args 截断 200**：avoidance 类 judge 靠"工具调用**参数**"判断是否近似重复，args 截断
  可能切掉判据。
- **assistant 截断 150（更贴近新目标）**：本方案新增核心目标是"把 mask 收窄到 assistant
  决策/推理那 1-2 步"，judge 要靠切片文本**核对 `positive_criteria` + 定位 `evidence_step`**；
  决策推理一旦超过 150 字被截，judge 可能既核不准判据、也指不准决定性步 —— 这比 args 截断
  更直接威胁"精定位"这个最大收益。

**先不动，PR-3 pilot 一并观察 judge 是否因 assistant/args 看不全而误判或定位不准，再决定
放宽哪个、对哪类能力放宽。** 先观察再调，不预调。

---

## 双模型接力（已完成 + 本方案零额外接线）

- **已完成**：`inspector_serve.py` 的 `compile_model`（Opus, `UXFLOW_COMPILE_MODEL`）/
  `judge_model`（`UXFLOW_JUDGE_MODEL`）分流，已用 `scripts/check_model_split.py`
  验证两模型均 200 应答。**注意默认值**：`pipeline.py:38` / `inspector_serve.py` 的 judge
  默认是 `gpt-4o-mini`，"GPT-5.5"是**经 env 切换的目标态**，非代码默认；分流机制已就位，实际
  跑哪个取决于 `UXFLOW_JUDGE_MODEL` 是否设。下文"GPT-5.5"均指此目标态。
- 本方案下**无需再动模型 wiring**：rubric 生成走 compiler（已 Opus），judge 走 pipeline
  （judge_model，env 定），自动各归其位。这是先做分流的回报。
- 以下路径本次**不增强**（rubric=None、走旧 judge prompt），实现时无需为其接 rubric，
  只需保证新参数默认值让它们不炸：
  - 三个 smoke 脚本（`e2e_smoke.py` / `complex_smoke_report.py` / `uxflow_evolve.py`）
    仍用 `MODULE0_TEST_MODEL` 老逻辑，模型分流也不生效。
  - `backfill.py:60`（module0.5 标签回填）按 `label.description` 判定，**流程里根本没有
    sub_problem，也就没有 rubric 源** —— 天然走旧路，符合预期。

---

## 失败轨迹证据接入（可选输入；把标签从"猜"变成"照现场判"）

### 要解决的问题（与 rubric 同源的第二半）

现状 `failure_summary → target_capability` 这一跳（Call 2 打标）**唯一的证据就是用户那句话**：
`failure_summary` 本身是 LLM 从用户描述生成的，不锚任何真实证据。标签是"猜"出来的。

若输入除"用户问题"外还带**对应的失败轨迹**（真实错误现场），Call 2 就能对着现场判：
错在哪几步、演示了哪种失败、据此定哪个 capability label —— 而非从一句想象的失败描述凭空
联想。这与 rubric 决策 1/2 是**同一条哲学的第二半**：rubric 要求"锚在非 LLM 生成的证据
上"，而真实失败轨迹是比"用户一句话"强得多的锚。**同一份轨迹证据同时喂三处**：定 label、
覆盖 failure_summary、当 rubric 的 `negative_criteria` / `decisive_evidence` 的真实实例
（不再"想象反例"，照抄现场）。故与 rubric 合并在一份 spec、同批落地。

### 设计决策（冻结，勿推翻）

1. **失败轨迹是可选输入**：部分问题无对应轨迹。`failure_evidence=None` 时 Call 2 走现状
   "文字猜"路径，**逐字不变**。镜像 rubric=None 的降级模式，向后兼容天然成立。

2. **压缩在 service 层做，module0 绝不 import module1**（架构护栏）：轨迹加载/切片/摘要
   三件套（`load_trajectories` / `slice_trajectory` / `summarize_slice`）都在 module1。
   **现状 module0 不依赖 module1**（仅 module0_5 依赖）。若让 compiler 直接 import 它们，
   会新建一条 module0→module1 反向依赖边。改为：**orchestrator（本就同时 import module0/1）
   负责把失败轨迹 load+slice+summarize 成一段紧凑文本**，`compile()` 只多收一个
   `failure_evidence: str | None`。compiler 维持"文字进、spec 出"，可喂假文本单测。
   轨迹压缩是**能力无关**的、打标前做一次，语义完全正确。**未来若要把轨迹工具沉成
   module0/1 共享的 `trajectory_io` 底座，再抽 —— 现在留 seam 不预抽。**

3. **证据只注入 Call 2（打标），不进 Call 1（拆解）**：Call 1 保持纯文本，产出的是**候选**
   sub_problem（便宜、高召回、可能过拆）；轨迹在 Call 2 当**唯一裁判**做剪枝+校正。这是
   标准 propose-then-verify：Call 1 降级为"候选生成器"，不需要看轨迹（看了只把 prompt
   变长、把"纯结构拆解"职责搞浑）。

   **Call 2' 交互（勿漏）**：`build_call2_prime_messages`（澄清重评路径）**复用**
   `build_call2_messages`（`prompts.py:282`）再追加 `_CALL2_PRIME_EXTRA`。若把
   `failure_evidence` 注入点放在 `build_call2_messages`，Call 2' 会**被动继承轨迹证据**。
   本次决策：**Call 2' 路径不接入 failure_evidence**（它处理的是被澄清拆分后的子问题，
   与原始轨迹的证据步对应关系已错位，认领无意义）。实现上让 `build_call2_messages` 的
   `failure_evidence` 参默认 None，Call 2' 复用时**不透传**该参（显式传 None），避免泄漏。

4. **轨迹有"否决权"，无"提案权"（V1 有意画的不对称）**：Call 2 只能对 Call 1 **已产出**的
   sub_problem 认领/drop/改标签，**不能凭空 split 或新增**。故 Call 1 的三类错里：
   - **over-split**（拆出多余假问题）→ ✅ Call 2 按证据 drop（决策 6）。
   - **mis-label**（问题对、标签错）→ ✅ Call 2 按现场覆盖（决策 5）。
   - **under-split / 发现**（用户没描述、但轨迹里真实发生的失败）→ ❌ **V1 救不了**，需
     Call 2 有"从证据新造 sub_problem"的能力。**提案比否决难得多、错误率高得多，且会把
     Call 2 从"对候选核对"变成"开放式发现"，判据来源失控 → 明确压 V2，单独立项。**
   这条不对称是**有意选择**，不是"Call 1 碰巧没喂轨迹"的副产物。

5. **轨迹优先，覆盖用户文字（但留痕，可审计）**：轨迹在场且认领到证据步时，用**现场观测到
   的失败**覆盖 `failure_summary` 的措辞、并据现场定 `target_capability`。**但 `raw_text`
   （用户原话）原样保留** —— "覆盖"是留痕替换，不是抹掉，能对比"原描述 vs 现场"。守住
   rubric"把隐藏假设外置成可审计"的哲学。

   **与决策 1（rubric 从 raw_text/failure_summary 蒸）的时序**：覆盖与 rubric 蒸馏在**同一次
   Call 2** 内。约定**先覆盖、后蒸**：轨迹在场时 rubric 蒸的是**被现场校正后**的
   failure_summary（+ raw_text 原话），这不与决策 1 冲突、反而更强 —— 判据从此锚在真实证据
   而非想象失败上，与本节哲学同向。无轨迹时退化为决策 1 原样（从原始 failure_summary 蒸）。

6. **认领不到证据 → drop（轨迹赢），但带独立可审计 reason**：轨迹在场、却对某条 sub_problem
   **找不到任何证据步**，判定为"该 sub_problem 是 Call 1 的 over-split"，**drop 之**（倒向
   精度/剔除，与"伪相关毒害 SFT"的北极星一致；且"轨迹=对口失败现场"前提下，真问题理应在
   场，召回损失小）。**护栏（成本近零）**：此类 drop 必须落**独立 `drop_reason`（如
   `no_trajectory_evidence`）**，不混进现有 `"other"`，使误杀能在 §1.4 审计桶被捞出
   （复用"错了要能被发现，而非静默"）。作用域两条：**仅轨迹在场时启用**（无轨迹永不因此
   drop）、**逐 sub_problem 判**（只 drop 零证据那条，不牵连同批别的）。

   **前置（勿漏，否则决策 6 直接炸）**：`no_trajectory_evidence` 必须先加进 `drop_reason`
   的**8 个同源落点**，否则 LLM 输出它时 `DroppedSubProblem.__post_init__`
   （`schema.py:114-115`，`drop_reason not in DROP_REASONS` 即抛 ValueError）会崩，或 prompt
   没教这个值 → LLM 根本不产它。**同源点比想象多——两份 Call 2 system prompt
   （`_CALL2_SYSTEM_WITH_TAXONOMY` 有词表 / `_CALL2_SYSTEM_EMPTY_TAXONOMY` 空词表冷启动，
   不是"Call 2 与 Call 2'"），每份里 drop_reason 枚举出现 3 次**：
   - `schema.py:35` `DROP_REASONS = frozenset({...})` 加值。
   - **输出格式 JSON 示例**（`prompts.py:87` + `:155`）——**LLM 直接照抄的模板，最易漏、漏了
     必失效**。这两行硬编码 `"ambiguous|not_applicable|label_diverged|other"`，不改则 LLM
     的输出样例里根本没有新值。
   - **判定树表格**（`prompts.py:110` + `:178`）各加一行。
   - **约束措辞**（`prompts.py:124` + `:192`）"取值只能是上表 N 个之一"，N 跟着改。
   - `interface-contract.md` §1.4 drop_reason 枚举加值。
   - 判定标准写清：**仅当 failure_evidence 在场、且该 sub_problem 认领不到任何证据步**才用
     此 reason；无轨迹时此 reason 不可能出现（防 LLM 在纯文本路径误用）。
   （即：`schema` 1 处 + 两份 prompt 各 3 处 = 6 + 契约 1 处，共 8 个落点，勿只改判定树。）

### 输入契约：manifest 升级 JSONL，裸行兼容

```jsonl
{"question": "agent 老是重复读同一个文件", "failure_trajectory": "traj/p12.json"}
{"question": "另一个问题", "failure_trajectory": null}
一行纯文本                       ← 向后兼容：等价 {question: 该行, failure_trajectory: null}
```

- 解析规则：**一行能 parse 成 JSON object → 结构化读**；否则整行当 `question`、轨迹为空。
  老的纯文本清单**一字不改照跑**。
- 轨迹**传路径引用**（不内联），service 用 `load_trajectories` 读 —— 与语料轨迹**同 schema**
  （已确认），零 adapter。
- 落点：`orchestrator.py:67` `manifest_lines: list[str]` 及 `:91-92` 逐行 `strip()` 的读法要
  升级为"每行先试 JSON、失败退化为纯文本"。**写路径（`run_ingest`，同文件 `:248` 逐行读）
  也逐行读 manifest，需同步适配**，否则 ingest 遇 JSONL 行会把整个 JSON 串当 question。

### 数据结构改动

**SubProblem 加字段（`schema.py:58`，与 rubric 同批）**

```
failure_evidence: LabelEvidence | None = None
```

```
@dataclass(frozen=True)
LabelEvidence:
    trajectory_id:   str
    evidence_steps:  list[int]   # 标签据哪几步定的；人工抽审一跳到现场
    observed_failure: str        # 现场真实观测到的失败（覆盖 failure_summary 的证据来源）
```

- **可空**：镜像 rubric，无轨迹时恒 None，向后兼容。
- `evidence_steps` / `observed_failure` 是**可回溯性关键**：每个据轨迹定的标签都能追溯
  "据哪几步、看到什么失败"，与 JudgeResult 的 `evidence_step` 同一套可审计精神。
- **契约同步（硬要求）**：又一次 §1.2 加字段，与 rubric 同批改 `interface-contract.md`，
  勿漏。SubProblem docstring 的"12 required fields"措辞一并复核（rubric + failure_evidence
  都是**可选**字段，不进 required 计数，但 docstring 要提及）。

### 生成侧改动（Call 2）

- **`compiler.compile` 扩签名**：`compile(self, raw_input, *, failure_evidence=None)`。
  `failure_evidence` 为 service 压缩好的紧凑文本（或 None）。**默认 None 保证三个 smoke
  脚本、无轨迹清单不炸。**
- **Call 2 prompt（`prompts.py` `build_call2_messages`）扩参**：注入 `failure_evidence`
  文本（None 时走现状纯文字格式）。prompt 内硬约束追加：
  - **逐 sub_problem** 去轨迹认领"哪几步演示了**这条**的失败"（→ `evidence_steps`）。
    轨迹挂在整条 raw_input 上、一条轨迹对多个 sub_problem，故须逐条认领，不假设一轨一能力。
  - 认领到 → 据现场定 `target_capability`、用现场覆盖 `failure_summary`、产出
    `observed_failure`；**同一份现场同时供 rubric 的 `negative_criteria` /
    `decisive_evidence`**（照抄，不想象）。
  - **认领不到 → 该条 route=drop、`drop_reason="no_trajectory_evidence"`**（决策 6）。
    **该 reason 须先加进 8 个同源落点**（`schema.py:35` `DROP_REASONS` + 两份 Call 2 prompt
    各 3 处：`prompts.py:87/155` 输出格式示例、`:110/178` 判定树、`:124/192` 措辞 +
    `interface-contract.md` §1.4），否则一输出即崩或静默失效，详见决策 6 前置。
  - `capability_kind` 的 **avoidance / recovery 两型受益最大**：本最难判（找"本可犯错的
    锚点"/"error→处置"），现在错误现场直接在眼前。
- **`compiler.py` 采集**：`_build_sub_problem`（`compiler.py:258`）把解析出的
  `failure_evidence` 塞进 SubProblem；**沿用 rubric 的"内部单独 try、失败置 None 不 drop
  整条"**（勿套 `compiler.py:134-138` 外层 drop 模式）。**但注意与决策 6 的区别**：
  "LLM 没产出合法 evidence 结构" → 置 None（增强失败，不惩罚问题）；"LLM 明确判无证据步"
  → route=drop（这是 Call 2 的判定结果，走既有 route=drop 路，不是解析降级）。两者不同源，
  勿混。
- **成本注意**：Call 2 token 三重叠加撞 `max_tokens`（详见副作用表）。此处特有约束：**轨迹
  压缩长度须设上限**（复用 summarizer 的截断常量思路，`summarizer.py:20-22`），别让单条长
  轨迹吃满预算。具体上限值见开放点。

### 执行侧（judge）——本节零改动

失败轨迹**只作用于 module0 打标/rubric 生成**，不进 judge。judge 侧的一切（含前述 rubric
prompt 重写、evidence_step/criteria_hit）与本节正交，不叠加改动。

### 铁律（两条硬不变量，写进代码断言）

1. **失败轨迹永不进 module1 的正例语料 / 索引**。它是"错误现场"，一旦被当正例召回选进
   SFT 即灾难。service 里读**失败轨迹**的路径，必须与喂 module1 index 的 **ingest 写路径**
   （`run_ingest` 的 `problem_store.add` / 轨迹入库）**物理隔离，绝不共用**。加断言。
2. **taxonomy 治理不变**：轨迹只提升"选/提议标签"的准确度，受控词表注入照旧，标签来源不变。

---

## 副作用与验收

| 副作用 | 说明 | 应对 |
|---|---|---|
| 擦边片被压分 / legacy 路径归零 | judge 变严：生产 `run_scored`/`search` 路径对 match=false 是 **rerank 0.3 衰减保留**（`rerank.py` `_MISS_DECAY`），**不移除**；只有 legacy `run()` 才按 `min_confidence` 硬过滤（`pipeline.py:262`）→ 可能某能力归零 | 生产路径下擦边片是被**压分沉底**非移除；"是否对 scored 路径也硬过滤"是**未决策点**，见开放点。无论哪种，加"某能力正例不足"告警，别静默 |
| Call 2 撞 max_tokens | 多产 rubric（输出↑）+ 轨迹文本进 prompt（输入↑）+ 多产 evidence（输出↑），三重叠加更易撞 `max_tokens=4000`（`compiler.py:65`） | 轨迹压缩设长度上限；小样本先看是否截断，必要时调高 |
| judge 成本↑ | 最高量档换 GPT-5.5 + prompt 变长 | 小样本先看账单/延迟；未来可加"便宜模型粗筛→GPT 精判"两段式（**本次不做**） |
| 契约漂移 | schema 加字段（rubric / failure_evidence）+ drop_reason 加枚举值 | 同步改 interface-contract.md（§1.2 字段 + §1.5/1.6 子 schema + §1.4 drop_reason）+ drop_reason **8 个同源落点**（schema 1 + 两份 prompt 各 3 + 契约 1，见决策 6 前置）（硬要求） |
| 轨迹 over-drop | 决策 6"零证据即 drop"可能误杀"轨迹没覆盖到"的真问题 | 独立 `drop_reason="no_trajectory_evidence"` 落审计桶，可事后捞；仅轨迹在场启用、逐条判 |
| manifest 双读点 | JSONL 升级须覆盖预览 + ingest 两条读 manifest 的路径 | `orchestrator.py:91-92`（预览）+ `:248`（`run_ingest` 写路径）都改，漏一处该路径把 JSON 串当 question |

**成功判据**（无 ground truth，用可观测代理）：
1. 原"落点是工具 dump / 平淡过渡步"的片段，mask 明显收窄到 assistant 决策步。
2. 擦边片（有相关上下文但未真展示能力）被判 match=false 的比例上升。
3. 每个 match 都能**在 InspectorView 上**回溯到"命中 rubric 哪条 + 证据在第几步"。
   **前提**：judge 产出后，字段要走通 rerank 传播链（/runs）和 judge_cache（search）**两条**
   才到得了界面——只产出不传播 = 判据 #3 静默为零，见"可回溯字段的传播链"节。
4. （轨迹接入）带失败轨迹的问题，`target_capability` 由现场证据支撑 —— 人审几条，确认
   标签不再"看文字猜"，且 `evidence_steps` 指向的步确实演示了该失败；无轨迹的问题行为
   与改前逐字一致。

---

## 落地顺序（分 3 个独立可验 PR，勿合并）

### PR-1：数据结构 + 透传（安全地基，零行为变化）

- `schema.py` 加 `CapabilityRubric` / `CAPABILITY_KINDS` / SubProblem.rubric（可空）。
- `schema.py` 加 `LabelEvidence` / SubProblem.failure_evidence（可空）—— 与 rubric 同批。
  新字段加在 `schema.py:72`（route 字段之后，非 `:58` 装饰器行）。**顺手更新
  `validate_problem_spec`（`schema.py:136-163`）读两个可选字段** + SubProblem docstring 的
  "12 required fields"措辞（可选字段不进 required 计数，但 docstring 要提及 rubric/
  failure_evidence 存在）。`validate_problem_spec` 现只在测试用到，但不改会静默丢新字段。
- `models.py` JudgeResult 加 `evidence_step` / `criteria_hit`（带默认值）。
- `orchestrator.py` `_spec_to_dict` + `parsing.py` parse 透传 rubric **和 failure_evidence**
  （均可选字段）。
- **manifest JSONL 读取升级**：`orchestrator.py:91-92`（预览）+ `:248`（`run_ingest` 写路径）
  两处逐行读，改为"先试 JSON object、失败退化为纯文本"。**近似零行为变化**（唯一窄边界：
  一条本身恰好是合法 JSON object 的纯文本问题——如以 `{` 开头——会从"当 question"变成
  "当结构化行"。SWE 问题描述极少长这样，可接受；实现时可要求结构化行**必须含 `question`
  键**否则退化为纯文本，进一步缩小边界）。
- 同步 interface-contract.md：§1.2 加 rubric / failure_evidence 字段 + 新增 §1.5
  CapabilityRubric、§1.6 LabelEvidence 两个子 schema。（**JudgeResult 不在此契约** —— 它是
  模块 1 内部产物，本契约只管"模块 0↔模块 1 输入端"，无需为其改契约。）**注意：契约已先行
  写入这些定义（当前处于"契约 ≠ 代码"窗口），PR-1 实现者是让 `schema.py` 追平契约，不是
  再改契约——落地后与契约逐字对齐即可。**
- **验收**：rubric / failure_evidence 恒 None，judge 走旧路，纯文本清单照跑，现有测试全绿
  → 证明零破坏。

### PR-2：生成侧

- Call 2 prompt（`prompts.py`）+ compiler 采集（`compiler.py`）产 rubric。
- **失败轨迹证据接入**（同属生成侧，同批）：
  - orchestrator 层把失败轨迹 load+slice+summarize 成紧凑文本（**复用 module1 三件套，
    module0 不 import module1**）；`compile()` 扩 `failure_evidence` 参。
  - Call 2 prompt 扩参：逐 sub_problem 认领证据步 → 定 label / 覆盖 failure_summary /
    产 `observed_failure` / 喂 rubric 反例；认领不到 → `drop_reason="no_trajectory_evidence"`。
  - **`no_trajectory_evidence` 加进 8 个同源落点**（`schema.py:35` `DROP_REASONS` + 两份
    Call 2 prompt 各 3 处：输出格式示例 `prompts.py:87/155`、判定树 `:110/178`、措辞
    `:124/192` + `interface-contract.md` §1.4）—— **尤其 :87/:155 的输出格式示例是 LLM 照抄
    模板，漏了决策 6 静默失效**；漏 schema 侧则一输出即抛 ValueError。详见决策 6 前置。
  - compiler 采集 failure_evidence（内部单独 try 置 None，勿套外层 drop 模式；注意与
    route=drop 判定不同源）。
  - 铁律：失败轨迹读路径与 ingest 写路径物理隔离，加断言。
- **验收**：跑一条清单，**人眼审几张 rubric 卡 + 几条带轨迹的 label** —— 确认判据从
  raw_text 蒸、标签由现场证据支撑、`evidence_steps` 指向的步确实演示该失败、三型标注合理；
  **另跑一条纯文本清单确认逐字不变**。

### PR-3：执行侧（动质量的手术）

- judge prompt 重写 + prompt 组装扩参 + 解析（`judge.py`）。
- **`judge_batch` 全部 4 处调用点扩参**：`pipeline.py:253` / `309` / `367` +
  `backfill.py:60`（后者传 None 走旧路，仅保证不炸）。**勿漏 367（search 生产路径）**。
- **可回溯字段传播链（/runs 与 search 两路共用，最易漏，与 judge_cache 平级）**：
  `module2/models.py` `ScoredCandidate` 加 `evidence_step`/`criteria_hit` 成员 →
  `module2/rerank.py:38-49` `rerank()` 补搬这两字段 → `service/viewmodel.py:64-77`
  InspectorView surface。**不补这条链，judge_cache 补得再全，/runs 路径也拿不到证据步**
  （/runs 根本不走 cache）。详见"可回溯字段的传播链"节。
- **judge cache 扩字段（仅 search 路径；列式存储，5 个点缺一即蒸发；DDL 加列 ≠ 生效）**：
  写路径 `pipeline.py:373` `judge_cache.put` 构造 dict 补字段 → `stores.py:279-291` `put()`
  的 `INSERT` 列清单 +`VALUES` 补列（**补写主战场**）→ `stores.py:62-68` `CREATE TABLE`
  加列 + **旧库幂等 `ALTER TABLE ADD COLUMN` 迁移**（`CREATE TABLE IF NOT EXISTS` 不给旧库
  加列）；读路径 `stores.py:259-269` `get()` 的 `SELECT` + 返回 dict 补 key
  → `pipeline.py:28` `_dict_to_judge_result` 补读。**只改 CREATE TABLE 不动 get()/put() 或
  漏旧库迁移，都是隐蔽假修复。** 详见"JudgeResult 新字段的持久化"节。
- **验收**：小样本对比改前/改后的 mask 落点与 match 率，核对成功判据；**分别验证 /runs
  路径（走 rerank 链）与 search 路径（走 cache）都在 InspectorView 带出
  evidence_step/criteria_hit**，而非只有首判、或只有某一条路径。

**PR-1 是安全地基，PR-3 才动质量。逐个跑通再下一步。**

---

## 开放点（待后续拍板，不阻塞本次）

1. **正例不足告警的形态**：日志？InspectorView 字段？SSE 事件？PR-2/3 时定。
2. **scored 生产路径是否对 match=false 硬过滤**：现状是 rerank 0.3 衰减保留（非移除），
   judge 变严后擦边片只是沉底、仍可能被 module3 选入。要不要在 scored 路径也加一道
   confidence 硬门槛，把变严的收益真正落到"剔除"而非"压分"？PR-3 观察 match 率后定。
3. **summarizer args 截断是否放宽**：PR-3 pilot 后定。
4. **diff 交叉验证审计**：缝进 pipeline（持续体温计）还是独立脚本（便宜但有盲区）——
   未来独立议题，涉及是否动 llm_gateway 多 provider 编排。
5. **轨迹"提案权"（V2）**：V1 轨迹只有否决权（drop over-split）、无提案权（不能从证据
   新造 sub_problem），故救不了 under-split / 用户没描述但现场真实发生的失败。开放：是否
   让 Call 2 具备"从轨迹证据发现新 sub_problem"的能力 —— 提案比否决错误率高得多、会把
   Call 2 从"对候选核对"变成"开放式发现"，判据来源失控，需单独立项设计。
6. **轨迹压缩策略**：失败轨迹喂 Call 2 前的 slice+summarize 用哪套截断/选步策略、长度上限
   多少（既要够判、又别撞 max_tokens）。PR-2 小样本观察后定。
7. **Call 1 拆解粒度与轨迹的错位**：一条轨迹挂在整条 raw_input、对多个 sub_problem；若
   Call 1 拆解与轨迹实际演示的失败边界系统性错位，逐条认领的命中率会低。PR-2 观察认领
   命中率后，再决定是否需要"轨迹辅助拆解"（属 V2 提案权范畴）。
