# SPDX-License-Identifier: Apache-2.0
"""Batch3 编排层: run_search (读路径/只读铁律) + run_ingest (写路径/D4 隔离)."""
from dataclasses import dataclass, field

from service.orchestrator import PipelineDeps, run_search, run_ingest
from service.dedup import ProblemCompiler, normalize_question


# ── fakes (复用 test_orchestrator.py 风格) ─────────────────────────────
@dataclass
class FakeSubProblem:
    id: str
    failure_summary: str
    target_capability: list
    confidence: float = 0.9
    origin: str = "original"
    parent_id: object = None
    raw_text: str = ""
    trajectory_signal: str = ""
    hyde_positive: list = field(default_factory=list)
    keywords: list = field(default_factory=list)
    route: str = "pass"


@dataclass
class FakeFilters:
    languages: object = None
    tools_used: object = None
    has_verification_step: object = None


@dataclass
class FakeSpec:
    raw_input: str
    domain: str
    sub_problems: list


@dataclass
class FakeScored:
    trajectory_id: str
    slice_index: int
    trajectory_path: str
    sub_problem_id: str
    capability: list
    relevance_score: float
    judge_confidence: float
    loss_mask_spans: list
    judge_match: bool = True
    embedding: list = field(default_factory=list)
    bm25_tokens: list = field(default_factory=list)


@dataclass
class FakeStep:
    index: int
    role: str
    content: str
    tool_call_name: object = None
    tool_call_args: object = None
    tool_result: object = None


@dataclass
class FakeTrajectory:
    id: str
    steps: list


class FakeCompiler:
    """按 raw_input 内容可选抛异常 (D4 测试用)。"""

    def __init__(self, raise_on=None):
        self._n = 0
        self._raise_on = raise_on or set()

    async def compile(self, raw_input, *, failure_evidence=None):
        self._n += 1
        if raw_input in self._raise_on:
            raise ValueError(f"boom::{raw_input}")
        sp = FakeSubProblem(
            id="p1",
            failure_summary=f"summary::{raw_input}",
            target_capability=["valid_syntax_in_toolcall"],
        )
        sp.structured_filters = FakeFilters()
        return FakeSpec(raw_input=raw_input, domain="agentic_swe", sub_problems=[sp])


class FakeEmbedder:
    """embed(text) -> 固定向量 (dedup 路径靠 FakeProblemStore.nearest 控制命中)。"""

    def __init__(self):
        self.calls = []

    def embed(self, text):
        self.calls.append(text)
        return [1.0, 0.0, 0.0]


class FakeProblemStore:
    """内存 store，记录 add 调用；nearest 可配置返回 None 或 (pid, sim, spec)。"""

    def __init__(self, nearest_result=None, records=None):
        self._nearest_result = nearest_result
        self._records = records or {}
        self.add_calls = []
        self.nearest_calls = []

    def nearest(self, embedding):
        self.nearest_calls.append(embedding)
        return self._nearest_result

    def get(self, pid):
        return self._records.get(pid)

    def add(self, raw_question, embedding, spec):
        self.add_calls.append((raw_question, embedding, spec))
        return f"pid::{raw_question}"


class FakePipeline:
    def __init__(self, trajectories=None):
        self._trajectories = trajectories or []
        self.ingest_calls = []

    async def search(self, *, problem_specs, judge_cache, on_progress=None):
        out = []
        total = sum(len(s["sub_problems"]) for s in problem_specs)
        done = 0
        for spec in problem_specs:
            for sp in spec["sub_problems"]:
                out.append(FakeScored(
                    "t1", 0, "dummy.jsonl", sp["id"],
                    ["valid_syntax_in_toolcall"], 0.8, 0.9,
                    [{"start_step": 0, "end_step": 1}],
                ))
                done += 1
                if on_progress is not None:
                    on_progress(done, total)
        return out

    def ingest_trajectories(self, paths, *, on_trajectory=None):
        self.ingest_calls.append(list(paths))
        for path in paths:
            for traj in self._trajectories:
                if on_trajectory is not None:
                    on_trajectory(traj, str(path))


class FakeTrajectoryStore:
    def __init__(self):
        self.upsert_calls = []

    def upsert(self, traj_id, steps, *, source_path):
        self.upsert_calls.append((traj_id, steps, source_path))


class FakeJudgeCache:
    def __init__(self):
        self.put_calls = []

    def get(self, *args):
        return None

    def put(self, *args):
        self.put_calls.append(args)


def fake_select(candidates, *, sub_problem_ids, selection, general):
    return {
        "targeted": list(candidates),
        "general": [],
        "manifest": {"targeted_count": len(candidates),
                     "general_count": 0, "general_ratio": 0.3},
    }


def _search_deps(*, nearest_result=None, records=None, compiler=None,
                 problem_store=None):
    return PipelineDeps(
        compiler=compiler or FakeCompiler(),
        pipeline=FakePipeline(),
        select_fn=fake_select,
        load_trajectories_fn=lambda p: [],
        problem_store=problem_store or FakeProblemStore(
            nearest_result=nearest_result, records=records),
        judge_cache=FakeJudgeCache(),
        embedder=FakeEmbedder(),
    )


# ── 1. run_search 命中已有问题（只读）─────────────────────────────────
async def test_run_search_dedup_hit_is_readonly(tmp_path):
    hit_spec = {"raw_input": "老问题", "domain": "agentic_swe",
                "sub_problems": [{"id": "old.p1",
                                  "failure_summary": "s",
                                  "target_capability": ["valid_syntax_in_toolcall"]}]}
    store = FakeProblemStore(
        nearest_result=("pidX", 0.97, hit_spec),
        records={"pidX": {"problem_id": "pidX", "raw_question": "老问题",
                          "spec": hit_spec}},
    )
    compiler = FakeCompiler()
    deps = _search_deps(compiler=compiler, problem_store=store)
    view, _ = await run_search("新提问但很像老问题", deps=deps, emit=lambda e: None)

    assert view["mode"] == "search"
    assert view["dedup"] is not None
    assert view["dedup"]["matched_problem_id"] == "pidX"
    assert view["dedup"]["similarity"] == 0.97
    assert view["dedup"]["matched_question"] == "老问题"
    # 只读铁律 + 命中不 compile:
    assert compiler._n == 0
    assert store.add_calls == []


# ── 2. run_search 新问题（只读！）────────────────────────────────────
async def test_run_search_new_problem_is_readonly(tmp_path):
    store = FakeProblemStore(nearest_result=None)
    compiler = FakeCompiler()
    deps = _search_deps(compiler=compiler, problem_store=store)
    view, _ = await run_search("全新问题", deps=deps, emit=lambda e: None)

    assert view["mode"] == "search"
    assert view["dedup"] is None
    assert compiler._n == 1
    # 关键: 只读预览不写库
    assert store.add_calls == []


# ── 3. ProblemCompiler LRU（第二次命中缓存不重复 compile）──────────────
async def test_problem_compiler_lru_avoids_recompile():
    store = FakeProblemStore(nearest_result=None)  # 恒未命中持久库
    compiler = FakeCompiler()
    pc = ProblemCompiler(
        compiler, store, FakeEmbedder(), tau_q=0.90,
        serialize_fn=lambda so, q: {"raw_input": so.raw_input,
                                    "sub_problems": []},
    )
    spec1, d1 = await pc.get_or_compile("重复问题")
    spec2, d2 = await pc.get_or_compile("重复问题")
    assert d1 is None and d2 is None
    assert compiler._n == 1          # 第二次命中 LRU
    assert spec1 is spec2
    assert store.add_calls == []     # 只读


async def test_problem_compiler_lru_evicts_over_cap():
    store = FakeProblemStore(nearest_result=None)
    compiler = FakeCompiler()
    pc = ProblemCompiler(
        compiler, store, FakeEmbedder(), tau_q=0.90, lru_cap=2,
        serialize_fn=lambda so, q: {"q": q},
    )
    await pc.get_or_compile("q1")
    await pc.get_or_compile("q2")
    await pc.get_or_compile("q3")          # 逐出 q1
    assert normalize_question("q1") not in pc._lru
    await pc.get_or_compile("q1")          # q1 被逐出 → 重新 compile
    assert compiler._n == 4


# ── 4. run_ingest 仅轨迹 ─────────────────────────────────────────────
async def test_run_ingest_trajectory_only(tmp_path):
    traj = FakeTrajectory("t1", [FakeStep(0, "user", "hi")])
    pipeline = FakePipeline(trajectories=[traj])
    tstore = FakeTrajectoryStore()
    deps = PipelineDeps(
        compiler=FakeCompiler(), pipeline=pipeline, select_fn=fake_select,
        load_trajectories_fn=lambda p: [], trajectory_store=tstore,
        problem_store=FakeProblemStore(), embedder=FakeEmbedder(),
    )
    view, _ = await run_ingest(trajectory_path="traj.jsonl", deps=deps,
                               emit=lambda e: None)
    assert pipeline.ingest_calls == [["traj.jsonl"]]
    assert len(tstore.upsert_calls) == 1
    assert tstore.upsert_calls[0][0] == "t1"
    assert view["summary"]["added"] == 0        # 无清单 → 问题计数为 0
    assert view["summary"]["traj_ingested"] == 1


# ── 4b. run_ingest 传了轨迹但一条都没解析出来 ────────────────────────
async def test_run_ingest_empty_trajectory_file_counts_zero_not_none(tmp_path):
    """空/全非法的轨迹文件 → traj_ingested 是 0，不是 None。

    前端按 `!= null` 门控该段：0 要显示"处理轨迹 0 条"（传了但没解析出东西，
    是用户需要看到的信号），None 才整段隐藏（压根没传轨迹）。
    """
    pipeline = FakePipeline(trajectories=[])       # 文件里没有可解析的轨迹
    tstore = FakeTrajectoryStore()
    deps = PipelineDeps(
        compiler=FakeCompiler(), pipeline=pipeline, select_fn=fake_select,
        load_trajectories_fn=lambda p: [], trajectory_store=tstore,
        problem_store=FakeProblemStore(), embedder=FakeEmbedder(),
    )
    view, _ = await run_ingest(trajectory_path="empty.jsonl", deps=deps,
                               emit=lambda e: None)
    assert tstore.upsert_calls == []
    assert view["summary"]["traj_ingested"] == 0


# ── 5. run_ingest 仅清单 ─────────────────────────────────────────────
async def test_run_ingest_manifest_only(tmp_path):
    store = FakeProblemStore(nearest_result=None)
    deps = PipelineDeps(
        compiler=FakeCompiler(), pipeline=FakePipeline(), select_fn=fake_select,
        load_trajectories_fn=lambda p: [], problem_store=store,
        embedder=FakeEmbedder(),
    )
    view, _ = await run_ingest(manifest_lines=["问题A", "问题B", "  "],
                               deps=deps, emit=lambda e: None)
    assert len(store.add_calls) == 2
    assert view["summary"]["added"] == 2
    assert view["summary"]["skipped_dup"] == 0
    assert view["summary"]["failed"] == 0
    assert view["summary"]["traj_ingested"] is None   # 没传轨迹 ≠ 传了但 0 条


# ── 6. run_ingest 去重跳过 ───────────────────────────────────────────
async def test_run_ingest_skips_duplicate(tmp_path):
    store = FakeProblemStore(nearest_result=("pidX", 0.95, {}))
    compiler = FakeCompiler()
    deps = PipelineDeps(
        compiler=compiler, pipeline=FakePipeline(), select_fn=fake_select,
        load_trajectories_fn=lambda p: [], problem_store=store,
        embedder=FakeEmbedder(),
    )
    view, _ = await run_ingest(manifest_lines=["重复问题"], deps=deps,
                               emit=lambda e: None)
    assert view["summary"]["skipped_dup"] == 1
    assert view["summary"]["added"] == 0
    assert compiler._n == 0
    assert store.add_calls == []


# ── 7. run_ingest 逐行隔离（D4）─────────────────────────────────────
async def test_run_ingest_per_line_isolation(tmp_path):
    store = FakeProblemStore(nearest_result=None)
    compiler = FakeCompiler(raise_on={"坏问题"})
    deps = PipelineDeps(
        compiler=compiler, pipeline=FakePipeline(), select_fn=fake_select,
        load_trajectories_fn=lambda p: [], problem_store=store,
        embedder=FakeEmbedder(),
    )
    view, _ = await run_ingest(manifest_lines=["好问题1", "坏问题", "好问题3"],
                               deps=deps, emit=lambda e: None)
    assert view["summary"]["added"] == 2    # 第3行仍被处理
    assert view["summary"]["failed"] == 1
    assert len(store.add_calls) == 2


# ── 8. run_ingest 两者都传 ───────────────────────────────────────────
async def test_run_ingest_both_trajectory_and_manifest(tmp_path):
    traj = FakeTrajectory("t1", [FakeStep(0, "user", "hi")])
    pipeline = FakePipeline(trajectories=[traj])
    tstore = FakeTrajectoryStore()
    store = FakeProblemStore(nearest_result=None)
    deps = PipelineDeps(
        compiler=FakeCompiler(), pipeline=pipeline, select_fn=fake_select,
        load_trajectories_fn=lambda p: [], trajectory_store=tstore,
        problem_store=store, embedder=FakeEmbedder(),
    )
    view, _ = await run_ingest(manifest_lines=["问题A"],
                               trajectory_path="traj.jsonl", deps=deps,
                               emit=lambda e: None)
    assert len(tstore.upsert_calls) == 1
    assert len(store.add_calls) == 1
    assert view["summary"]["added"] == 1
    assert view["summary"]["traj_ingested"] == 1


# ── 9. PipelineDeps 向后兼容（四个位置参数）─────────────────────────
def test_pipeline_deps_backward_compatible():
    deps = PipelineDeps(
        compiler=FakeCompiler(),
        pipeline=FakePipeline(),
        select_fn=fake_select,
        load_trajectories_fn=lambda p: [],
    )
    assert deps.problem_store is None
    assert deps.trajectory_store is None
    assert deps.judge_cache is None
    assert deps.embedder is None


# ── 10. PR-2: run_ingest 结构化行压缩失败轨迹并传 failure_evidence ──────
class CapturingCompiler(FakeCompiler):
    def __init__(self):
        super().__init__()
        self.evidence_seen = []

    async def compile(self, raw_input, *, failure_evidence=None):
        self.evidence_seen.append(failure_evidence)
        return await super().compile(raw_input, failure_evidence=failure_evidence)


def _ingest_loader(path):
    steps = [FakeStep(0, "user", "修 bug"),
             FakeStep(1, "assistant", "改这里", tool_call_name="Edit",
                      tool_call_args="foo.py")]
    return [FakeTrajectory("traj_evi", steps)]


async def test_run_ingest_structured_line_passes_evidence(tmp_path):
    store = FakeProblemStore(nearest_result=None)
    tstore = FakeTrajectoryStore()
    compiler = CapturingCompiler()
    deps = PipelineDeps(
        compiler=compiler, pipeline=FakePipeline(), select_fn=fake_select,
        load_trajectories_fn=_ingest_loader, problem_store=store,
        trajectory_store=tstore, embedder=FakeEmbedder(),
    )
    view, _ = await run_ingest(
        manifest_lines=['{"question": "重复读文件", "failure_trajectory": "traj/p.json"}'],
        deps=deps, emit=lambda e: None)
    assert view["summary"]["added"] == 1
    assert len(compiler.evidence_seen) == 1
    assert compiler.evidence_seen[0] is not None
    assert "traj_evi" in compiler.evidence_seen[0]
    # 物理隔离（决策 铁律1，显式断言）：失败轨迹只被压缩喂 compile，
    # 绝不进任何正例存储 —— problem_store 只 add question+spec，
    # trajectory_store 在无 --trajectory 输入时一次 upsert 都没有。
    assert len(store.add_calls) == 1
    assert store.add_calls[0][0] == "重复读文件"
    assert tstore.upsert_calls == []  # 失败轨迹绝不落 trajectory_store


async def test_run_ingest_plain_text_line_passes_none_evidence(tmp_path):
    store = FakeProblemStore(nearest_result=None)
    compiler = CapturingCompiler()
    deps = PipelineDeps(
        compiler=compiler, pipeline=FakePipeline(), select_fn=fake_select,
        load_trajectories_fn=_ingest_loader, problem_store=store,
        embedder=FakeEmbedder(),
    )
    await run_ingest(manifest_lines=["纯文本问题"], deps=deps, emit=lambda e: None)
    assert compiler.evidence_seen == [None]
