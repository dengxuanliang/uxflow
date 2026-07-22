# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field

from service.orchestrator import run_pipeline, PipelineDeps
from service.orchestrator import _spec_to_dict, _parse_manifest_line
from service.orchestrator import _compress_failure_trajectory
from module1.models import Step, Trajectory


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
    # empty defaults so real module3 dedup/select can read them (cosine + MinHash)
    embedding: list = field(default_factory=list)
    bm25_tokens: list = field(default_factory=list)


class FakeCompiler:
    def __init__(self):
        self._n = 0
        self.evidence_seen = []  # PR-2: 记录每次 compile 收到的 failure_evidence

    async def compile(self, raw_input, *, failure_evidence=None):
        self._n += 1
        self.evidence_seen.append(failure_evidence)
        sp = FakeSubProblem(
            id="p1",
            failure_summary=f"summary::{raw_input}",
            target_capability=["valid_syntax_in_toolcall"],
        )
        # add structured_filters attribute expected by serializer
        sp.structured_filters = FakeFilters()
        return FakeSpec(raw_input=raw_input, domain="agentic_swe", sub_problems=[sp])


class FakePipeline:
    async def run_scored(self, *, trajectory_paths, problem_specs, on_progress=None):
        # one hit per sub_problem; mirror real run_scored's per-sub_problem
        # on_progress callback so the orchestrator's progress wiring is exercised
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


def fake_select(candidates, *, sub_problem_ids, selection, general):
    return {
        "targeted": list(candidates),
        "general": [],
        "manifest": {"targeted_count": len(candidates),
                     "general_count": 0, "general_ratio": 0.3},
    }


def _deps():
    return PipelineDeps(
        compiler=FakeCompiler(),
        pipeline=FakePipeline(),
        select_fn=fake_select,
        load_trajectories_fn=lambda p: [],
    )


async def test_run_pipeline_emits_stages_and_builds_view(tmp_path):
    traj = tmp_path / "t.jsonl"
    traj.write_text('{"id":"t1","messages":[]}\n')
    events = []
    view, trajectories = await run_pipeline(
        manifest_lines=["代码总有语法错误"],
        trajectory_path=traj,
        deps=_deps(),
        emit=lambda ev: events.append(ev),
    )
    stages = [e["stage"] for e in events]
    assert "module0" in stages
    assert "module1" in stages
    assert "module3" in stages
    assert view["problems"][0]["id"].endswith("p1")
    assert view["manifest"]["targeted_count"] == 1


async def test_multiline_manifest_prefixes_problem_ids(tmp_path):
    traj = tmp_path / "t.jsonl"
    traj.write_text('{"id":"t1","messages":[]}\n')
    view, _ = await run_pipeline(
        manifest_lines=["抱怨A", "抱怨B"],
        trajectory_path=traj,
        deps=_deps(),
        emit=lambda ev: None,
    )
    ids = [p["id"] for p in view["problems"]]
    assert len(ids) == 2
    assert len(set(ids)) == 2  # 跨行不碰撞
    assert all("p1" in i for i in ids)


async def test_blank_manifest_lines_skipped(tmp_path):
    traj = tmp_path / "t.jsonl"
    traj.write_text('{"id":"t1","messages":[]}\n')
    view, _ = await run_pipeline(
        manifest_lines=["抱怨A", "", "  "],
        trajectory_path=traj,
        deps=_deps(),
        emit=lambda ev: None,
    )
    assert len(view["problems"]) == 1  # 空行不编译


async def test_module1_emits_per_subproblem_judge_progress(tmp_path):
    # Two complaints → 2 sub_problems (FakeCompiler yields one each). The judge
    # phase must emit determinate "精判 i/N" events with index/total so the
    # frontend progress bar advances instead of freezing.
    traj = tmp_path / "t.jsonl"
    traj.write_text('{"id":"t1","messages":[]}\n')
    events = []
    await run_pipeline(
        manifest_lines=["抱怨A", "抱怨B"],
        trajectory_path=traj,
        deps=_deps(),
        emit=lambda ev: events.append(ev),
    )
    progress = [e for e in events
                if e["stage"] == "module1" and e.get("index", 0) >= 1]
    assert len(progress) == 2  # one per sub_problem
    assert progress[-1]["index"] == 2 and progress[-1]["total"] == 2
    assert "精判 2/2" in progress[-1]["msg"]


def _deps_real_select():
    from module3.pipeline import select_final_dataset
    return PipelineDeps(
        compiler=FakeCompiler(),
        pipeline=FakePipeline(),
        select_fn=select_final_dataset,   # 真实 module3
        load_trajectories_fn=lambda p: [],
    )


async def test_run_pipeline_with_real_module3_select(tmp_path):
    traj = tmp_path / "t.jsonl"
    traj.write_text('{"id":"t1","messages":[]}\n')
    view, _ = await run_pipeline(
        manifest_lines=["代码总有语法错误"],
        trajectory_path=traj,
        deps=_deps_real_select(),
        emit=lambda ev: None,
    )
    m = view["manifest"]
    assert set(m) >= {"targeted_count", "general_count", "general_ratio"}
    assert isinstance(m["targeted_count"], int) and m["targeted_count"] >= 0


async def test_orchestrator_survives_when_emit_would_be_called(tmp_path):
    # sanity: on_progress path executes without aborting the run (issue 6 guard
    # lives in real pipeline.run_scored; here we ensure the wiring is exercised)
    traj = tmp_path / "t.jsonl"
    traj.write_text('{"id":"t1","messages":[]}\n')
    events = []
    view, _ = await run_pipeline(
        manifest_lines=["a", "b"], trajectory_path=traj,
        deps=_deps(), emit=lambda ev: events.append(ev))
    assert view is not None


# ── PR-1: rubric / failure_evidence 序列化透传（_spec_to_dict）──


def test_spec_to_dict_rubric_none_serializes_null():
    from module0.schema import ProblemSpec, SubProblem, StructuredFilters
    sp = SubProblem(
        id="p1", origin="original", parent_id=None,
        raw_text="t", failure_summary="s",
        target_capability=["x"], trajectory_signal="sig",
        hyde_positive=["h1", "h2"], keywords=["k"],
        structured_filters=StructuredFilters(), confidence=0.9, route="pass",
    )
    spec = ProblemSpec(raw_input="ri", domain="agentic_swe", sub_problems=[sp])
    d = _spec_to_dict(spec, id_prefix="L1.")
    out = d["sub_problems"][0]
    assert out["rubric"] is None
    assert out["failure_evidence"] is None
    assert out["id"] == "L1.p1"


def test_spec_to_dict_rubric_present_serializes_nested_dict():
    from module0.schema import (
        ProblemSpec, SubProblem, StructuredFilters,
        CapabilityRubric, LabelEvidence,
    )
    sp = SubProblem(
        id="p1", origin="original", parent_id=None,
        raw_text="t", failure_summary="s",
        target_capability=["x"], trajectory_signal="sig",
        hyde_positive=["h1", "h2"], keywords=["k"],
        structured_filters=StructuredFilters(), confidence=0.9, route="pass",
        rubric=CapabilityRubric(
            positive_criteria=["p"], negative_criteria=["n"],
            decisive_evidence="d", capability_kind="presence"),
        failure_evidence=LabelEvidence(
            trajectory_id="traj_1", evidence_steps=[2], observed_failure="obs"),
    )
    spec = ProblemSpec(raw_input="ri", domain="agentic_swe", sub_problems=[sp])
    out = _spec_to_dict(spec, id_prefix="L1.")["sub_problems"][0]
    assert out["rubric"] == {
        "positive_criteria": ["p"], "negative_criteria": ["n"],
        "decisive_evidence": "d", "capability_kind": "presence",
    }
    assert out["failure_evidence"] == {
        "trajectory_id": "traj_1", "evidence_steps": [2], "observed_failure": "obs",
    }


def test_spec_to_dict_fake_without_optional_attrs_defaults_null():
    # 老 spec 对象/测试 fake 无 rubric 属性 → getattr 兜底为 None
    sp = FakeSubProblem(id="p1", failure_summary="s", target_capability=["x"])
    sp.structured_filters = FakeFilters()
    spec = FakeSpec(raw_input="ri", domain="agentic_swe", sub_problems=[sp])
    out = _spec_to_dict(spec, id_prefix="L1.")["sub_problems"][0]
    assert out["rubric"] is None
    assert out["failure_evidence"] is None


# ── PR-1: manifest JSONL 行解析（三种退化路径）──


def test_parse_manifest_line_plain_text():
    q, ft = _parse_manifest_line("代码总有语法错误")
    assert q == "代码总有语法错误"
    assert ft is None


def test_parse_manifest_line_structured_object_with_question():
    line = '{"question": "修复失败", "failure_trajectory": {"id": "t9"}}'
    q, ft = _parse_manifest_line(line)
    assert q == "修复失败"
    assert ft == {"id": "t9"}


def test_parse_manifest_line_structured_object_null_trajectory():
    line = '{"question": "q only", "failure_trajectory": null}'
    q, ft = _parse_manifest_line(line)
    assert q == "q only"
    assert ft is None


def test_parse_manifest_line_brace_but_not_valid_json_falls_back():
    line = "{this is not json but starts with brace}"
    q, ft = _parse_manifest_line(line)
    assert q == line
    assert ft is None


def test_parse_manifest_line_json_object_without_question_falls_back():
    line = '{"foo": "bar"}'
    q, ft = _parse_manifest_line(line)
    assert q == line  # 整行当纯文本
    assert ft is None


def test_parse_manifest_line_json_array_falls_back():
    line = '["a", "b"]'
    q, ft = _parse_manifest_line(line)
    assert q == line
    assert ft is None


def test_parse_manifest_line_non_string_question_falls_back():
    # question 非字符串（int/null/空串）→ 视为无效结构化行，退回整行当纯文本，
    # 防止非 str 注入 compile()/日志（run_pipeline 的编译循环不隔离单行异常）。
    for line in ('{"question": 123}', '{"question": null}', '{"question": ""}'):
        q, ft = _parse_manifest_line(line)
        assert q == line, f"应退回整行: {line}"
        assert ft is None


async def test_run_pipeline_structured_manifest_line_compiles_question(tmp_path):
    # 结构化行：编译输入应是 question，不是整行 JSON
    traj = tmp_path / "t.jsonl"
    traj.write_text('{"id":"t1","messages":[]}\n')
    seen = []

    class CapturingCompiler(FakeCompiler):
        async def compile(self, raw_input, *, failure_evidence=None):
            seen.append(raw_input)
            return await super().compile(raw_input, failure_evidence=failure_evidence)

    deps = PipelineDeps(
        compiler=CapturingCompiler(),
        pipeline=FakePipeline(),
        select_fn=fake_select,
        load_trajectories_fn=lambda p: [],
    )
    await run_pipeline(
        manifest_lines=['{"question": "结构化问题", "failure_trajectory": null}'],
        trajectory_path=traj, deps=deps, emit=lambda ev: None)
    assert seen == ["结构化问题"]


# ── PR-2: 失败轨迹压缩 _compress_failure_trajectory ────────────────────


def _real_load_trajectories(path):
    """按 path 内容回不同轨迹的假 loader（模拟 load_trajectories 签名）。"""
    steps = [
        Step(index=0, role="user", content="修一下这个 bug"),
        Step(index=1, role="assistant", content="我来看看",
             tool_call_name="Read", tool_call_args="foo.py"),
        Step(index=2, role="tool", content="def f(:\n  pass", tool_result="def f(:\n  pass"),
        Step(index=3, role="assistant", content="改这里",
             tool_call_name="Edit", tool_call_args="foo.py"),
    ]
    return [Trajectory(id="traj_evi", steps=steps, raw_messages=[])]


def test_compress_failure_trajectory_valid_path_produces_text():
    text = _compress_failure_trajectory("traj/p12.json", _real_load_trajectories)
    assert text is not None
    assert "traj_evi" in text
    assert "Step 0" in text  # summarizer 逐步格式


def test_compress_failure_trajectory_none_path_returns_none():
    # None / 非字符串 / 空串 → 无轨迹，返回 None（向后兼容）
    assert _compress_failure_trajectory(None, _real_load_trajectories) is None
    assert _compress_failure_trajectory({"id": "t"}, _real_load_trajectories) is None
    assert _compress_failure_trajectory("", _real_load_trajectories) is None


def test_compress_failure_trajectory_empty_load_returns_none():
    # 路径合法但 load 出空 → None（compiler 走纯文字）
    assert _compress_failure_trajectory("traj/x.json", lambda p: []) is None


def test_compress_failure_trajectory_truncates_over_limit():
    from service import orchestrator as orch

    def big_loader(path):
        # 单条超长轨迹：多步长内容，压缩后超字符上限
        steps = [Step(index=i, role="assistant", content="X" * 500,
                      tool_call_name="Bash", tool_call_args="Y" * 500)
                 for i in range(30)]
        return [Trajectory(id="big", steps=steps, raw_messages=[])]

    text = _compress_failure_trajectory("traj/big.json", big_loader)
    assert text is not None
    assert len(text) <= orch._MAX_EVIDENCE_CHARS + len("\n...(截断)")
    assert text.endswith("...(截断)")


async def test_run_pipeline_structured_line_compresses_and_passes_evidence(tmp_path):
    # 带 failure_trajectory 的结构化行 → 压缩并把非 None failure_evidence 传给 compile
    traj = tmp_path / "t.jsonl"
    traj.write_text('{"id":"t1","messages":[]}\n')
    compiler = FakeCompiler()
    deps = PipelineDeps(
        compiler=compiler,
        pipeline=FakePipeline(),
        select_fn=fake_select,
        load_trajectories_fn=_real_load_trajectories,  # 非空 → 有压缩文本
    )
    await run_pipeline(
        manifest_lines=['{"question": "重复读文件", "failure_trajectory": "traj/p.json"}'],
        trajectory_path=traj, deps=deps, emit=lambda ev: None)
    assert len(compiler.evidence_seen) == 1
    assert compiler.evidence_seen[0] is not None
    assert "traj_evi" in compiler.evidence_seen[0]


async def test_run_pipeline_plain_text_line_passes_none_evidence(tmp_path):
    # 纯文本行 → failure_evidence=None（load 不会被以失败轨迹身份调用）
    traj = tmp_path / "t.jsonl"
    traj.write_text('{"id":"t1","messages":[]}\n')
    compiler = FakeCompiler()
    deps = PipelineDeps(
        compiler=compiler,
        pipeline=FakePipeline(),
        select_fn=fake_select,
        load_trajectories_fn=_real_load_trajectories,
    )
    await run_pipeline(
        manifest_lines=["纯文本抱怨"],
        trajectory_path=traj, deps=deps, emit=lambda ev: None)
    assert compiler.evidence_seen == [None]


