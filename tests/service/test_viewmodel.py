# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field

from service.viewmodel import build_inspector_view


@dataclass
class FakeScored:
    trajectory_id: str
    slice_index: int
    sub_problem_id: str
    capability: list
    relevance_score: float
    judge_confidence: float
    loss_mask_spans: list
    judge_match: bool = True
    evidence_step: int | None = None
    criteria_hit: list = field(default_factory=list)


def _spec():
    # 最小 ProblemSpec dict（contract §1 形态）
    return {
        "raw_input": "x",
        "domain": "agentic_swe",
        "sub_problems": [
            {
                "id": "p1",
                "failure_summary": "写入 py 文件语法错误",
                "target_capability": ["valid_syntax_in_toolcall", "self_verification"],
                "confidence": 0.9,
            }
        ],
    }


def test_view_maps_problems_and_capabilities():
    scored = [
        FakeScored("t1", 0, "p1", ["valid_syntax_in_toolcall"], 0.8, 0.9,
                   [{"start_step": 1, "end_step": 2}]),
    ]
    select_result = {
        "targeted": [scored[0]],
        "manifest": {"targeted_count": 1, "general_count": 0, "general_ratio": 0.3},
    }
    view = build_inspector_view(
        run_id="r1", spec=_spec(), scored=scored, select_result=select_result
    )
    assert view["run_id"] == "r1"
    assert len(view["problems"]) == 1
    p = view["problems"][0]
    assert p["id"] == "p1"
    assert p["failure_summary"].startswith("写入")
    assert p["confidence"] == 0.9
    caps = {c["label"]: c for c in p["capabilities"]}
    assert set(caps) == {"valid_syntax_in_toolcall", "self_verification"}
    # 每个能力有稳定色
    assert caps["valid_syntax_in_toolcall"]["color"].startswith("#")
    # taxonomy parent 富集为后续保留（spec §10）
    assert caps["valid_syntax_in_toolcall"]["parent"] is None
    # 命中轨迹挂到对应能力下
    vs = caps["valid_syntax_in_toolcall"]
    assert vs["hit_count"] == 1
    hit = vs["hit_trajectories"][0]
    assert hit["trajectory_id"] == "t1"
    assert hit["slice_index"] == 0
    assert hit["relevance_score"] == 0.8
    assert hit["judge_confidence"] == 0.9
    assert hit["loss_mask_spans"] == [{"start_step": 1, "end_step": 2}]
    # judge 判定是按 (轨迹, slice, sub_problem) 定级的，不是按单个能力；
    # 同一 sub_problem 下所有 target_capability 共享同一份命中集合（issue 5）
    assert caps["self_verification"]["hit_count"] == 1
    assert caps["self_verification"]["hit_trajectories"] == vs["hit_trajectories"]


def test_selected_flag_reflects_module3_targeted():
    in_set = FakeScored("t1", 0, "p1", ["valid_syntax_in_toolcall"], 0.8, 0.9,
                        [{"start_step": 0, "end_step": 1}])
    out_set = FakeScored("t2", 0, "p1", ["valid_syntax_in_toolcall"], 0.5, 0.6,
                         [{"start_step": 0, "end_step": 1}])
    select_result = {
        "targeted": [in_set],  # 只有 t1 进选集
        "manifest": {"targeted_count": 1, "general_count": 0, "general_ratio": 0.3},
    }
    view = build_inspector_view(
        run_id="r1", spec=_spec(), scored=[in_set, out_set],
        select_result=select_result,
    )
    hits = view["problems"][0]["capabilities"][0]["hit_trajectories"]
    by_traj = {h["trajectory_id"]: h for h in hits}
    assert by_traj["t1"]["selected"] is True
    assert by_traj["t2"]["selected"] is False


def test_selected_keyed_on_slice_index_not_just_trajectory():
    # 同 trajectory_id + 同 sub_problem_id，仅 slice_index 不同：
    # 只有 slice 0 进 targeted，锁定 selected 按 3-tuple 键控而非 trajectory_id-only。
    s0 = FakeScored("t1", 0, "p1", ["valid_syntax_in_toolcall"], 0.8, 0.9,
                    [{"start_step": 0, "end_step": 1}])
    s1 = FakeScored("t1", 1, "p1", ["valid_syntax_in_toolcall"], 0.7, 0.8,
                    [{"start_step": 2, "end_step": 3}])
    select_result = {"targeted": [s0],  # 只有 slice 0 进选集
                     "manifest": {"targeted_count": 1, "general_count": 0, "general_ratio": 0.3}}
    view = build_inspector_view(run_id="r1", spec=_spec(), scored=[s0, s1],
                                select_result=select_result)
    hits = {h["slice_index"]: h for h in
            view["problems"][0]["capabilities"][0]["hit_trajectories"]}
    assert hits[0]["selected"] is True
    assert hits[1]["selected"] is False  # 同 traj 同 sub_problem，仅 slice 不同 → 不串


def test_selected_reflected_across_subproblem_cards():
    from module3.merge import MergedCandidate
    # 同一 (t1, slice0) 覆盖 p1、p2，进 targeted → 两个子问题卡片下都 selected
    spec = {"raw_input": "x", "domain": "agentic_swe", "sub_problems": [
        {"id": "p1", "failure_summary": "a", "target_capability": ["c1"], "confidence": 0.9},
        {"id": "p2", "failure_summary": "b", "target_capability": ["c2"], "confidence": 0.9},
    ]}
    scored = [
        FakeScored("t1", 0, "p1", ["c1"], 0.9, 0.9, [{"start_step": 0, "end_step": 1}]),
        FakeScored("t1", 0, "p2", ["c2"], 0.7, 0.8, [{"start_step": 0, "end_step": 1}]),
    ]
    merged = MergedCandidate(
        trajectory_id="t1", slice_index=0, trajectory_path="/x.jsonl",
        sub_problem_ids=["p1", "p2"], relevance_by_problem={"p1": 0.9, "p2": 0.7},
        relevance_score=0.9, loss_mask_spans=[{"start_step": 0, "end_step": 1}],
        embedding=[1.0] + [0.0] * 1023, bm25_tokens=[])
    view = build_inspector_view(
        run_id="r", spec=spec, scored=scored,
        select_result={"targeted": [merged], "manifest": {"targeted_count": 1}})
    for p in view["problems"]:
        hit = p["capabilities"][0]["hit_trajectories"][0]
        assert hit["selected"] is True


def test_manifest_passthrough():
    select_result = {
        "targeted": [],
        "manifest": {"targeted_count": 0, "general_count": 5, "general_ratio": 0.25},
    }
    view = build_inspector_view(
        run_id="r1", spec=_spec(), scored=[], select_result=select_result
    )
    assert view["manifest"]["general_count"] == 5
    assert view["manifest"]["general_ratio"] == 0.25


def test_candidate_for_other_subproblem_is_ignored():
    # judge 按 (轨迹, slice, sub_problem) 定级，candidate.capability 总是整份
    # target_capability 列表（module2 rerank 的产物），因此 issue 5 修复后不再
    # 按单个 label 过滤；仍然存在的边界是 sub_problem_id —— 属于别的
    # sub_problem 的 candidate 不应挂到本 sub_problem 下。
    scored = [FakeScored("t1", 0, "other_problem", ["valid_syntax_in_toolcall"],
                         0.8, 0.9, [{"start_step": 0, "end_step": 1}])]
    select_result = {"targeted": [], "manifest": {
        "targeted_count": 0, "general_count": 0, "general_ratio": 0.3}}
    view = build_inspector_view(
        run_id="r1", spec=_spec(), scored=scored, select_result=select_result
    )
    total_hits = sum(c["hit_count"] for c in view["problems"][0]["capabilities"])
    assert total_hits == 0  # 不同 sub_problem 的候选不挂到这里


def test_inspector_view_surfaces_evidence_step_and_criteria_hit():
    """PR-3：InspectorView 从 ScoredCandidate 带出 evidence_step/criteria_hit。"""
    scored = [FakeScored("t1", 0, "p1", ["valid_syntax_in_toolcall"], 0.8, 0.9,
                         [{"start_step": 1, "end_step": 2}],
                         evidence_step=2, criteria_hit=["写出可运行代码"])]
    select_result = {"targeted": [], "manifest": {
        "targeted_count": 0, "general_count": 0, "general_ratio": 0.3}}
    view = build_inspector_view(
        run_id="r1", spec=_spec(), scored=scored, select_result=select_result
    )
    hit = view["problems"][0]["capabilities"][0]["hit_trajectories"][0]
    assert hit["evidence_step"] == 2
    assert hit["criteria_hit"] == ["写出可运行代码"]


def test_runs_path_evidence_survives_real_rerank_to_view_e2e():
    """PR-3 /runs 传播链贯穿 e2e（防未来某一环重构断链）：用真实 rerank() 产真实
    ScoredCandidate，喂 build_inspector_view，断言 judge 的 evidence_step/criteria_hit
    一路活到人审界面。此路径不走 judge_cache（/runs 生产路径的可回溯性全靠这条链）。"""
    from module1.models import JudgeResult, TrajectorySignature
    from module1.store import RecallHit
    from module2.rerank import rerank

    sig = TrajectorySignature(
        trajectory_id="t1", slice_index=0, step_range=(0, 5), step_count=6,
        turn_count=1, languages=["python"], tools_used=["Bash"],
        has_error_pattern=False, has_success_pattern=True,
        has_verification_step=False, bm25_tokens=["python"], embedding=[0.1] * 1024,
    )
    hits = [RecallHit(signature=sig, rrf_score=1.0)]
    judged = [JudgeResult(match=True, confidence=0.9,
                          spans=[{"start_step": 1, "end_step": 2}],
                          evidence_step=2, criteria_hit=["写出可运行代码"])]
    sub = {"id": "p1", "target_capability": ["valid_syntax_in_toolcall"]}

    scored = rerank(hits, judged, sub, trajectory_path="/x.jsonl")  # 真实 ScoredCandidate
    select_result = {"targeted": [], "manifest": {
        "targeted_count": 0, "general_count": 0, "general_ratio": 0.3}}
    view = build_inspector_view(
        run_id="r1", spec=_spec(), scored=scored, select_result=select_result
    )
    hit = view["problems"][0]["capabilities"][0]["hit_trajectories"][0]
    assert hit["evidence_step"] == 2                    # judge → rerank → view，未断链
    assert hit["criteria_hit"] == ["写出可运行代码"]


def test_inspector_view_evidence_fields_default_when_absent():
    """老 ScoredCandidate 无 evidence_step/criteria_hit（getattr 降级）→ None/[]。"""
    @dataclass
    class LegacyScored:
        trajectory_id: str
        slice_index: int
        sub_problem_id: str
        capability: list
        relevance_score: float
        judge_confidence: float
        loss_mask_spans: list
        judge_match: bool = True

    scored = [LegacyScored("t1", 0, "p1", ["valid_syntax_in_toolcall"], 0.8, 0.9,
                           [{"start_step": 1, "end_step": 2}])]
    select_result = {"targeted": [], "manifest": {
        "targeted_count": 0, "general_count": 0, "general_ratio": 0.3}}
    view = build_inspector_view(
        run_id="r1", spec=_spec(), scored=scored, select_result=select_result
    )
    hit = view["problems"][0]["capabilities"][0]["hit_trajectories"][0]
    assert hit["evidence_step"] is None
    assert hit["criteria_hit"] == []


def test_build_trajectory_index():
    from service.viewmodel import build_trajectory_index

    @dataclass
    class FakeStep:
        index: int
        role: str
        content: str
        tool_call_name: object = None
        tool_call_args: object = None
        tool_result: object = None

    @dataclass
    class FakeTraj:
        id: str
        steps: list = field(default_factory=list)

    trajs = [FakeTraj("t1", [FakeStep(0, "user", "hi")])]
    idx = build_trajectory_index(trajs)
    assert "t1" in idx
    assert idx["t1"]["trajectory_id"] == "t1"
    assert idx["t1"]["steps"][0]["role"] == "user"
    assert idx["t1"]["steps"][0]["content"] == "hi"


def test_problem_exposes_full_module0_compile_output():
    spec = {
        "raw_input": "x", "domain": "agentic_swe",
        "sub_problems": [{
            "id": "p1", "failure_summary": "s", "confidence": 0.9,
            "origin": "original", "route": "pass", "raw_text": "原始抱怨片段",
            "target_capability": ["valid_syntax_in_toolcall"],
            "trajectory_signal": "grep SyntaxError in tool output",
            "keywords": ["SyntaxError", "colon"],
            "hyde_positive": ["段一：正确做法……", "段二：验证……"],
            "structured_filters": {"languages": ["python"], "tools_used": ["Edit"],
                                   "has_verification_step": True},
        }],
    }
    view = build_inspector_view(
        run_id="r1", spec=spec, scored=[],
        select_result={"targeted": [], "manifest": {}},
    )
    c = view["problems"][0]["compile"]
    assert c["raw_text"] == "原始抱怨片段"
    assert c["origin"] == "original"
    assert "route" not in c
    assert c["trajectory_signal"].startswith("grep")
    assert c["keywords"] == ["SyntaxError", "colon"]
    assert len(c["hyde_positive"]) == 2
    assert c["structured_filters"]["languages"] == ["python"]
    assert c["structured_filters"]["has_verification_step"] is True
    assert c["target_capability"] == ["valid_syntax_in_toolcall"]


def test_hit_count_excludes_judge_miss():
    # a judge miss (judge_match=False) must NOT appear as a hit
    miss = FakeScored("t1", 0, "p1", ["valid_syntax_in_toolcall"], 0.3, 0.2,
                      [{"start_step": 0, "end_step": 1}])
    miss.judge_match = False
    view = build_inspector_view(
        run_id="r", spec=_spec(), scored=[miss],
        select_result={"targeted": [], "manifest": {}})
    caps = view["problems"][0]["capabilities"]
    assert all(c["hit_count"] == 0 for c in caps)


def test_empty_spans_excluded():
    # judge_match True but empty spans → not a real demonstration, exclude
    nospan = FakeScored("t1", 0, "p1", ["valid_syntax_in_toolcall"], 0.8, 0.9, [])
    nospan.judge_match = True
    view = build_inspector_view(
        run_id="r", spec=_spec(), scored=[nospan],
        select_result={"targeted": [], "manifest": {}})
    caps = view["problems"][0]["capabilities"]
    assert all(c["hit_count"] == 0 for c in caps)


def test_multi_capability_no_double_count():
    # one sub_problem with 2 target capabilities, one matched slice carrying both
    # → both capabilities report the SAME single hit (not double-counted)
    c = FakeScored("t1", 0, "p1",
                   ["valid_syntax_in_toolcall", "self_verification"], 0.8, 0.9,
                   [{"start_step": 1, "end_step": 2}])
    c.judge_match = True
    view = build_inspector_view(
        run_id="r", spec=_spec(), scored=[c],
        select_result={"targeted": [], "manifest": {}})
    caps = view["problems"][0]["capabilities"]
    assert len(caps) == 2
    assert all(cap["hit_count"] == 1 for cap in caps)
    # both capabilities point at the same single (traj, slice) hit
    ids = {(h["trajectory_id"], h["slice_index"])
           for cap in caps for h in cap["hit_trajectories"]}
    assert ids == {("t1", 0)}


def test_subproblem_without_id_skipped():
    spec = {"raw_input": "x", "domain": "agentic_swe", "sub_problems": [
        {"failure_summary": "no id", "target_capability": ["x"], "confidence": 0.9},
        {"id": "p1", "failure_summary": "ok", "target_capability": [], "confidence": 0.9},
    ]}
    view = build_inspector_view(run_id="r", spec=spec, scored=[],
                                select_result={"targeted": [], "manifest": {}})
    assert [p["id"] for p in view["problems"]] == ["p1"]  # id-less one skipped


def test_compile_has_no_route_field():
    view = build_inspector_view(run_id="r", spec=_spec(), scored=[],
                                select_result={"targeted": [], "manifest": {}})
    assert "route" not in view["problems"][0]["compile"]
