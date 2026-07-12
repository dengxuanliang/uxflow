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
    # self_verification 无命中
    assert caps["self_verification"]["hit_count"] == 0


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


def test_candidate_capability_not_in_subproblem_is_ignored():
    # candidate 报了一个不在该 sub_problem target_capability 里的 label → 不挂
    scored = [FakeScored("t1", 0, "p1", ["some_other_label"], 0.8, 0.9,
                         [{"start_step": 0, "end_step": 1}])]
    select_result = {"targeted": [], "manifest": {
        "targeted_count": 0, "general_count": 0, "general_ratio": 0.3}}
    view = build_inspector_view(
        run_id="r1", spec=_spec(), scored=scored, select_result=select_result
    )
    total_hits = sum(c["hit_count"] for c in view["problems"][0]["capabilities"])
    assert total_hits == 0  # 无匹配能力，忽略该命中


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
