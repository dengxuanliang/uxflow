# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field

from service.orchestrator import run_pipeline, PipelineDeps


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

    async def compile(self, raw_input):
        self._n += 1
        sp = FakeSubProblem(
            id="p1",
            failure_summary=f"summary::{raw_input}",
            target_capability=["valid_syntax_in_toolcall"],
        )
        # add structured_filters attribute expected by serializer
        sp.structured_filters = FakeFilters()
        return FakeSpec(raw_input=raw_input, domain="agentic_swe", sub_problems=[sp])


class FakePipeline:
    async def run_scored(self, *, trajectory_paths, problem_specs):
        # one hit per sub_problem
        out = []
        for spec in problem_specs:
            for sp in spec["sub_problems"]:
                out.append(FakeScored(
                    "t1", 0, "dummy.jsonl", sp["id"],
                    ["valid_syntax_in_toolcall"], 0.8, 0.9,
                    [{"start_step": 0, "end_step": 1}],
                ))
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

