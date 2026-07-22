import pytest
from module0.schema import (
    SubProblem,
    StructuredFilters,
    DroppedSubProblem,
    CapabilityRubric,
    CAPABILITY_KINDS,
    LabelEvidence,
    LANGUAGES,
    TOOLS_USED,
    DROP_REASONS,
    validate_problem_spec,
)


def test_structured_filters_enums():
    assert "python" in LANGUAGES
    assert "other" in LANGUAGES
    assert len(LANGUAGES) == 8
    assert "Bash" in TOOLS_USED
    assert "other" in TOOLS_USED
    assert len(TOOLS_USED) == 12
    assert "ambiguous" in DROP_REASONS
    assert "no_trajectory_evidence" in DROP_REASONS
    assert len(DROP_REASONS) == 5


def test_dropped_sub_problem_no_trajectory_evidence_reason():
    # PR-2: 失败轨迹认领不到证据步 → drop_reason=no_trajectory_evidence 构造合法
    d = DroppedSubProblem(
        id="p9", origin="original", parent_id=None,
        raw_text="t", failure_summary="s",
        target_capability=["x"], trajectory_signal="s",
        hyde_positive=["h1", "h2"], keywords=["k"],
        structured_filters=StructuredFilters(),
        confidence=0.4, route="drop",
        drop_reason="no_trajectory_evidence",
    )
    assert d.drop_reason == "no_trajectory_evidence"


def test_sub_problem_valid():
    sp = SubProblem(
        id="p1",
        origin="original",
        parent_id=None,
        raw_text="test",
        failure_summary="test summary",
        target_capability=["valid_syntax_in_toolcall"],
        trajectory_signal="signal",
        hyde_positive=["hyp1", "hyp2"],
        keywords=["kw1"],
        structured_filters=StructuredFilters(languages=["python"]),
        confidence=0.9,
        route="pass",
    )
    assert sp.id == "p1"
    assert sp.origin == "original"


def test_sub_problem_clarified_requires_parent_id():
    with pytest.raises(ValueError, match="parent_id"):
        SubProblem(
            id="p1a",
            origin="clarified",
            parent_id=None,
            raw_text="t",
            failure_summary="s",
            target_capability=["x"],
            trajectory_signal="s",
            hyde_positive=["h1", "h2"],
            keywords=["k"],
            structured_filters=StructuredFilters(),
            confidence=0.85,
            route="pass",
        )


def test_sub_problem_target_capability_1_to_3():
    with pytest.raises(ValueError, match="target_capability"):
        SubProblem(
            id="p1", origin="original", parent_id=None,
            raw_text="t", failure_summary="s",
            target_capability=[],
            trajectory_signal="s", hyde_positive=["h1", "h2"],
            keywords=["k"], structured_filters=StructuredFilters(),
            confidence=0.9, route="pass",
        )


def test_structured_filters_validates_enums():
    with pytest.raises(ValueError, match="languages"):
        StructuredFilters(languages=["invalid_lang"])


def test_problem_spec_valid(golden_problem_spec):
    ps = validate_problem_spec(golden_problem_spec)
    assert ps.domain == "agentic_swe"
    assert len(ps.sub_problems) == 2
    assert ps.sub_problems[0].id == "p1"


def test_problem_spec_rejects_drop_in_sub_problems():
    with pytest.raises(ValueError, match="route"):
        validate_problem_spec({
            "raw_input": "test",
            "domain": "agentic_swe",
            "sub_problems": [{"id": "p1", "route": "drop"}],
        })


def test_dropped_sub_problem():
    d = DroppedSubProblem(
        id="p5", origin="original", parent_id=None,
        raw_text="t", failure_summary="s",
        target_capability=["x"], trajectory_signal="s",
        hyde_positive=["h1", "h2"], keywords=["k"],
        structured_filters=StructuredFilters(),
        confidence=0.5, route="drop",
        drop_reason="not_applicable",
    )
    assert d.route == "drop"
    assert d.drop_reason == "not_applicable"


# ── PR-1: 判据卡 / 标签证据（可选增强字段，契约 §1.5/§1.6）──


def test_capability_kinds_enum():
    assert CAPABILITY_KINDS == {"presence", "avoidance", "recovery"}


def test_capability_rubric_constructible_and_frozen():
    r = CapabilityRubric(
        positive_criteria=["writes a failing test first"],
        negative_criteria=["jumps straight to impl"],
        decisive_evidence="a test-authoring step preceding impl",
        capability_kind="presence",
    )
    assert r.capability_kind == "presence"
    assert r.positive_criteria == ["writes a failing test first"]
    with pytest.raises(Exception):  # frozen dataclass → FrozenInstanceError
        r.capability_kind = "recovery"


def test_label_evidence_constructible_and_frozen():
    e = LabelEvidence(
        trajectory_id="traj_007",
        evidence_steps=[3, 5],
        observed_failure="generated code with a SyntaxError",
    )
    assert e.trajectory_id == "traj_007"
    assert e.evidence_steps == [3, 5]
    with pytest.raises(Exception):
        e.observed_failure = "x"


def test_sub_problem_optional_fields_default_none():
    sp = SubProblem(
        id="p1", origin="original", parent_id=None,
        raw_text="t", failure_summary="s",
        target_capability=["x"], trajectory_signal="s",
        hyde_positive=["h1", "h2"], keywords=["k"],
        structured_filters=StructuredFilters(),
        confidence=0.9, route="pass",
    )
    assert sp.rubric is None
    assert sp.failure_evidence is None


def test_sub_problem_accepts_rubric_and_evidence():
    r = CapabilityRubric(
        positive_criteria=["p"], negative_criteria=["n"],
        decisive_evidence="d", capability_kind="recovery",
    )
    e = LabelEvidence(trajectory_id="t1", evidence_steps=[1], observed_failure="f")
    sp = SubProblem(
        id="p1", origin="original", parent_id=None,
        raw_text="t", failure_summary="s",
        target_capability=["x"], trajectory_signal="s",
        hyde_positive=["h1", "h2"], keywords=["k"],
        structured_filters=StructuredFilters(),
        confidence=0.9, route="pass",
        rubric=r, failure_evidence=e,
    )
    assert sp.rubric is r
    assert sp.failure_evidence is e


def test_validate_problem_spec_rebuilds_nested_rubric_and_evidence():
    data = {
        "raw_input": "test", "domain": "agentic_swe",
        "sub_problems": [{
            "id": "p1", "origin": "original", "parent_id": None,
            "raw_text": "t", "failure_summary": "s",
            "target_capability": ["x"], "trajectory_signal": "sig",
            "hyde_positive": ["h1", "h2"], "keywords": ["k"],
            "structured_filters": {"languages": ["python"]},
            "confidence": 0.9, "route": "pass",
            "rubric": {
                "positive_criteria": ["p"], "negative_criteria": ["n"],
                "decisive_evidence": "d", "capability_kind": "avoidance",
            },
            "failure_evidence": {
                "trajectory_id": "traj_1", "evidence_steps": [2, 4],
                "observed_failure": "obs",
            },
        }],
    }
    ps = validate_problem_spec(data)
    sp = ps.sub_problems[0]
    assert isinstance(sp.rubric, CapabilityRubric)
    assert sp.rubric.capability_kind == "avoidance"
    assert isinstance(sp.failure_evidence, LabelEvidence)
    assert sp.failure_evidence.evidence_steps == [2, 4]


def test_validate_problem_spec_missing_optional_fields_stay_none():
    data = {
        "raw_input": "test", "domain": "agentic_swe",
        "sub_problems": [{
            "id": "p1", "origin": "original", "parent_id": None,
            "raw_text": "t", "failure_summary": "s",
            "target_capability": ["x"], "trajectory_signal": "sig",
            "hyde_positive": ["h1", "h2"], "keywords": ["k"],
            "structured_filters": {},
            "confidence": 0.9, "route": "pass",
            # no rubric / failure_evidence keys at all
        }],
    }
    ps = validate_problem_spec(data)
    assert ps.sub_problems[0].rubric is None
    assert ps.sub_problems[0].failure_evidence is None


def test_validate_problem_spec_explicit_null_optional_fields_stay_none():
    data = {
        "raw_input": "test", "domain": "agentic_swe",
        "sub_problems": [{
            "id": "p1", "origin": "original", "parent_id": None,
            "raw_text": "t", "failure_summary": "s",
            "target_capability": ["x"], "trajectory_signal": "sig",
            "hyde_positive": ["h1", "h2"], "keywords": ["k"],
            "structured_filters": {},
            "confidence": 0.9, "route": "pass",
            "rubric": None, "failure_evidence": None,
        }],
    }
    ps = validate_problem_spec(data)
    assert ps.sub_problems[0].rubric is None
    assert ps.sub_problems[0].failure_evidence is None
