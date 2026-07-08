import pytest
from module0.schema import (
    ProblemSpec,
    SubProblem,
    StructuredFilters,
    DroppedSubProblem,
    LANGUAGES,
    TOOLS_USED,
    OUTCOME_TRANSITIONS,
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
    assert "failed→success" in OUTCOME_TRANSITIONS
    assert len(OUTCOME_TRANSITIONS) == 4
    assert "ambiguous" in DROP_REASONS
    assert len(DROP_REASONS) == 4


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
