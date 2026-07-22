"""Problem Spec schema — strict alignment with interface-contract.md.

All enumerations, field names, and validation rules are frozen per contract.
Do NOT add, rename, or relax without updating the contract document.
"""
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 dengxuanliang

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "ProblemSpec",
    "SubProblem",
    "StructuredFilters",
    "DroppedSubProblem",
    "CapabilityRubric",
    "CAPABILITY_KINDS",
    "LabelEvidence",
    "LANGUAGES",
    "TOOLS_USED",
    "DROP_REASONS",
    "validate_problem_spec",
]

# ── 契约 §2 冻结枚举 ──

LANGUAGES = frozenset([
    "python", "cpp", "java", "javascript", "go", "bash", "html", "other",
])

TOOLS_USED = frozenset([
    "Bash", "Read", "Write", "Edit", "Glob", "Grep",
    "WebFetch", "WebSearch", "Task", "TodoWrite", "NotebookEdit", "other",
])

DROP_REASONS = frozenset([
    "ambiguous", "not_applicable", "label_diverged", "no_trajectory_evidence", "other",
])

# 契约 §1.5 冻结枚举 — CapabilityRubric.capability_kind 的合法取值
CAPABILITY_KINDS = frozenset({"presence", "avoidance", "recovery"})


@dataclass(frozen=True)
class CapabilityRubric:
    """契约 §1.5 CapabilityRubric — SubProblem.rubric 的类型（可选增强字段）。

    由 Call 2 从 raw_text/failure_summary 蒸出；判据卡方案引入。null 时
    module1 judge 回退旧 prompt（向后兼容）。
    """
    positive_criteria: list[str]   # 1-3 条可观测的"正向展示"判据
    negative_criteria: list[str]   # 1-3 条反例/失败模式
    decisive_evidence: str         # 决定性证据的"形态"描述；禁止硬编码检测器规则
    capability_kind: str           # one of CAPABILITY_KINDS


@dataclass(frozen=True)
class LabelEvidence:
    """契约 §1.6 LabelEvidence — SubProblem.failure_evidence 的类型（可选增强字段）。

    失败轨迹接入引入。仅当输入 manifest 为该问题提供失败轨迹、且 Call 2
    认领到证据步时非 None；无轨迹或纯文本路径恒 None。
    """
    trajectory_id: str             # 失败轨迹 id
    evidence_steps: list[int]      # 标签据以判定的失败轨迹步号
    observed_failure: str          # 现场观测到的真实失败


@dataclass
class StructuredFilters:
    """契约 §1.3 StructuredFilters — 3 fields, all optional."""
    languages: list[str] | None = None
    tools_used: list[str] | None = None
    has_verification_step: bool | None = None

    def __post_init__(self):
        if self.languages:
            for v in self.languages:
                if v not in LANGUAGES:
                    raise ValueError(f"languages: invalid value '{v}'. Must be one of {sorted(LANGUAGES)}")
        if self.tools_used:
            for v in self.tools_used:
                if v not in TOOLS_USED:
                    raise ValueError(f"tools_used: invalid value '{v}'. Must be one of {sorted(TOOLS_USED)}")


@dataclass
class SubProblem:
    """契约 §1.2 SubProblem — 12 required fields.

    rubric / failure_evidence 是可选增强字段（缺省 None，不计入 12 个必填字段）：
    缺省时行为与改动前逐字一致（judge 走旧 prompt、无标签证据）。
    """
    id: str
    origin: str  # "original" | "clarified"
    parent_id: str | None
    raw_text: str
    failure_summary: str
    target_capability: list[str]  # 1-3 labels
    trajectory_signal: str
    hyde_positive: list[str]  # 2-3 segments
    keywords: list[str]
    structured_filters: StructuredFilters
    confidence: float  # 0-1
    route: str  # "pass" in sub_problems[]
    rubric: CapabilityRubric | None = None       # 契约 §1.5，可选；缺省 None
    failure_evidence: LabelEvidence | None = None  # 契约 §1.6，可选；缺省 None

    def __post_init__(self):
        if self.origin not in ("original", "clarified"):
            raise ValueError(f"origin must be 'original' or 'clarified', got '{self.origin}'")
        if self.origin == "clarified" and not self.parent_id:
            raise ValueError("parent_id is required when origin=='clarified'")
        if not (1 <= len(self.target_capability) <= 3):
            raise ValueError(f"target_capability must have 1-3 items, got {len(self.target_capability)}")
        if not (2 <= len(self.hyde_positive) <= 3):
            raise ValueError(f"hyde_positive must have 2-3 segments, got {len(self.hyde_positive)}")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"confidence must be 0-1, got {self.confidence}")
        if self.route != "pass":
            raise ValueError(f"route must be 'pass' in sub_problems[], got '{self.route}'")


@dataclass
class DroppedSubProblem:
    """契约 §1.4 — dropped sub-problems for audit (not in sub_problems[]).

    Note: NO length validation on target_capability/hyde_positive here —
    dropped problems may have incomplete fields. Only route + drop_reason
    are validated.
    """
    id: str
    origin: str
    parent_id: str | None
    raw_text: str
    failure_summary: str
    target_capability: list[str]
    trajectory_signal: str
    hyde_positive: list[str]
    keywords: list[str]
    structured_filters: StructuredFilters
    confidence: float
    route: str  # "drop"
    drop_reason: str  # one of DROP_REASONS

    def __post_init__(self):
        if self.route != "drop":
            raise ValueError(f"route must be 'drop' for DroppedSubProblem, got '{self.route}'")
        if self.drop_reason not in DROP_REASONS:
            raise ValueError(f"drop_reason must be one of {sorted(DROP_REASONS)}, got '{self.drop_reason}'")


@dataclass
class ProblemSpec:
    """契约 §1.1 顶层结构。"""
    raw_input: str
    domain: str  # 固定 "agentic_swe"
    sub_problems: list[SubProblem]


def _parse_structured_filters(d: dict | None) -> StructuredFilters:
    if not d:
        return StructuredFilters()
    return StructuredFilters(
        languages=d.get("languages"),
        tools_used=d.get("tools_used"),
        has_verification_step=d.get("has_verification_step"),
    )


def _parse_rubric(d: dict | None) -> CapabilityRubric | None:
    """Rebuild CapabilityRubric from nested dict; None (or missing) stays None."""
    if not d:
        return None
    return CapabilityRubric(
        positive_criteria=d["positive_criteria"],
        negative_criteria=d["negative_criteria"],
        decisive_evidence=d["decisive_evidence"],
        capability_kind=d["capability_kind"],
    )


def _parse_failure_evidence(d: dict | None) -> LabelEvidence | None:
    """Rebuild LabelEvidence from nested dict; None (or missing) stays None."""
    if not d:
        return None
    return LabelEvidence(
        trajectory_id=d["trajectory_id"],
        evidence_steps=d["evidence_steps"],
        observed_failure=d["observed_failure"],
    )


def validate_problem_spec(data: dict) -> ProblemSpec:
    """Parse and validate a raw dict into a ProblemSpec.

    Raises ValueError on any schema violation.
    """
    sub_problems = []
    for sp_data in data.get("sub_problems", []):
        if sp_data.get("route") != "pass":
            raise ValueError(f"route must be 'pass' in sub_problems[], got '{sp_data.get('route')}'")
        sub_problems.append(SubProblem(
            id=sp_data["id"],
            origin=sp_data["origin"],
            parent_id=sp_data.get("parent_id"),
            raw_text=sp_data["raw_text"],
            failure_summary=sp_data["failure_summary"],
            target_capability=sp_data["target_capability"],
            trajectory_signal=sp_data["trajectory_signal"],
            hyde_positive=sp_data["hyde_positive"],
            keywords=sp_data["keywords"],
            structured_filters=_parse_structured_filters(sp_data.get("structured_filters")),
            confidence=sp_data["confidence"],
            route=sp_data["route"],
            rubric=_parse_rubric(sp_data.get("rubric")),
            failure_evidence=_parse_failure_evidence(sp_data.get("failure_evidence")),
        ))
    return ProblemSpec(
        raw_input=data["raw_input"],
        domain=data["domain"],
        sub_problems=sub_problems,
    )
