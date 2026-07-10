"""Data models for the trajectory pipeline."""

from __future__ import annotations
from dataclasses import dataclass, field

__all__ = ["Step", "Trajectory", "Slice", "TrajectorySignature", "JudgeResult", "SFTCandidate"]


@dataclass
class Step:
    index: int
    role: str
    content: str
    tool_call_name: str | None = None
    tool_call_args: str | None = None
    tool_result: str | None = None


@dataclass
class Trajectory:
    id: str
    steps: list[Step]
    raw_messages: list[dict]

    @property
    def step_count(self) -> int:
        return len(self.steps)


@dataclass
class Slice:
    trajectory_id: str
    slice_index: int
    steps: list[Step]
    start_step: int
    end_step: int

    @property
    def step_count(self) -> int:
        return len(self.steps)


@dataclass
class TrajectorySignature:
    trajectory_id: str
    slice_index: int
    step_range: tuple[int, int]
    step_count: int
    turn_count: int
    languages: list[str]
    tools_used: list[str]
    has_error_pattern: bool
    has_success_pattern: bool
    has_verification_step: bool
    bm25_tokens: list[str]
    embedding: list[float] = field(default_factory=list)
    capability_labels: list[str] | None = None


@dataclass
class JudgeResult:
    match: bool
    confidence: float
    spans: list[dict]
    reasoning: str = ""


@dataclass
class SFTCandidate:
    trajectory_id: str
    trajectory_path: str
    matched_problems: list[dict]
