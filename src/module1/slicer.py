"""Trajectory slicing: ≤10 steps whole, >10 steps semantic split.

Boundary signals (spec §2.2):
- User message (highest priority)
- Tool type switch (explore→modify→verify)
- Subtask transition phrases
- Verification step

Algorithm: greedy split at highest-scoring boundary every 5-10 steps.
"""

from __future__ import annotations

import re

from module1.models import Step, Trajectory, Slice

__all__ = ["slice_trajectory"]

_STEP_THRESHOLD = 10
_MIN_SLICE_SIZE = 4
_MAX_SLICE_SIZE = 10

_EXPLORE_TOOLS = {"Read", "Grep", "Glob", "WebFetch", "WebSearch"}
_MODIFY_TOOLS = {"Write", "Edit"}
_VERIFY_KEYWORDS = re.compile(r"test|check|pytest|run|verify|build", re.IGNORECASE)
_TRANSITION_PHRASES = re.compile(
    r"现在|接下来|然后来|Let me now|Next,? I|Now I|Moving on", re.IGNORECASE
)


def slice_trajectory(trajectory: Trajectory) -> list[Slice]:
    """Slice a trajectory into matching units."""
    if not trajectory.steps:
        return []

    if len(trajectory.steps) <= _STEP_THRESHOLD:
        return [Slice(
            trajectory_id=trajectory.id,
            slice_index=0,
            steps=trajectory.steps,
            start_step=0,
            end_step=len(trajectory.steps) - 1,
        )]

    boundaries = _find_boundaries(trajectory.steps)
    return _split_at_boundaries(trajectory, boundaries)


def _score_boundary(steps: list[Step], idx: int) -> float:
    """Score a potential boundary point (between idx-1 and idx)."""
    if idx <= 0 or idx >= len(steps):
        return 0.0

    step = steps[idx]
    score = 0.0

    # User message = strongest boundary
    if step.role == "user":
        score += 10.0

    # Tool type switch
    if idx > 0:
        prev_tool = steps[idx - 1].tool_call_name or ""
        curr_tool = step.tool_call_name or ""
        prev_type = _tool_type(prev_tool)
        curr_type = _tool_type(curr_tool)
        if prev_type and curr_type and prev_type != curr_type:
            score += 5.0

    # Transition phrase in assistant content
    if step.role == "assistant" and _TRANSITION_PHRASES.search(step.content[:200]):
        score += 4.0

    # Verification step
    if step.tool_call_name == "Bash" and step.tool_call_args and _VERIFY_KEYWORDS.search(step.tool_call_args):
        score += 3.0

    return score


def _tool_type(name: str) -> str | None:
    if name in _EXPLORE_TOOLS:
        return "explore"
    if name in _MODIFY_TOOLS:
        return "modify"
    if name == "Bash":
        return "verify"
    return None


def _find_boundaries(steps: list[Step]) -> list[tuple[int, float]]:
    """Find and score all candidate boundary points."""
    candidates = []
    for i in range(1, len(steps)):
        score = _score_boundary(steps, i)
        if score > 0:
            candidates.append((i, score))
    return sorted(candidates, key=lambda x: -x[1])


def _split_at_boundaries(trajectory: Trajectory, boundaries: list[tuple[int, float]]) -> list[Slice]:
    """Greedy split: pick best boundaries that keep slices within size range."""
    n = len(trajectory.steps)
    # Collect valid split points
    split_points = set()
    for idx, _score in boundaries:
        split_points.add(idx)

    # Greedy: walk forward, split at best available boundary in window
    cuts = [0]
    pos = 0
    while pos < n:
        # Target next cut at pos + _MAX_SLICE_SIZE
        window_end = min(pos + _MAX_SLICE_SIZE, n)
        window_start = pos + _MIN_SLICE_SIZE

        # Find best boundary in [window_start, window_end]
        best = None
        best_score = -1
        for idx, score in boundaries:
            if window_start <= idx <= window_end and idx not in cuts:
                if score > best_score:
                    best = idx
                    best_score = score

        if best is not None:
            cuts.append(best)
            pos = best
        else:
            # No good boundary; force cut at max
            if window_end < n:
                cuts.append(window_end)
                pos = window_end
            else:
                break

    # Build slices from cuts
    slices = []
    for i in range(len(cuts)):
        start = cuts[i]
        end = cuts[i + 1] - 1 if i + 1 < len(cuts) else n - 1
        slices.append(Slice(
            trajectory_id=trajectory.id,
            slice_index=i,
            steps=trajectory.steps[start:end + 1],
            start_step=start,
            end_step=end,
        ))
    return slices
