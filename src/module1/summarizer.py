"""Summarize a slice into compact text for LLM judge.

Each step is compressed to one line. Preserves enough operation content
for the judge to determine "was this capability correctly demonstrated".

Truncation limits (spec §5):
  - assistant reasoning: first 150 chars
  - tool_call args: first 200 chars
  - tool_result: first 300 chars
"""

from __future__ import annotations

from module1.models import Slice, Step

__all__ = ["summarize_slice"]

_ASSISTANT_LIMIT = 150
_ARGS_LIMIT = 200
_RESULT_LIMIT = 300


def summarize_slice(slice_obj: Slice) -> str:
    """Compress a slice into a multi-line summary, one line per step."""
    if not slice_obj.steps:
        return ""

    lines = []
    for step in slice_obj.steps:
        line = _summarize_step(step)
        if line:
            lines.append(line)
    return "\n".join(lines)


def _summarize_step(step: Step) -> str:
    """Compress a single step to one line."""
    idx = step.index

    if step.role == "assistant":
        reasoning = _truncate(step.content, _ASSISTANT_LIMIT)
        if step.tool_call_name:
            args = _truncate(step.tool_call_args or "", _ARGS_LIMIT)
            return f"Step {idx}: [assistant] {reasoning} → call {step.tool_call_name}(\"{args}\")"
        else:
            return f"Step {idx}: [assistant] {reasoning}"

    elif step.role == "tool":
        result = _truncate(step.tool_result or step.content, _RESULT_LIMIT)
        return f"Step {idx}: [result] {result}"

    elif step.role == "user":
        content = _truncate(step.content, _ASSISTANT_LIMIT)
        return f"Step {idx}: [user] {content}"

    elif step.role == "system":
        content = _truncate(step.content, _ASSISTANT_LIMIT)
        return f"Step {idx}: [system] {content}"

    return ""


def _truncate(text: str, limit: int) -> str:
    """Truncate text and escape newlines."""
    text = text.replace("\n", "\\n")
    if len(text) > limit:
        return text[:limit] + "..."
    return text
