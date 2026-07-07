"""Parse LLM raw text responses into structured dicts.

Handles: markdown fence stripping, JSON extraction, field validation,
and graceful error reporting. Does NOT do schema-level validation
(SubProblem/StructuredFilters) — that's compiler.py's job.
"""

from __future__ import annotations

import json
import re

__all__ = [
    "parse_call1_response",
    "parse_call2_response",
    "parse_call3_response",
    "ParseError",
]

_MAX_CLARIFIED_PER_PROBLEM = 4


class ParseError(ValueError):
    """Raised when LLM output cannot be parsed into expected structure."""
    pass


def _extract_json(text: str) -> str:
    """Strip markdown fences and surrounding text to find JSON."""
    if not text:
        raise ParseError("Empty response")
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    for i, c in enumerate(text):
        if c in ('{', '['):
            depth = 0
            open_c = c
            close_c = '}' if c == '{' else ']'
            for j in range(i, len(text)):
                if text[j] == open_c:
                    depth += 1
                elif text[j] == close_c:
                    depth -= 1
                    if depth == 0:
                        return text[i:j+1]
            break
    raise ParseError(f"No valid JSON found in response: {text[:200]}...")


def parse_call1_response(raw: str) -> list[dict]:
    """Parse Call 1 output: {sub_problems: [{id, raw_text, failure_summary}]}."""
    json_str = _extract_json(raw)
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ParseError(f"Invalid JSON: {e}") from e

    if isinstance(data, dict):
        items = data.get("sub_problems", [])
    elif isinstance(data, list):
        items = data
    else:
        raise ParseError(f"Expected dict or list, got {type(data).__name__}")

    for item in items:
        for field in ("id", "raw_text", "failure_summary"):
            if field not in item:
                raise ParseError(f"Missing required field '{field}' in Call 1 output")
    return items


def parse_call2_response(raw: str) -> list[dict]:
    """Parse Call 2/2' output: [{id, target_capability, ..., confidence, route}]."""
    json_str = _extract_json(raw)
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ParseError(f"Invalid JSON: {e}") from e

    if isinstance(data, dict) and "results" in data:
        items = data["results"]
    elif isinstance(data, list):
        items = data
    else:
        raise ParseError(f"Expected list, got {type(data).__name__}")

    required = ("id", "target_capability", "trajectory_signal", "hyde_positive",
                "keywords", "structured_filters", "confidence", "route")
    for item in items:
        missing = [f for f in required if f not in item]
        if missing:
            raise ParseError(f"Missing fields in Call 2 output: {missing}")
    return items


def parse_call3_response(raw: str) -> list[dict]:
    """Parse Call 3 output: [{original_id, clarified: [{id, raw_text, failure_summary}]}]."""
    json_str = _extract_json(raw)
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ParseError(f"Invalid JSON: {e}") from e

    if not isinstance(data, list):
        raise ParseError(f"Expected list, got {type(data).__name__}")

    for item in data:
        if "original_id" not in item or "clarified" not in item:
            raise ParseError("Missing 'original_id' or 'clarified' in Call 3 output")
        if len(item["clarified"]) > _MAX_CLARIFIED_PER_PROBLEM:
            item["clarified"] = item["clarified"][:_MAX_CLARIFIED_PER_PROBLEM]
        for c in item["clarified"]:
            for field in ("id", "raw_text", "failure_summary"):
                if field not in c:
                    raise ParseError(f"Missing '{field}' in clarified sub-problem")
    return data
