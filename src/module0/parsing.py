"""Parse LLM raw text responses into structured dicts.

Handles: markdown fence stripping, JSON extraction, field validation,
and graceful error reporting. Does NOT do schema-level validation
(SubProblem/StructuredFilters) — that's compiler.py's job.
"""
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 dengxuanliang

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


def _scan_balanced(text: str, start: int) -> int | None:
    """Return the index just past the balanced bracket group starting at `start`.

    String-literal aware: braces/brackets inside JSON string values (including
    escaped quotes) are NOT counted. Returns None if never balanced.
    """
    open_c = text[start]
    close_c = '}' if open_c == '{' else ']'
    depth = 0
    in_string = False
    escaped = False
    for j in range(start, len(text)):
        ch = text[j]
        if in_string:
            if escaped:
                escaped = False
            elif ch == '\\':
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == open_c:
            depth += 1
        elif ch == close_c:
            depth -= 1
            if depth == 0:
                return j + 1
    return None


def _extract_json(text: str) -> str:
    """Strip markdown fences and surrounding text to find JSON.

    String-literal aware: braces inside string values (e.g. code snippets in
    trajectory_signal / hyde_positive) do not break extraction.
    """
    if not text:
        raise ParseError("Empty response")
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    for i, c in enumerate(text):
        if c in ('{', '['):
            end = _scan_balanced(text, i)
            if end is not None:
                return text[i:end]
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
