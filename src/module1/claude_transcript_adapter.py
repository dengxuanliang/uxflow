"""Convert Claude Code transcript events into UXFlow trajectory JSONL records."""
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 dengxuanliang

from __future__ import annotations

import json
import pathlib
from collections.abc import Iterable
from typing import Any

__all__ = [
    "convert_transcript_file",
    "convert_transcript_records",
    "iter_transcript_events",
]


def iter_transcript_events(path: str | pathlib.Path) -> Iterable[dict[str, Any]]:
    """Yield object JSONL events from a Claude transcript, skipping malformed lines."""
    with pathlib.Path(path).open(errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(event, dict):
                yield event


def convert_transcript_file(path: str | pathlib.Path) -> dict[str, Any] | None:
    path = pathlib.Path(path)
    return convert_transcript_records(path.stem, iter_transcript_events(path))


def convert_transcript_records(
    trajectory_id: str,
    records: Iterable[dict[str, Any]],
) -> dict[str, Any] | None:
    messages: list[dict[str, Any]] = []
    has_user = False
    has_assistant = False
    has_tool_use = False
    has_tool_result = False

    for event in records:
        event_type = event.get("type")
        message = event.get("message")

        if event_type == "system":
            content = _stringify_content(event.get("content"))
            if content:
                messages.append({"role": "system", "content": content})
            continue

        if not isinstance(message, dict):
            continue

        role = message.get("role")
        if role == "user":
            converted, saw_tool_result = _convert_user_message(message)
            if converted:
                messages.extend(converted)
                has_user = True
            has_tool_result = has_tool_result or saw_tool_result
        elif role == "assistant":
            converted, saw_tool_use = _convert_assistant_message(message)
            if converted:
                messages.append(converted)
                has_assistant = True
            has_tool_use = has_tool_use or saw_tool_use

    if not (has_user and has_assistant and has_tool_use and has_tool_result):
        return None
    return {"id": trajectory_id, "messages": messages}


def _convert_user_message(message: dict[str, Any]) -> tuple[list[dict[str, Any]], bool]:
    content = message.get("content")
    if isinstance(content, str):
        text = content.strip()
        return ([{"role": "user", "content": text}] if text else []), False

    converted: list[dict[str, Any]] = []
    saw_tool_result = False
    if isinstance(content, list):
        text_parts: list[str] = []
        for block in content:
            if not isinstance(block, dict):
                continue
            block_type = block.get("type")
            if block_type == "text":
                text = _stringify_content(block.get("text"))
                if text:
                    text_parts.append(text)
            elif block_type == "tool_result":
                saw_tool_result = True
                converted.append(
                    {
                        "role": "tool",
                        "tool_call_id": str(block.get("tool_use_id") or ""),
                        "content": _stringify_content(block.get("content")),
                    }
                )
        if text_parts:
            converted.insert(0, {"role": "user", "content": "\n\n".join(text_parts)})
    return converted, saw_tool_result


def _convert_assistant_message(message: dict[str, Any]) -> tuple[dict[str, Any] | None, bool]:
    content = message.get("content")
    if isinstance(content, str):
        text = content.strip()
        return ({"role": "assistant", "content": text} if text else None), False

    if not isinstance(content, list):
        return None, False

    text_parts: list[str] = []
    reasoning_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        block_type = block.get("type")
        if block_type == "text":
            text = _stringify_content(block.get("text"))
            if text:
                text_parts.append(text)
        elif block_type == "thinking":
            thinking = _stringify_content(block.get("thinking"))
            if thinking:
                reasoning_parts.append(thinking)
        elif block_type == "tool_use":
            tool_calls.append(_convert_tool_use(block))

    if not text_parts and not reasoning_parts and not tool_calls:
        return None, False

    converted: dict[str, Any] = {"role": "assistant", "content": "\n\n".join(text_parts)}
    if reasoning_parts:
        converted["reasoning"] = "\n\n".join(reasoning_parts)
    if tool_calls:
        converted["tool_calls"] = tool_calls
    return converted, bool(tool_calls)


def _convert_tool_use(block: dict[str, Any]) -> dict[str, Any]:
    arguments = _compact_json(block.get("input", {}))
    return {
        "id": str(block.get("id") or ""),
        "type": "function",
        "function": {
            "name": str(block.get("name") or ""),
            "arguments": arguments,
        },
    }


def _compact_json(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    except TypeError:
        return json.dumps(str(value), ensure_ascii=False, separators=(",", ":"))


def _stringify_content(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        parts = [_stringify_content(item) for item in value]
        return "\n".join(part for part in parts if part)
    if isinstance(value, dict):
        for key in ("text", "content"):
            if key in value:
                return _stringify_content(value[key])
        return _compact_json(value)
    return str(value).strip()
