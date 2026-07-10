"""Load and parse trajectories from JSONL."""
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 dengxuanliang

from __future__ import annotations
import hashlib
import json
import pathlib
from module1.models import Step, Trajectory

__all__ = ["load_trajectories", "parse_trajectory"]


def load_trajectories(path: str | pathlib.Path) -> list[Trajectory]:
    path = pathlib.Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return []
    trajectories = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            trajectories.append(parse_trajectory(json.loads(line)))
    return trajectories


def parse_trajectory(data: dict) -> Trajectory:
    traj_id = data.get(
        "id",
        f"traj_{hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()[:16]}",
    )
    messages = data.get("messages", [])
    steps = []
    step_idx = 0

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "") or ""

        if role == "assistant":
            tool_calls = msg.get("tool_calls", [])
            if tool_calls:
                for tc in tool_calls:
                    fn = tc.get("function", tc) if isinstance(tc, dict) else {}
                    steps.append(Step(
                        index=step_idx, role="assistant", content=content,
                        tool_call_name=fn.get("name"),
                        tool_call_args=fn.get("arguments", ""),
                    ))
                    step_idx += 1
            else:
                steps.append(Step(index=step_idx, role="assistant", content=content))
                step_idx += 1
        elif role == "tool":
            steps.append(Step(index=step_idx, role="tool", content=content, tool_result=content))
            step_idx += 1
        elif role in ("user", "system"):
            steps.append(Step(index=step_idx, role=role, content=content))
            step_idx += 1

    return Trajectory(id=traj_id, steps=steps, raw_messages=messages)
