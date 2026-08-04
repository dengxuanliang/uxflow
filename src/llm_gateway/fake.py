"""FakeGateway — scripted LLM replay for zero-config demos and CI smoke runs.

Duck-types LLMGateway.call() so entry-point scripts can run the full pipeline
with no LiteLLM proxy. Responses are FIXED REPLAYS keyed off the prompt, not
model output: they prove the pipeline wires together, never that its picks are
good. Selecting this backend is always announced (see uxflow_runtime.banner).

Deliberately kept out of LLMGateway itself: that class owns real transport,
retries and the adaptive circuit breaker, and folding a fake path into it would
blur what those mechanisms are observing.
"""
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 dengxuanliang

from __future__ import annotations

import json

__all__ = ["FakeGateway"]

# Replay data is inlined rather than read from fixtures/: that directory ships
# with the repo but not inside the built wheel, so a non-editable install would
# lose it. Mirrors fixtures/problem_specs/test_01.json.
_SUB_PROBLEMS = [
    {
        "id": "p1",
        "raw_text": "写入py文件有语法错误",
        "failure_summary": "写入 py 文件时产生语法错误",
        "target_capability": ["valid_syntax_in_toolcall"],
        "trajectory_signal": "observation 含 SyntaxError 且前序 tool_call 含 python 代码写入",
        "hyde_positive": [
            "假设: 工具正确写入 python 文件，无语法错误，执行结果 exit code 0",
            "假设: python 文件包含合法 import 和函数定义，lint 通过",
        ],
        "keywords": ["SyntaxError", "python", "import"],
        "structured_filters": {
            "languages": ["python"],
            "tools_used": ["Write", "Edit"],
            "has_verification_step": None,
        },
        "confidence": 0.92,
        "route": "pass",
    },
    {
        "id": "p2",
        "raw_text": "工具调用结构经常出错",
        "failure_summary": "tool_call JSON 结构畸形或错位到 thinking 块",
        "target_capability": ["wellformed_tool_call"],
        "trajectory_signal": "tool_call 块 JSON 解析失败或 tool_call 内容出现在 thinking/content 字段",
        "hyde_positive": [
            "假设: tool_call 正确使用 JSON 格式，包含 name + arguments 字段",
            "假设: thinking 块不包含 tool_call 内容，工具调用独立且结构清晰",
        ],
        "keywords": ["tool_call", "JSON", "malformed", "thinking"],
        "structured_filters": {
            "languages": None,
            "tools_used": None,
            "has_verification_step": None,
        },
        "confidence": 0.88,
        "route": "pass",
    },
]

# Marker phrases from the real system prompts (module0.prompts, module1.judge).
# Matching on these keeps the fake honest: if a prompt is reworded, replay stops
# matching and the run fails loudly rather than returning a silently wrong shape.
_CALL1_MARKER = "问题分解专家"
_CALL2_MARKER = "SWE 问题分析专家"
_CALL3_MARKER = "语义消歧专家"
_JUDGE_MARKER = "SFT 数据质量评审员"

_USAGE = {"status_code": 200, "prompt_tokens": 0, "completion_tokens": 0}


def _call1_payload() -> str:
    """Sub-problem decomposition: id + raw_text + failure_summary only."""
    return json.dumps(
        {
            "sub_problems": [
                {
                    "id": sp["id"],
                    "raw_text": sp["raw_text"],
                    "failure_summary": sp["failure_summary"],
                }
                for sp in _SUB_PROBLEMS
            ]
        },
        ensure_ascii=False,
    )


def _call2_payload() -> str:
    """Enriched sub-problems: the full contract as module0 Call 2 returns it."""
    return json.dumps(_SUB_PROBLEMS, ensure_ascii=False)


def _judge_payload(prompt: str) -> str:
    """One verdict per slice in the batch, all positive.

    Slice count is read back off the prompt so the array length always matches
    what module1 expects; a mismatch there is silently dropped downstream.
    """
    n = max(prompt.count("--- 切片 "), 1)
    return json.dumps(
        [
            {
                "match": True,
                "confidence": 0.9,
                "spans": [{"start_step": 0, "end_step": 1}],
                "reasoning": "fake backend: scripted positive verdict",
            }
            for _ in range(n)
        ],
        ensure_ascii=False,
    )


class FakeGateway:
    """Scripted stand-in for LLMGateway. Same call signature, no network."""

    def __init__(self):
        self.calls: list[tuple[list[dict], str]] = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def call(
        self,
        messages: list[dict],
        model: str,
        *,
        temperature: float = 0.1,
        max_tokens: int = 1000,
    ) -> tuple[str | None, dict]:
        self.calls.append((messages, model))
        system = messages[0].get("content", "") if messages else ""
        user = messages[-1].get("content", "") if messages else ""

        if _JUDGE_MARKER in system:
            return _judge_payload(user), dict(_USAGE)
        if _CALL1_MARKER in system:
            return _call1_payload(), dict(_USAGE)
        if _CALL2_MARKER in system:
            return _call2_payload(), dict(_USAGE)
        if _CALL3_MARKER in system:
            # Disambiguation is only reached when Call 2 flags something
            # ambiguous; the scripted Call 2 never does.
            return "[]", dict(_USAGE)

        raise RuntimeError(
            "FakeGateway has no scripted reply for this prompt — the real "
            "system prompts likely changed. Update src/llm_gateway/fake.py "
            "or run against a real proxy (UXFLOW_LLM_BACKEND=real)."
        )
