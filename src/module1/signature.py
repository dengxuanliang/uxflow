"""Phase 1: Extract structural signature from a slice.

Pure rules + regex. No LLM cost. Embedding is optional (pass
embedding_model=None to skip for unit tests).
"""
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 dengxuanliang

from __future__ import annotations

import re

from module1.models import Slice, Step, TrajectorySignature

__all__ = ["extract_signature", "build_embedding_text"]

_ERROR_PATTERNS = [
    re.compile(r"Traceback \(most recent call last\)", re.IGNORECASE),
    re.compile(r"SyntaxError:", re.IGNORECASE),
    re.compile(r"Error:", re.IGNORECASE),
    re.compile(r"Exception:", re.IGNORECASE),
    re.compile(r"FAILED", re.IGNORECASE),
    re.compile(r"panic:", re.IGNORECASE),
]

_SUCCESS_PATTERNS = [
    re.compile(r"file created", re.IGNORECASE),
    re.compile(r"file edited", re.IGNORECASE),
    re.compile(r"successfully", re.IGNORECASE),
    re.compile(r"\d+ passed", re.IGNORECASE),
    re.compile(r"PASSED"),
    re.compile(r"OK$", re.MULTILINE),
]

_VERIFY_KEYWORDS = re.compile(r"test|check|pytest|run|verify|build", re.IGNORECASE)

_LANGUAGE_PATTERNS = {
    "python": re.compile(r"\.py\b|python|import\s|def\s|class\s", re.IGNORECASE),
    "javascript": re.compile(r"\.js\b|\.ts\b|node|require\(|import.*from", re.IGNORECASE),
    "bash": re.compile(r"\.sh\b|bash|#!/bin", re.IGNORECASE),
    "java": re.compile(r"\.java\b|public\s+class", re.IGNORECASE),
    "cpp": re.compile(r"\.(cpp|cc|h)\b|#include", re.IGNORECASE),
    "go": re.compile(r"\.go\b|package\s+main|func\s+main", re.IGNORECASE),
    "html": re.compile(r"\.html\b|<html|<div", re.IGNORECASE),
}

_BM25_ERROR_TOKENS = re.compile(
    r"(SyntaxError|TypeError|ValueError|KeyError|ImportError|"
    r"AttributeError|RuntimeError|FileNotFoundError|"
    r"Traceback|Exception|FAILED|Error|panic)"
)
_PATH_TOKEN = re.compile(r"\b[\w./-]+\.(?:py|js|ts|json|yaml|yml|md)\b")
_PY_DEF_TOKEN = re.compile(r"\bdef\s+([A-Za-z_][A-Za-z0-9_]*)")
_IDENT_TOKEN = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]{2,}\b")
_STOP_IDENTIFIERS = {
    "def", "return", "class", "import", "from", "with", "open", "self",
    "true", "false", "none", "syntaxerror", "traceback", "error",
}


def extract_signature(
    slice_obj: Slice,
    *,
    embedding_model=None,
) -> TrajectorySignature:
    """Extract structured signature from a slice."""
    steps = slice_obj.steps

    # Tools used
    tools_used = list({s.tool_call_name for s in steps if s.tool_call_name})

    # Languages (from tool_call args content)
    languages = _detect_languages(steps)

    # Error/success patterns (from tool_result)
    tool_results = [s.tool_result or s.content for s in steps if s.role == "tool"]
    has_error = any(p.search(r) for r in tool_results for p in _ERROR_PATTERNS)
    has_success = any(p.search(r) for r in tool_results for p in _SUCCESS_PATTERNS)

    # Verification step
    has_verify = any(
        s.tool_call_name == "Bash" and s.tool_call_args and _VERIFY_KEYWORDS.search(s.tool_call_args)
        for s in steps
    )

    # Turn count (user messages)
    turn_count = sum(1 for s in steps if s.role == "user")

    # BM25 tokens
    bm25 = _extract_bm25_tokens(steps, tools_used)

    # Embedding (optional)
    embedding = []
    if embedding_model is not None:
        summary_text = _build_summary_for_embedding(steps)
        embedding = embedding_model.embed(summary_text)

    return TrajectorySignature(
        trajectory_id=slice_obj.trajectory_id,
        slice_index=slice_obj.slice_index,
        step_range=(slice_obj.start_step, slice_obj.end_step),
        step_count=slice_obj.step_count,
        turn_count=turn_count,
        languages=languages,
        tools_used=tools_used,
        has_error_pattern=has_error,
        has_success_pattern=has_success,
        has_verification_step=has_verify,
        bm25_tokens=bm25,
        embedding=embedding,
    )


def _detect_languages(steps: list[Step]) -> list[str]:
    all_text = " ".join(
        (s.tool_call_args or "") + " " + (s.content or "")
        for s in steps if s.role == "assistant"
    )
    detected = []
    for lang, pattern in _LANGUAGE_PATTERNS.items():
        if pattern.search(all_text):
            detected.append(lang)
    return detected if detected else ["other"]


def _extract_bm25_tokens(steps: list[Step], tools_used: list[str]) -> list[str]:
    tokens = set()
    # Add tool names (lowercased)
    for t in tools_used:
        tokens.add(t.lower())
    # Extract error keywords from tool results
    for s in steps:
        text = " ".join(
            part for part in (s.tool_call_args, s.tool_result, s.content) if part
        )
        for match in _BM25_ERROR_TOKENS.finditer(text):
            tokens.add(match.group(0))
        for match in _PATH_TOKEN.finditer(text):
            tokens.add(match.group(0))
        for match in _PY_DEF_TOKEN.finditer(text):
            tokens.add(match.group(1))
        for match in _IDENT_TOKEN.finditer(text):
            ident = match.group(0)
            if ident.lower() not in _STOP_IDENTIFIERS:
                tokens.add(ident)
    return sorted(tokens)


def build_embedding_text(slice_obj: Slice) -> str:
    """extract_signature 会为该切片嵌入的**同一段**文本。

    批量调用方（ingest 写路径）用它先把整批文本备齐，一次 embed_batch 后回填
    sig.embedding —— 模型调用数从每切片一次降到每批一次，向量逐字节相同。
    与 extract_signature 共用 _build_summary_for_embedding，单一来源保证两条
    路径不会漂移。
    """
    return _build_summary_for_embedding(slice_obj.steps)


def _build_summary_for_embedding(steps: list[Step]) -> str:
    """Build a short text summary for embedding."""
    parts = []
    for s in steps:
        if s.tool_call_args:
            parts.append(s.tool_call_args[:160])
        if s.role == "assistant" and s.content:
            parts.append(s.content[:100])
            if s.tool_call_name:
                parts.append(s.tool_call_name)
        elif s.role == "tool" and s.tool_result:
            parts.append(s.tool_result[:50])
    return " ".join(parts)[:1000]
