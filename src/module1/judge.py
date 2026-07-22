"""Phase 4: LLM judge for capability demonstration matching.

Sends compressed slice summaries to LLM via gateway.
Batch strategy: 3 slices per request (≤10k token budget).
Output: JudgeResult per slice (match/confidence/spans/reasoning).
"""
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 dengxuanliang

from __future__ import annotations

import json
import re

from module1.models import JudgeResult, Slice
from module1.summarizer import summarize_slice

__all__ = ["Judge", "_build_judge_prompt", "_parse_judge_response"]

_DEFAULT_BATCH_SIZE = 3

# 旧 system prompt：rubric=None（无判据卡）路径逐字沿用，保证向后兼容行为与改动前一致。
_JUDGE_SYSTEM_PROMPT = """你是一个 SFT 数据质量评审员。你的任务是判断给定的轨迹切片是否正向演示了目标能力。

判断标准：
1. 切片中是否有步骤正确执行了目标能力（而非复现了失败）
2. 如果有，标注具体哪几步是正例演示（start_step 到 end_step）
3. 给出置信度（0-1）和简短推理

对每个切片，输出 JSON 格式：
{"match": bool, "confidence": float, "spans": [{"start_step": int, "end_step": int}], "reasoning": string}

多个切片时输出 JSON 数组。"""

# 新 system prompt：rubric（判据卡）在场时启用，三步走判定。仅当 _build_judge_prompt
# 注入了 rubric 对照段时使用；无 rubric 时走上面的旧 prompt（判据卡方案 PR-3）。
_JUDGE_SYSTEM_PROMPT_RUBRIC = """你是一个 SFT 数据质量评审员。给你一张能力"判据卡（rubric）"和若干轨迹切片，\
判断每个切片是否**真正正向演示**了目标能力，并把证据精确定位到决策步。严格按以下三步走：

① 对照：逐条核对切片是否满足 rubric 的 positive_criteria（正向判据），是否踩中 negative_criteria（反例/失败模式）。\
擦边、"只是含有相关上下文"、平淡过渡都不算满足——必须有明确证据支撑某条 positive_criteria。

② 定位：指出决定性证据在第几步（evidence_step）。这一步是"证明该能力被演示"的关键决策/推理步。\
**如果指不出明确的决定性步，则 match=false**（宁缺毋滥，杀掉"看到超时就说 Found it"这类伪相关擦边片）。

③ 收窄 mask：spans 只圈**决定性步的 assistant 决策/推理那 1-2 步**。工具输出 dump、环境返回、观测结果\
是 context 不是 target，一律排除在 spans 之外。

按 capability_kind 调整判读视角（仅影响你找哪类证据，不改变判定标准）：
- presence（有痕）：找正向文本/结构标记本身，如"先写测试"。证据 = 该正向行为出现的决策步。
- avoidance（无痕）：正例 = 坏模式的"缺席"。找"本可犯错的锚点 + 其后没犯错的枢轴步"，mask 圈枢轴步。
- recovery（转折）：找 error → 正确处置的转折。mask 圈"处置"那一步。

对每个切片，输出 JSON 格式（criteria_hit 填命中的 positive_criteria 原文，evidence_step 填决定性步号，\
指不出则 evidence_step=null 且 match=false）：
{"match": bool, "confidence": float, "evidence_step": int|null, "criteria_hit": [string], \
"spans": [{"start_step": int, "end_step": int}], "reasoning": string}

多个切片时输出 JSON 数组（顺序与切片一致）。"""


class Judge:
    """LLM judge for capability demonstration assessment."""

    def __init__(self, gateway, model: str, batch_size: int = _DEFAULT_BATCH_SIZE):
        self._gateway = gateway
        self._model = model
        self._batch_size = batch_size

    async def judge_batch(
        self,
        *,
        slices: list[Slice],
        target_capability: list[str],
        trajectory_signal: str,
        rubric: dict | None = None,
    ) -> list[JudgeResult]:
        """Judge multiple slices, batching into LLM calls.

        Returns one JudgeResult per input slice, in the same order.

        rubric is the serialized CapabilityRubric dict (positive_criteria /
        negative_criteria / decisive_evidence / capability_kind) or None. When
        None, the judge falls back to the legacy prompt for backward
        compatibility (behavior identical to before the rubric card feature).
        """
        # rubric 在场 → 三步走 system prompt；None → 旧 prompt（逐字向后兼容）。
        system_prompt = (
            _JUDGE_SYSTEM_PROMPT_RUBRIC if rubric is not None else _JUDGE_SYSTEM_PROMPT
        )
        all_results: list[JudgeResult] = []

        for i in range(0, len(slices), self._batch_size):
            batch = slices[i:i + self._batch_size]
            prompt = _build_judge_prompt(
                slices=batch,
                target_capability=target_capability,
                trajectory_signal=trajectory_signal,
                rubric=rubric,
            )
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]

            response, _usage = await self._gateway.call(messages, self._model)

            if response is None:
                # Gateway failure: return no-match for entire batch
                all_results.extend([
                    JudgeResult(match=False, confidence=0.0, spans=[], reasoning="gateway_error")
                    for _ in batch
                ])
            else:
                results = _parse_judge_response(response, n_expected=len(batch))
                all_results.extend(results)

        return all_results


def _build_judge_prompt(
    *,
    slices: list[Slice],
    target_capability: list[str],
    trajectory_signal: str,
    rubric: dict | None = None,
) -> str:
    """Build the user prompt for judge LLM call.

    rubric 到这里是 **dict**（经 _spec_to_dict 序列化，非 CapabilityRubric 对象），
    按 dict 形状消费（rubric["positive_criteria"]，不是 .positive_criteria）。None
    时不注入对照段，退化为旧格式（向后兼容）。
    """
    parts = []
    parts.append(f"目标能力: {', '.join(target_capability)}")
    parts.append(f"轨迹信号: {trajectory_signal}")

    if rubric is not None:
        parts.append("")
        parts.append("=== 能力判据卡（rubric）===")
        kind = rubric.get("capability_kind", "")
        if kind:
            parts.append(f"capability_kind: {kind}")
        positive = rubric.get("positive_criteria") or []
        if positive:
            parts.append("positive_criteria（正向判据，需明确满足）:")
            for c in positive:
                parts.append(f"  - {c}")
        negative = rubric.get("negative_criteria") or []
        if negative:
            parts.append("negative_criteria（反例/失败模式，踩中则不算）:")
            for c in negative:
                parts.append(f"  - {c}")
        decisive = rubric.get("decisive_evidence", "")
        if decisive:
            parts.append(f"decisive_evidence（决定性证据形态）: {decisive}")

    parts.append("")

    for i, s in enumerate(slices):
        summary = summarize_slice(s)
        parts.append(f"--- 切片 {i+1} (trajectory={s.trajectory_id}, steps {s.start_step}-{s.end_step}) ---")
        parts.append(summary)
        parts.append("")

    parts.append(f"请对以上 {len(slices)} 个切片分别判断，输出 JSON 数组（{len(slices)} 个元素）。")
    return "\n".join(parts)


def _parse_judge_response(raw: str | None, n_expected: int) -> list[JudgeResult]:
    """Parse LLM judge response into JudgeResult list.

    Handles malformed responses gracefully by returning no-match defaults.
    """
    if not raw:
        return [JudgeResult(match=False, confidence=0.0, spans=[], reasoning="empty_response")
                for _ in range(n_expected)]

    # Try to extract JSON from response (may have markdown fences)
    json_str = _extract_json(raw)

    try:
        data = json.loads(json_str)
    except (json.JSONDecodeError, TypeError):
        return [JudgeResult(match=False, confidence=0.0, spans=[], reasoning="parse_error")
                for _ in range(n_expected)]

    # Normalize to list
    if isinstance(data, dict):
        data = [data]

    results = []
    for i in range(n_expected):
        if i < len(data) and isinstance(data[i], dict):
            item = data[i]
            try:
                match_val = item.get("match", False)
                if isinstance(match_val, str):
                    match_val = match_val.lower() in ("true", "1", "yes")
                results.append(JudgeResult(
                    match=bool(match_val),
                    confidence=float(item.get("confidence", 0.0)),
                    spans=_clean_spans(item.get("spans")),
                    reasoning=str(item.get("reasoning", "")),
                    evidence_step=_clean_evidence_step(item.get("evidence_step")),
                    criteria_hit=_clean_criteria_hit(item.get("criteria_hit")),
                ))
            except (ValueError, TypeError):
                results.append(JudgeResult(
                    match=False, confidence=0.0, spans=[], reasoning="field_coercion_error"
                ))
        else:
            results.append(JudgeResult(
                match=False, confidence=0.0, spans=[], reasoning="missing_in_response"
            ))
    return results


def _clean_spans(raw) -> list[dict]:
    """Keep only well-formed spans: int start/end with start <= end."""
    if not isinstance(raw, list):
        return []
    cleaned = []
    for s in raw:
        if not isinstance(s, dict):
            continue
        start, end = s.get("start_step"), s.get("end_step")
        if isinstance(start, bool) or isinstance(end, bool):
            continue  # bool is a subclass of int; reject explicitly
        if not (isinstance(start, int) and isinstance(end, int)):
            continue
        if start > end:
            continue
        cleaned.append({"start_step": start, "end_step": end})
    return cleaned


def _clean_evidence_step(raw) -> int | None:
    """Coerce evidence_step to int; missing/null/malformed → None (缺失降级)."""
    if raw is None:
        return None
    if isinstance(raw, bool):  # bool is int subclass; reject explicitly
        return None
    if isinstance(raw, int):
        return raw
    return None


def _clean_criteria_hit(raw) -> list[str]:
    """Keep only string entries; missing/malformed → [] (缺失降级)."""
    if not isinstance(raw, list):
        return []
    return [c for c in raw if isinstance(c, str)]


def _extract_json(text: str) -> str:
    """Extract JSON from text that may contain markdown fences."""
    # Try stripping markdown code fences
    fence_match = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
    if fence_match:
        return fence_match.group(1).strip()
    # Try finding array or object directly
    for start_char, end_char in [("[", "]"), ("{", "}")]:
        start = text.find(start_char)
        if start != -1:
            end = text.rfind(end_char)
            if end > start:
                return text[start:end + 1]
    return text
