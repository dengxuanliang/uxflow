"""Phase 4: LLM judge for capability demonstration matching.

Sends compressed slice summaries to LLM via gateway.
Batch strategy: 3 slices per request (≤10k token budget).
Output: JudgeResult per slice (match/confidence/spans/reasoning).
"""

from __future__ import annotations

import json
import re

from module1.models import JudgeResult, Slice
from module1.summarizer import summarize_slice

__all__ = ["Judge", "_build_judge_prompt", "_parse_judge_response"]

_DEFAULT_BATCH_SIZE = 3

_JUDGE_SYSTEM_PROMPT = """你是一个 SFT 数据质量评审员。你的任务是判断给定的轨迹切片是否正向演示了目标能力。

判断标准：
1. 切片中是否有步骤正确执行了目标能力（而非复现了失败）
2. 如果有，标注具体哪几步是正例演示（start_step 到 end_step）
3. 给出置信度（0-1）和简短推理

对每个切片，输出 JSON 格式：
{"match": bool, "confidence": float, "spans": [{"start_step": int, "end_step": int}], "reasoning": string}

多个切片时输出 JSON 数组。"""


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
    ) -> list[JudgeResult]:
        """Judge multiple slices, batching into LLM calls.

        Returns one JudgeResult per input slice, in the same order.
        """
        all_results: list[JudgeResult] = []

        for i in range(0, len(slices), self._batch_size):
            batch = slices[i:i + self._batch_size]
            prompt = _build_judge_prompt(
                slices=batch,
                target_capability=target_capability,
                trajectory_signal=trajectory_signal,
            )
            messages = [
                {"role": "system", "content": _JUDGE_SYSTEM_PROMPT},
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
) -> str:
    """Build the user prompt for judge LLM call."""
    parts = []
    parts.append(f"目标能力: {', '.join(target_capability)}")
    parts.append(f"轨迹信号: {trajectory_signal}")
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
                    spans=item.get("spans") if isinstance(item.get("spans"), list) else [],
                    reasoning=str(item.get("reasoning", "")),
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
