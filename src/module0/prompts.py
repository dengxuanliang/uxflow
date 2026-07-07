"""LLM prompt templates for Call 1/2/3/2'.

Each function returns a list[dict] (OpenAI messages format) ready to pass
to gateway.call(). Prompt design follows spec §2 Part 1-6.5 exactly.

Key design decisions:
- Call 2 system prompt injects taxonomy via Taxonomy.to_prompt_text()
- Empty taxonomy triggers "free proposal" mode (spec Part 2 table)
- Call 2' reuses Call 2 prompt but with explicit "不输出 drop_reason" constraint
- All prompts request JSON output with specified schema
"""

from __future__ import annotations

from module0.taxonomy import Taxonomy

__all__ = [
    "build_call1_messages",
    "build_call2_messages",
    "build_call3_messages",
    "build_call2_prime_messages",
]

# ── Call 1: 子问题分解 (Part 1) ──

_CALL1_SYSTEM = """你是一个问题分解专家。任务：把用户的问题描述拆分成互不重叠的原子子问题。

## 分解规则
- 按逗号/分号/语义转折切分
- 合并重复描述
- 每条必须是单一失败模式
- 整句只描述一个问题则输出单条
- 无法确定是否该拆时保守不拆，标 needs_review: true

## 输出格式（严格 JSON）
```json
{
  "sub_problems": [
    {
      "id": "p1",
      "raw_text": "从原始描述切出的片段",
      "failure_summary": "一句话标准化失败描述"
    }
  ]
}
```

只输出 JSON，不要其他解释。"""


def build_call1_messages(raw_input: str) -> list[dict]:
    return [
        {"role": "system", "content": _CALL1_SYSTEM},
        {"role": "user", "content": raw_input},
    ]


# ── Call 2: 批量打标 + 自评 (Part 2-6) ──

_CALL2_SYSTEM_WITH_TAXONOMY = """你是一个 SWE 问题分析专家。对每个子问题完成以下全部字段：

## 你可用的能力标签词表
{taxonomy_injection}

优先从上述词表选取标签（1-3个）。确无匹配才提议新标签（小写英文+下划线，动宾结构，描述"正确做法"，附 description + parent 建议，标 taxonomy_extension: true）。

## 结构化过滤条件 structured_filters
字段（全部可选，无则设 null）：
- languages: {languages_enum}
- tools_used: {tools_enum}
- outcome_transition: {outcome_enum}
- min_turns: 整数 ≥1
- has_verification_step: 布尔值

## 输出格式（严格 JSON 数组）
对每个子问题输出：
```json
{{
  "id": "p1",
  "target_capability": ["label1"],
  "trajectory_signal": "在轨迹中应匹配什么模式",
  "hyde_positive": ["假设正例片段1(200-500token)", "假设正例片段2"],
  "keywords": ["关键词1", "关键词2"],
  "structured_filters": {{"languages": ["python"], "outcome_transition": ["failed→success"], ...}},
  "confidence": 0.0-1.0,
  "route": "pass" 或 "drop",
  "drop_reason": "ambiguous|not_applicable|label_diverged|other（仅 route==drop 时填）"
}}
```

## 置信度评估维度
- 子问题是否有歧义（高权重）
- 能力标签是否唯一命中（中）
- trajectory_signal 是否可操作（中）
- HyDE 变体是否一致（低）

## 分流规则
- confidence ≥ 0.8 → route: "pass"
- confidence < 0.8 → route: "drop"，并标注 drop_reason

只输出 JSON 数组，不要其他解释。"""

_CALL2_SYSTEM_EMPTY_TAXONOMY = """你是一个 SWE 问题分析专家。对每个子问题完成以下全部字段：

## 能力标签
当前词表为空（冷启动）。请自由提议标签（1-3个）：
- 命名规则：小写英文+下划线，动宾结构，描述"正确做法"
- 每个标签附：description（一句话中文）、parent 建议（无则 null）
- 标 taxonomy_extension: true

## 结构化过滤条件 structured_filters
字段（全部可选，无则设 null）：
- languages: {languages_enum}
- tools_used: {tools_enum}
- outcome_transition: {outcome_enum}
- min_turns: 整数 ≥1
- has_verification_step: 布尔值

## 输出格式（严格 JSON 数组）
对每个子问题输出：
```json
{{
  "id": "p1",
  "target_capability": ["label1"],
  "trajectory_signal": "在轨迹中应匹配什么模式",
  "hyde_positive": ["假设正例片段1(200-500token)", "假设正例片段2"],
  "keywords": ["关键词1", "关键词2"],
  "structured_filters": {{"languages": ["python"], "outcome_transition": ["failed→success"], ...}},
  "confidence": 0.0-1.0,
  "route": "pass" 或 "drop",
  "drop_reason": "ambiguous|not_applicable|label_diverged|other（仅 route==drop 时填）"
}}
```

## 置信度评估维度
- 子问题是否有歧义（高权重）
- 能力标签是否唯一命中（中）
- trajectory_signal 是否可操作（中）
- HyDE 变体是否一致（低）

## 分流规则
- confidence ≥ 0.8 → route: "pass"
- confidence < 0.8 → route: "drop"，并标注 drop_reason

只输出 JSON 数组，不要其他解释。"""


def build_call2_messages(sub_problems: list[dict], taxonomy: Taxonomy) -> list[dict]:
    from module0.schema import LANGUAGES, TOOLS_USED, OUTCOME_TRANSITIONS

    format_kwargs = {
        "languages_enum": sorted(LANGUAGES),
        "tools_enum": sorted(TOOLS_USED),
        "outcome_enum": sorted(OUTCOME_TRANSITIONS),
    }

    if taxonomy.is_empty:
        system = _CALL2_SYSTEM_EMPTY_TAXONOMY.format(**format_kwargs)
    else:
        system = _CALL2_SYSTEM_WITH_TAXONOMY.format(
            taxonomy_injection=taxonomy.to_prompt_text(),
            **format_kwargs,
        )

    user_content = "请对以下子问题逐一分析：\n\n"
    for sp in sub_problems:
        user_content += f"- id: {sp['id']}\n  raw_text: {sp['raw_text']}\n  failure_summary: {sp['failure_summary']}\n\n"

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content},
    ]


# ── Call 3: 歧义澄清 (Part 6.5) ──

_CALL3_SYSTEM = """你是一个语义消歧专家。对每个被标记为"ambiguous"的子问题：

## 任务
列出该子问题所有合理的互斥解释，每条解释产出一条独立的被消歧新子问题。

## 约束
- 每条原始子问题最多拆 4 条消歧解释
- 新 id 格式：原 id + 字母后缀（如 p12 → p12a, p12b, p12c）
- 每条消歧子问题必须包含 failure_summary（重新描述，清晰无歧义）

## 输出格式（严格 JSON 数组）
```json
[
  {
    "original_id": "p12",
    "clarified": [
      {"id": "p12a", "raw_text": "原始文本", "failure_summary": "消歧后的明确描述"},
      {"id": "p12b", "raw_text": "原始文本", "failure_summary": "另一种消歧描述"}
    ]
  }
]
```

只输出 JSON，不要其他解释。"""


def build_call3_messages(ambiguous_problems: list[dict]) -> list[dict]:
    user_content = "以下子问题因语义歧义被标记为 drop。请消歧：\n\n"
    for sp in ambiguous_problems:
        user_content += f"- id: {sp['id']}\n  raw_text: {sp['raw_text']}\n  failure_summary: {sp['failure_summary']}\n"
        if sp.get("target_capability"):
            user_content += f"  已有标签猜测: {sp['target_capability']}\n"
        user_content += "\n"

    return [
        {"role": "system", "content": _CALL3_SYSTEM},
        {"role": "user", "content": user_content},
    ]


# ── Call 2': 消歧子问题重评 ──

_CALL2_PRIME_EXTRA = """

## 额外约束（Call 2' 专用）
- 这些是经过消歧的子问题，不再输出 drop_reason
- 不再触发任何进一步澄清
- confidence < 0.8 的直接丢弃，不再进入后续流程
- 输出同 Call 2 格式，但 route 只可能是 "pass"（confidence ≥ 0.8 时）或省略"""


def build_call2_prime_messages(clarified_sub_problems: list[dict], taxonomy: Taxonomy) -> list[dict]:
    """Same as Call 2 but with additional constraints for no further disambiguation."""
    base_msgs = build_call2_messages(clarified_sub_problems, taxonomy)
    base_msgs[0]["content"] += _CALL2_PRIME_EXTRA
    return base_msgs
