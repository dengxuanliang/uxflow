from module0.prompts import build_call1_messages, build_call2_messages, build_call3_messages
from module0.taxonomy import Taxonomy


def test_call1_messages_structure():
    msgs = build_call1_messages("写入py文件有语法错误，并且工具调用结构经常出错")
    assert len(msgs) >= 2
    assert msgs[0]["role"] == "system"
    assert msgs[-1]["role"] == "user"
    assert "写入py文件有语法错误" in msgs[-1]["content"]


def test_call1_system_prompt_contains_rules():
    msgs = build_call1_messages("test input")
    system = msgs[0]["content"]
    assert "子问题" in system or "sub-problem" in system.lower()
    assert "JSON" in system


def test_call2_messages_with_taxonomy(taxonomy_v0_path):
    taxonomy = Taxonomy.load(taxonomy_v0_path)
    sub_problems = [
        {"id": "p1", "raw_text": "写入py文件有语法错误", "failure_summary": "写入 py 文件语法错误"},
    ]
    msgs = build_call2_messages(sub_problems, taxonomy)
    assert len(msgs) >= 2
    system = msgs[0]["content"]
    assert "valid_syntax_in_toolcall" in system
    assert "structured_filters" in system or "languages" in system


def test_call2_messages_empty_taxonomy():
    taxonomy = Taxonomy.from_dict({"version": "0.0.0", "updated_at": "", "labels": []})
    sub_problems = [
        {"id": "p1", "raw_text": "test", "failure_summary": "test"},
    ]
    msgs = build_call2_messages(sub_problems, taxonomy)
    system = msgs[0]["content"]
    assert "自由提议" in system or "propose" in system.lower() or "freely" in system.lower()


def test_call3_messages():
    ambiguous_problems = [
        {"id": "p12", "raw_text": "多步任务未完成即终止", "failure_summary": "多步任务中途停止",
         "target_capability": ["persist_through_truncation"]},
    ]
    msgs = build_call3_messages(ambiguous_problems)
    assert len(msgs) >= 2
    system = msgs[0]["content"]
    assert "消歧" in system or "disambiguat" in system.lower()
    user = msgs[-1]["content"]
    assert "p12" in user


def test_call2_output_schema_mentioned():
    taxonomy = Taxonomy.from_dict({"version": "0.0.0", "updated_at": "", "labels": []})
    msgs = build_call2_messages([{"id": "p1", "raw_text": "t", "failure_summary": "s"}], taxonomy)
    system = msgs[0]["content"]
    for field in ["target_capability", "trajectory_signal", "hyde_positive", "keywords", "confidence", "route"]:
        assert field in system, f"Missing field '{field}' in Call 2 system prompt"


def test_call2_prompt_has_label_proposals_slot(taxonomy_v0_path):
    taxonomy = Taxonomy.load(taxonomy_v0_path)
    msgs = build_call2_messages(
        [{"id": "p1", "raw_text": "x", "failure_summary": "y"}], taxonomy)
    system = msgs[0]["content"]
    assert "label_proposals" in system


def test_call2_empty_taxonomy_prompt_has_label_proposals_slot():
    empty = Taxonomy(version="0.1.0", updated_at="", labels=[])
    msgs = build_call2_messages(
        [{"id": "p1", "raw_text": "x", "failure_summary": "y"}], empty)
    system = msgs[0]["content"]
    assert "label_proposals" in system


# ── PR-2: rubric 产出段 + capability_kind 三型定义（两份 prompt 都含）──


def _both_call2_systems(taxonomy_v0_path):
    """有词表 / 空词表 两份 Call 2 system prompt。"""
    from module0.taxonomy import Taxonomy as _T
    with_tax = build_call2_messages(
        [{"id": "p1", "raw_text": "x", "failure_summary": "y"}],
        _T.load(taxonomy_v0_path))[0]["content"]
    empty_tax = build_call2_messages(
        [{"id": "p1", "raw_text": "x", "failure_summary": "y"}],
        _T(version="0.1.0", updated_at="", labels=[]))[0]["content"]
    return with_tax, empty_tax


def test_call2_both_prompts_have_rubric_section(taxonomy_v0_path):
    for system in _both_call2_systems(taxonomy_v0_path):
        assert "rubric" in system
        assert "positive_criteria" in system
        assert "negative_criteria" in system
        assert "decisive_evidence" in system
        # 硬约束：从 raw_text/failure_summary 蒸、禁标签名凭空发挥
        assert "raw_text" in system and "failure_summary" in system


def test_call2_both_prompts_define_three_capability_kinds(taxonomy_v0_path):
    for system in _both_call2_systems(taxonomy_v0_path):
        assert "capability_kind" in system
        assert "presence" in system
        assert "avoidance" in system
        assert "recovery" in system


# ── PR-2: drop_reason no_trajectory_evidence 三处同源（两份 prompt 各 3 处）──


def test_call2_both_prompts_enum_string_has_new_reason(taxonomy_v0_path):
    # 输出格式 JSON 示例里的枚举串
    for system in _both_call2_systems(taxonomy_v0_path):
        assert "no_trajectory_evidence" in system


def test_call2_both_prompts_decision_tree_has_new_reason_row(taxonomy_v0_path):
    # 判定树表格该行 + 关键语义约束：无轨迹严禁输出此值
    for system in _both_call2_systems(taxonomy_v0_path):
        assert "| `no_trajectory_evidence` |" in system
        assert "严禁输出此值" in system


def test_call2_both_prompts_wording_says_five(taxonomy_v0_path):
    for system in _both_call2_systems(taxonomy_v0_path):
        assert "上表五个之一" in system
        assert "上表四个之一" not in system


def test_call2_both_prompts_guard_no_evidence_object_without_section(taxonomy_v0_path):
    # 对称守卫（PR-2 审查 🟡C）：无失败轨迹段时绝不输出 failure_evidence 对象，
    # 与 no_trajectory_evidence 的"严禁输出此值"守卫对称，防幻觉 trajectory_id 污染审计。
    for system in _both_call2_systems(taxonomy_v0_path):
        assert "绝不输出 `failure_evidence` 对象" in system


# ── PR-2: failure_evidence 注入（在场含轨迹文本、缺省与旧版一致）──


def test_call2_failure_evidence_injected_when_present(taxonomy_v0_path):
    from module0.taxonomy import Taxonomy as _T
    traj_text = "MAGIC_TRAJECTORY_TEXT_步骤3报错SyntaxError"
    msgs = build_call2_messages(
        [{"id": "p1", "raw_text": "x", "failure_summary": "y"}],
        _T.load(taxonomy_v0_path), failure_evidence=traj_text)
    system = msgs[0]["content"]
    assert traj_text in system
    assert "failure_evidence" in system
    assert "no_trajectory_evidence" in system  # 认领不到 → drop 说明在场


def test_call2_failure_evidence_none_matches_legacy(taxonomy_v0_path):
    from module0.taxonomy import Taxonomy as _T
    tax = _T.load(taxonomy_v0_path)
    subs = [{"id": "p1", "raw_text": "x", "failure_summary": "y"}]
    default = build_call2_messages(subs, tax)[0]["content"]
    explicit_none = build_call2_messages(subs, tax, failure_evidence=None)[0]["content"]
    # 默认 == 显式 None，且都不含证据认领段落
    assert default == explicit_none
    assert "失败轨迹证据（在场" not in default
    # rubric 段恒在（与证据段无关）
    assert "rubric" in default


def test_call2_prime_never_carries_failure_evidence(taxonomy_v0_path):
    # 决策 3：Call 2' 复用 build_call2_messages 显式传 None，永不带轨迹证据
    from module0.prompts import build_call2_prime_messages
    from module0.taxonomy import Taxonomy as _T
    msgs = build_call2_prime_messages(
        [{"id": "p1a", "raw_text": "x", "failure_summary": "y"}],
        _T.load(taxonomy_v0_path))
    system = msgs[0]["content"]
    assert "失败轨迹证据（在场" not in system
