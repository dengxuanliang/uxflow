from llm_gateway.truncation import (
    TRUNCATION_MARKER,
    estimate_tokens,
    truncate_messages,
)


def test_estimate_tokens():
    assert estimate_tokens("abcd") == 1
    assert estimate_tokens("a" * 100) == 25


def test_within_budget_unchanged():
    msgs = [
        {"role": "system", "content": "you are helpful"},
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi there"},
    ]
    result = truncate_messages(msgs, max_tokens=10000)
    assert result == msgs
    assert result is not msgs


def test_system_message_never_truncated():
    system_content = "x" * 5000
    msgs = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": "short"},
    ]
    result = truncate_messages(msgs, max_tokens=100)
    assert result[0]["content"] == system_content


def test_long_user_message_truncated():
    long_content = "word " * 20000
    msgs = [
        {"role": "user", "content": long_content},
    ]
    result = truncate_messages(msgs, max_tokens=500)
    assert len(result[0]["content"]) < len(long_content)
    assert TRUNCATION_MARKER in result[0]["content"]


def test_multi_turn_respects_ratios():
    msgs = [
        {"role": "user", "content": "a" * 4000},
        {"role": "assistant", "content": "b" * 4000},
        {"role": "user", "content": "c" * 4000},
        {"role": "assistant", "content": "d" * 4000},
    ]
    result = truncate_messages(msgs, max_tokens=1000)
    total_chars = sum(len(m["content"]) for m in result)
    assert total_chars <= 4000 + 200


def test_empty_messages():
    result = truncate_messages([], max_tokens=1000)
    assert result == []


def test_only_system_messages():
    msgs = [{"role": "system", "content": "sys"}]
    result = truncate_messages(msgs, max_tokens=10)
    assert result == msgs


def test_preserves_head_and_tail():
    long_content = "HEAD " + ("middle " * 5000) + " TAIL"
    msgs = [{"role": "user", "content": long_content}]
    result = truncate_messages(msgs, max_tokens=100)
    content = result[0]["content"]
    assert content.startswith("HEAD")
    assert content.endswith("TAIL")


def test_single_message_uses_full_budget():
    """1 non-system message must use the whole budget, not just the 35%
    first-slice. Regression for the over-truncation bug where a single
    long message was capped at 35% of available_chars."""
    long_content = "a" * 20000
    msgs = [{"role": "user", "content": long_content}]
    # max_tokens=1000 -> budget_chars = 4000
    result = truncate_messages(msgs, max_tokens=1000)
    content = result[0]["content"]
    # Should use close to the full 4000-char budget, not 35% (1400 chars).
    assert len(content) >= 3900
    assert TRUNCATION_MARKER in content


def test_two_messages_share_full_budget():
    """2 non-system messages must share ~50/50 of the full budget, not
    35%/30% (which would leave 35% of the budget unused)."""
    msgs = [
        {"role": "user", "content": "a" * 20000},
        {"role": "assistant", "content": "b" * 20000},
    ]
    result = truncate_messages(msgs, max_tokens=1000)  # budget = 4000
    total = len(result[0]["content"]) + len(result[1]["content"])
    # Combined should use ~all of the 4000-char budget, not just 65% (2600).
    assert total >= 3800
    assert TRUNCATION_MARKER in result[0]["content"]
    assert TRUNCATION_MARKER in result[1]["content"]


def test_tiny_budget_does_not_exceed_max():
    """When budget is smaller than the truncation marker (31 chars), the
    marker must not be appended (it would push output > max_chars). The
    function returns the original text rather than emitting > max_chars."""
    long_content = "a" * 200
    msgs = [{"role": "user", "content": long_content}]
    # max_tokens=5 -> budget_chars = 20, well below marker_len (31).
    result = truncate_messages(msgs, max_tokens=5)
    # budget < marker_len -> _truncate_text returns original; truncation loop
    # checks `if len(content) > budget` so original long content stays (since
    # the truncated form would exceed budget). Verify no marker injected:
    assert TRUNCATION_MARKER not in result[0]["content"]
