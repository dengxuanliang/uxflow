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
