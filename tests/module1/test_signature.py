from module1.models import Step, Slice
from module1.signature import extract_signature


def _make_slice(steps):
    return Slice(trajectory_id="t1", slice_index=0, steps=steps,
                 start_step=0, end_step=len(steps)-1)


def test_languages_from_tool_call():
    steps = [
        Step(index=0, role="assistant", content="fix python",
             tool_call_name="Write", tool_call_args='{"path": "main.py", "content": "import os"}'),
        Step(index=1, role="tool", content="file created", tool_result="file created"),
    ]
    sig = extract_signature(_make_slice(steps))
    assert "python" in sig.languages


def test_tools_used():
    steps = [
        Step(index=0, role="assistant", content="read",
             tool_call_name="Read", tool_call_args="src/main.py"),
        Step(index=1, role="tool", content="content", tool_result="content"),
        Step(index=2, role="assistant", content="edit",
             tool_call_name="Edit", tool_call_args="fix"),
        Step(index=3, role="tool", content="done", tool_result="done"),
    ]
    sig = extract_signature(_make_slice(steps))
    assert "Read" in sig.tools_used
    assert "Edit" in sig.tools_used


def test_error_pattern_detected():
    steps = [
        Step(index=0, role="assistant", content="run", tool_call_name="Bash", tool_call_args="python main.py"),
        Step(index=1, role="tool", content="Traceback (most recent call last):\n  SyntaxError: invalid syntax",
             tool_result="Traceback (most recent call last):\n  SyntaxError: invalid syntax"),
    ]
    sig = extract_signature(_make_slice(steps))
    assert sig.has_error_pattern is True


def test_success_pattern_detected():
    steps = [
        Step(index=0, role="assistant", content="test", tool_call_name="Bash", tool_call_args="pytest"),
        Step(index=1, role="tool", content="5 passed, 0 failed",
             tool_result="5 passed, 0 failed"),
    ]
    sig = extract_signature(_make_slice(steps))
    assert sig.has_success_pattern is True


def test_verification_step():
    steps = [
        Step(index=0, role="assistant", content="verify",
             tool_call_name="Bash", tool_call_args="pytest tests/"),
        Step(index=1, role="tool", content="passed", tool_result="passed"),
    ]
    sig = extract_signature(_make_slice(steps))
    assert sig.has_verification_step is True


def test_bm25_tokens():
    steps = [
        Step(index=0, role="assistant", content="fix",
             tool_call_name="Write", tool_call_args="main.py"),
        Step(index=1, role="tool", content="SyntaxError: unexpected EOF",
             tool_result="SyntaxError: unexpected EOF"),
    ]
    sig = extract_signature(_make_slice(steps))
    assert "SyntaxError" in sig.bm25_tokens or "syntaxerror" in sig.bm25_tokens


def test_turn_count():
    steps = [
        Step(index=0, role="user", content="fix this"),
        Step(index=1, role="assistant", content="ok", tool_call_name="Bash", tool_call_args="ls"),
        Step(index=2, role="tool", content="files", tool_result="files"),
        Step(index=3, role="user", content="also do that"),
        Step(index=4, role="assistant", content="sure", tool_call_name="Edit", tool_call_args="x"),
        Step(index=5, role="tool", content="done", tool_result="done"),
    ]
    sig = extract_signature(_make_slice(steps))
    assert sig.turn_count == 2
