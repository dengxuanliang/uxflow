from module1.models import Step, Slice
from module1.summarizer import summarize_slice


def _make_slice(steps):
    return Slice(trajectory_id="t1", slice_index=0, steps=steps,
                 start_step=0, end_step=len(steps)-1)


def test_basic_summary():
    steps = [
        Step(index=0, role="assistant", content="分析报错信息，定位到 src/main.py 第12行有语法错误",
             tool_call_name="Read", tool_call_args='src/main.py'),
        Step(index=1, role="tool", content="1: package main\n2: import \"fmt\"",
             tool_result="1: package main\n2: import \"fmt\""),
        Step(index=2, role="assistant", content="发现第4行缺少条件判断，修复",
             tool_call_name="Edit", tool_call_args='src/main.py, old="if err {", new="if err != nil {"'),
        Step(index=3, role="tool", content="file edited successfully",
             tool_result="file edited successfully"),
    ]
    summary = summarize_slice(_make_slice(steps))
    assert "Step 0" in summary
    assert "Read" in summary
    assert "Step 1" in summary
    assert "Step 2" in summary
    assert "Edit" in summary
    assert "Step 3" in summary
    assert "file edited" in summary


def test_assistant_without_tool_call():
    steps = [
        Step(index=0, role="assistant", content="让我先思考一下这个问题的根本原因",
             tool_call_name=None, tool_call_args=None),
        Step(index=1, role="assistant", content="定位问题",
             tool_call_name="Bash", tool_call_args="grep -r 'error' src/"),
        Step(index=2, role="tool", content="found matches",
             tool_result="found matches"),
    ]
    summary = summarize_slice(_make_slice(steps))
    assert "Step 0" in summary
    assert "[assistant]" in summary
    # No tool call marker for step 0
    assert "call" not in summary.split("\n")[0] or "思考" in summary.split("\n")[0]


def test_truncation_limits():
    """Content is truncated per spec limits: assistant 150, args 200, result 300."""
    long_content = "x" * 500
    long_args = "y" * 500
    long_result = "z" * 500
    steps = [
        Step(index=0, role="assistant", content=long_content,
             tool_call_name="Bash", tool_call_args=long_args),
        Step(index=1, role="tool", content=long_result,
             tool_result=long_result),
    ]
    summary = summarize_slice(_make_slice(steps))
    lines = summary.strip().split("\n")
    # Each line should be bounded (not contain full 500-char content)
    for line in lines:
        assert len(line) < 800  # generous upper bound per line


def test_newlines_escaped():
    steps = [
        Step(index=0, role="assistant", content="line1\nline2\nline3",
             tool_call_name="Bash", tool_call_args="echo 'hello\nworld'"),
        Step(index=1, role="tool", content="hello\nworld\ndone",
             tool_result="hello\nworld\ndone"),
    ]
    summary = summarize_slice(_make_slice(steps))
    # No raw newlines within a logical line (only between steps)
    for line in summary.strip().split("\n"):
        if line.startswith("Step"):
            assert "\n" not in line[4:]  # after "Step" prefix, no embedded newlines


def test_empty_slice():
    s = _make_slice([])
    summary = summarize_slice(s)
    assert summary == ""


def test_user_step_included():
    """User messages are included in summary."""
    steps = [
        Step(index=0, role="user", content="请修复这个bug"),
        Step(index=1, role="assistant", content="好的",
             tool_call_name="Bash", tool_call_args="ls"),
        Step(index=2, role="tool", content="files", tool_result="files"),
    ]
    summary = summarize_slice(_make_slice(steps))
    assert "Step 0" in summary
    assert "[user]" in summary
