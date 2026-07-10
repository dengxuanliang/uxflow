from module1.models import Step, Trajectory
from module1.slicer import slice_trajectory

STEP_THRESHOLD = 10


def _make_steps(n, roles=None):
    """Helper: create n steps with alternating assistant/tool roles."""
    steps = []
    for i in range(n):
        if roles:
            role = roles[i % len(roles)]
        else:
            role = "assistant" if i % 2 == 0 else "tool"
        steps.append(Step(
            index=i, role=role, content=f"step {i}",
            tool_call_name="Bash" if role == "assistant" else None,
            tool_call_args="cmd" if role == "assistant" else None,
            tool_result=f"result {i}" if role == "tool" else None,
        ))
    return steps


def test_short_trajectory_single_slice():
    """≤10 steps → 1 slice = whole trajectory."""
    steps = _make_steps(6)
    t = Trajectory(id="t1", steps=steps, raw_messages=[])
    slices = slice_trajectory(t)
    assert len(slices) == 1
    assert slices[0].start_step == 0
    assert slices[0].end_step == 5
    assert slices[0].step_count == 6


def test_exactly_10_steps_single_slice():
    steps = _make_steps(10)
    t = Trajectory(id="t1", steps=steps, raw_messages=[])
    slices = slice_trajectory(t)
    assert len(slices) == 1


def test_long_trajectory_multiple_slices():
    """>10 steps → multiple slices, each 5-10 steps."""
    steps = _make_steps(20)
    t = Trajectory(id="t1", steps=steps, raw_messages=[])
    slices = slice_trajectory(t)
    assert len(slices) >= 2
    for s in slices:
        assert s.step_count >= 3  # not too small
        assert s.step_count <= 12  # not too large (some tolerance)


def test_slices_cover_all_steps():
    """Slices must cover every step exactly once, no gaps/overlaps."""
    steps = _make_steps(15)
    t = Trajectory(id="t1", steps=steps, raw_messages=[])
    slices = slice_trajectory(t)
    all_indices = []
    for s in slices:
        all_indices.extend(range(s.start_step, s.end_step + 1))
    assert sorted(all_indices) == list(range(15))


def test_user_message_is_boundary():
    """A user message should trigger a slice boundary."""
    steps = _make_steps(14)
    # Insert a user message at index 7
    steps[7] = Step(index=7, role="user", content="new instruction",
                    tool_call_name=None, tool_call_args=None, tool_result=None)
    t = Trajectory(id="t1", steps=steps, raw_messages=[])
    slices = slice_trajectory(t)
    # Should have a boundary at or near index 7
    boundaries = [s.start_step for s in slices[1:]]
    assert 7 in boundaries or 8 in boundaries


def test_empty_trajectory():
    t = Trajectory(id="t1", steps=[], raw_messages=[])
    slices = slice_trajectory(t)
    assert slices == []
