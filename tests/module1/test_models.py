from module1.models import Step, Trajectory, Slice, TrajectorySignature, JudgeResult, SFTCandidate


def test_step_creation():
    s = Step(index=0, role="assistant", content="analyzing...",
             tool_call_name="Bash", tool_call_args="ls -la", tool_result=None)
    assert s.role == "assistant"
    assert s.tool_call_name == "Bash"


def test_trajectory_step_count():
    steps = [Step(index=i, role="assistant", content="", tool_call_name=None, tool_call_args=None, tool_result=None) for i in range(5)]
    t = Trajectory(id="traj_001", steps=steps, raw_messages=[])
    assert t.step_count == 5


def test_slice_from_trajectory():
    steps = [Step(index=i, role="assistant", content="", tool_call_name=None, tool_call_args=None, tool_result=None) for i in range(10)]
    s = Slice(trajectory_id="traj_001", slice_index=0, steps=steps[:5], start_step=0, end_step=4)
    assert s.step_count == 5


def test_judge_result():
    jr = JudgeResult(match=True, confidence=0.88, spans=[{"start_step": 2, "end_step": 5}], reasoning="good")
    assert jr.match
    assert jr.spans[0]["start_step"] == 2


def test_sft_candidate():
    c = SFTCandidate(
        trajectory_id="traj_001",
        trajectory_path="/data/traj_001.jsonl",
        matched_problems=[{
            "problem_spec_id": "spec_001",
            "sub_problem_id": "p1",
            "capability": ["valid_syntax_in_toolcall"],
            "confidence": 0.88,
            "loss_mask_spans": [{"start_step": 2, "end_step": 5}],
        }],
    )
    assert c.trajectory_id == "traj_001"
