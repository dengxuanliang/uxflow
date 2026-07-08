from module1.loader import load_trajectories, parse_trajectory


def test_load_from_jsonl(trajectories_path):
    trajs = load_trajectories(trajectories_path)
    assert len(trajs) >= 1
    assert trajs[0].id is not None
    assert trajs[0].step_count >= 1


def test_step_parsing(sample_trajectories):
    traj = parse_trajectory(sample_trajectories[0])
    roles = {s.role for s in traj.steps}
    assert "assistant" in roles


def test_tool_call_extraction(sample_trajectories):
    traj = parse_trajectory(sample_trajectories[0])
    tool_steps = [s for s in traj.steps if s.tool_call_name]
    assert len(tool_steps) >= 1


def test_empty_file(tmp_path):
    empty = tmp_path / "empty.jsonl"
    empty.write_text("")
    trajs = load_trajectories(empty)
    assert trajs == []
