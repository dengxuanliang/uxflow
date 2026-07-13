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


def test_malformed_and_nondict_lines_skipped(tmp_path):
    p = tmp_path / "mixed.jsonl"
    # line1 valid, line2 malformed JSON, line3 valid-JSON-but-not-dict, line4 valid
    p.write_text(
        '{"id":"a","messages":[]}\n'
        '}{ this is broken json\n'
        '[1,2,3]\n'
        '{"id":"b","messages":[]}\n'
    )
    trajs = load_trajectories(p)
    ids = [t.id for t in trajs]
    assert ids == ["a", "b"]   # both bad lines skipped, valid ones kept, no exception


def test_nonlist_messages_and_nondict_msg_skipped(tmp_path):
    p = tmp_path / "bad.jsonl"
    p.write_text(
        '{"id":"a","messages":[]}\n'
        '{"id":"bad1","messages":42}\n'
        '{"id":"bad2","messages":[1,2,3]}\n'
        '{"id":"b","messages":[{"role":"user","content":"hi"}]}\n'
    )
    trajs = load_trajectories(p)
    ids = [t.id for t in trajs]
    # bad1 (messages 非 list) 整行跳过；bad2 (messages 是 list、元素非 dict)
    # 进 parse_trajectory 但元素被逐个跳过 → 保留为 0-step Trajectory。
    assert "a" in ids and "b" in ids
    assert "bad1" not in ids
    assert "bad2" in ids
    bad2 = next(t for t in trajs if t.id == "bad2")
    assert bad2.step_count == 0


def test_nonlist_tool_calls_does_not_abort_file(tmp_path):
    # tool_calls that isn't a list must not crash parse_trajectory / abort the file
    p = tmp_path / "bad_tc.jsonl"
    p.write_text(
        '{"id":"a","messages":[]}\n'
        '{"id":"badtc","messages":[{"role":"assistant","tool_calls":42}]}\n'
        '{"id":"b","messages":[{"role":"user","content":"hi"}]}\n'
    )
    trajs = load_trajectories(p)
    ids = [t.id for t in trajs]
    # non-list tool_calls is ignored (assistant msg kept as a plain step); the
    # valid neighbors survive — the whole file is NOT aborted.
    assert "a" in ids and "b" in ids
    assert "badtc" in ids
    badtc = next(t for t in trajs if t.id == "badtc")
    assert badtc.step_count == 1  # the assistant msg becomes one plain step
