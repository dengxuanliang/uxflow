import json

from module1.claude_transcript_adapter import (
    convert_transcript_file,
    convert_transcript_records,
)


def test_converts_claude_blocks_to_loader_messages():
    records = [
        {"type": "system", "content": "system note"},
        {"type": "user", "message": {"role": "user", "content": "Fix it"}},
        {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "Need to inspect."},
                    {"type": "text", "text": "I will read the file."},
                    {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "Read",
                        "input": {"file_path": "src/a.py"},
                    },
                ],
            },
        },
        {
            "type": "user",
            "message": {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_1",
                        "content": "file contents",
                    }
                ],
            },
        },
    ]

    converted = convert_transcript_records("session_1", records)

    assert converted is not None
    assert converted["id"] == "session_1"
    assert converted["messages"] == [
        {"role": "system", "content": "system note"},
        {"role": "user", "content": "Fix it"},
        {
            "role": "assistant",
            "content": "I will read the file.",
            "reasoning": "Need to inspect.",
            "tool_calls": [
                {
                    "id": "toolu_1",
                    "type": "function",
                    "function": {
                        "name": "Read",
                        "arguments": "{\"file_path\":\"src/a.py\"}",
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "toolu_1", "content": "file contents"},
    ]


def test_skips_records_without_core_complete_tool_exchange():
    records = [
        {"type": "user", "message": {"role": "user", "content": "Fix it"}},
        {"type": "assistant", "message": {"role": "assistant", "content": "Done"}},
    ]

    assert convert_transcript_records("session_1", records) is None


def test_reads_jsonl_file_and_ignores_malformed_event_lines(tmp_path):
    path = tmp_path / "session.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps({"type": "user", "message": {"role": "user", "content": "Fix it"}}),
                "not-json",
                json.dumps(
                    {
                        "type": "assistant",
                        "message": {
                            "role": "assistant",
                            "content": [
                                {"type": "text", "text": "Reading."},
                                {
                                    "type": "tool_use",
                                    "id": "toolu_1",
                                    "name": "Read",
                                    "input": {"file_path": "a.py"},
                                },
                            ],
                        },
                    }
                ),
                json.dumps(
                    {
                        "type": "user",
                        "message": {
                            "role": "user",
                            "content": [
                                {
                                    "type": "tool_result",
                                    "tool_use_id": "toolu_1",
                                    "content": "ok",
                                }
                            ],
                        },
                    }
                ),
            ]
        )
        + "\n"
    )

    converted = convert_transcript_file(path)

    assert converted is not None
    assert converted["id"] == "session"
    assert [m["role"] for m in converted["messages"]] == ["user", "assistant", "tool"]
