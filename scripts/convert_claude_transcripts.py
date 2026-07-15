"""Convert Claude Code transcript event streams into UXFlow trajectory JSONL."""
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 dengxuanliang

from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))

from module1.claude_transcript_adapter import convert_transcript_file  # noqa: E402


ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = ROOT / "dataset" / "transcripts"
DEFAULT_OUTPUT = ROOT / "dataset" / "swe-chat-converted.jsonl"


def convert_dir(input_dir: pathlib.Path, output: pathlib.Path) -> tuple[int, int]:
    converted = 0
    skipped = 0
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w") as out:
        for path in sorted(input_dir.glob("*.jsonl")):
            trajectory = convert_transcript_file(path)
            if trajectory is None:
                skipped += 1
                continue
            out.write(json.dumps(trajectory, ensure_ascii=False, separators=(",", ":")))
            out.write("\n")
            converted += 1

    return converted, skipped


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=pathlib.Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output", type=pathlib.Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    converted, skipped = convert_dir(args.input_dir, args.output)
    print(f"input_dir={args.input_dir}")
    print(f"output={args.output}")
    print(f"converted={converted}")
    print(f"skipped={skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
