"""Generate rich, realistic SWE-agent trajectories for the Inspector demo.

Not a test fixture — writes fixtures/inspector_demo/candidate_trajectories.jsonl.
Each trajectory mimics a real Claude-Code-style agent loop (read → analyze →
reproduce → edit → verify, with real tracebacks / pytest output) so slices are
long enough to show the slicing effect and matched segments look authentic.
"""
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 dengxuanliang

import json
import pathlib

OUT = pathlib.Path(__file__).parent.parent / "fixtures" / "inspector_demo" / "candidate_trajectories.jsonl"

SYSTEM = ("You are an expert software engineering agent. Fix issues in the "
          "repository using the available tools. Reproduce before fixing, and "
          "verify your fix by running the tests before reporting done.")


_tc_counter = [0]


def a(content, name=None, args=None):
    """assistant step, optionally with one tool call."""
    m = {"role": "assistant", "content": content}
    if name:
        _tc_counter[0] += 1
        m["tool_calls"] = [{"id": f"call_{_tc_counter[0]:03d}", "type": "function",
                            "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)}}]
    return m


def tool(content):
    return {"role": "tool", "content": content}


def user(content):
    return {"role": "user", "content": content}


def traj(tid, msgs):
    return {"id": tid, "messages": [{"role": "system", "content": SYSTEM}] + msgs}


TRAJS = []

# ── 1. valid_syntax + correct_indentation: reproduce, fix syntax, fix indent, verify ──
TRAJS.append(traj("swe_001", [
    user("The module src/parser.py won't import — there's a syntax error and the "
         "indentation looks off in parse_config. Fix it and make the tests pass."),
    a("Let me first reproduce the failure by importing the module.",
      "Bash", {"command": "python -c 'import src.parser'"}),
    tool("  File \"src/parser.py\", line 12\n    def parse_config(path)\n"
         "                         ^\nSyntaxError: expected ':'"),
    a("Confirmed a SyntaxError on the function header. Reading the file to see the "
      "surrounding context before editing.",
      "Read", {"file_path": "src/parser.py"}),
    tool("10: import json\n11:\n12: def parse_config(path)\n13:   data = json.load(open(path))\n"
         "14:      for k in data:\n15:        yield k, data[k]\n"),
    a("Two problems: (1) missing colon on line 12, (2) inconsistent indentation on "
      "lines 13-15 (2 spaces then 6 spaces). Fixing the colon first.",
      "Edit", {"file_path": "src/parser.py", "old_string": "def parse_config(path)",
               "new_string": "def parse_config(path):"}),
    tool("file edited successfully"),
    a("Now normalizing the indentation inside the function to 4 spaces.",
      "Edit", {"file_path": "src/parser.py",
               "old_string": "   data = json.load(open(path))\n      for k in data:\n        yield k, data[k]",
               "new_string": "    data = json.load(open(path))\n    for k in data:\n        yield k, data[k]"}),
    tool("file edited successfully"),
    a("Re-importing to confirm the syntax error is gone.",
      "Bash", {"command": "python -c 'import src.parser; print(\"ok\")'"}),
    tool("ok"),
    a("Import works. Running the parser tests to verify behavior.",
      "Bash", {"command": "python -m pytest tests/test_parser.py -q"}),
    tool("....\n4 passed in 0.12s"),
    a("All 4 tests pass. The SyntaxError and the indentation are both fixed."),
]))

# ── 2. self_verification: implement then verify (positive) ──
TRAJS.append(traj("swe_002", [
    user("Add a `slugify` helper to src/text.py that lowercases and replaces "
         "spaces with hyphens."),
    a("Reading the existing text utilities to match the style.",
      "Read", {"file_path": "src/text.py"}),
    tool("1: import re\n2:\n3: def truncate(s, n):\n4:     return s[:n]\n"),
    a("Adding the slugify function.",
      "Edit", {"file_path": "src/text.py", "old_string": "def truncate(s, n):\n    return s[:n]",
               "new_string": "def truncate(s, n):\n    return s[:n]\n\n\ndef slugify(s):\n    return re.sub(r'\\s+', '-', s.strip().lower())"}),
    tool("file edited successfully"),
    a("Before reporting done, I'll verify with a quick manual check and then the tests.",
      "Bash", {"command": "python -c \"from src.text import slugify; print(slugify('  Hello World  '))\""}),
    tool("hello-world"),
    a("Output is correct. Running the test suite to be sure nothing regressed.",
      "Bash", {"command": "python -m pytest tests/test_text.py -q"}),
    tool("......\n6 passed in 0.08s"),
    a("Verified — slugify works and all text tests pass."),
]))

# ── 3. reproduce_before_fix: reproduce a KeyError first ──
TRAJS.append(traj("swe_003", [
    user("Users report that GET /profile crashes when the account has no avatar. "
         "Fix it in src/views.py."),
    a("I'll reproduce the crash first before changing anything.",
      "Bash", {"command": "python -m pytest tests/test_views.py::test_profile_no_avatar -q"}),
    tool("F\n=================================== FAILURES ===================================\n"
         "___________________________ test_profile_no_avatar ___________________________\n"
         "    def render_profile(account):\n"
         ">       url = account['avatar']['url']\n"
         "E       KeyError: 'avatar'\n\nsrc/views.py:23: KeyError"),
    a("Reproduced: KeyError on account['avatar'] when the key is missing. "
      "Reading the function to fix it properly.",
      "Read", {"file_path": "src/views.py"}),
    tool("21: def render_profile(account):\n22:     name = account['name']\n"
         "23:     url = account['avatar']['url']\n24:     return {'name': name, 'avatar': url}\n"),
    a("Using .get with a default so a missing avatar falls back to a placeholder.",
      "Edit", {"file_path": "src/views.py",
               "old_string": "    url = account['avatar']['url']",
               "new_string": "    url = account.get('avatar', {}).get('url', '/static/default.png')"}),
    tool("file edited successfully"),
    a("Re-running the exact test I used to reproduce.",
      "Bash", {"command": "python -m pytest tests/test_views.py::test_profile_no_avatar -q"}),
    tool(".\n1 passed in 0.05s"),
    a("The reproducing test now passes. Fix confirmed."),
]))

# ── 4. avoid_redundant_repetition: NEGATIVE — retries same thing 5x, loops ──
TRAJS.append(traj("swe_004", [
    user("The build fails with 'ModuleNotFoundError: numpy'. Make it work."),
    a("Trying to install numpy.", "Bash", {"command": "pip install numpy"}),
    tool("ERROR: Could not find a version that satisfies the requirement numpy\n"
         "ERROR: No matching distribution found for numpy"),
    a("Retrying the install.", "Bash", {"command": "pip install numpy"}),
    tool("ERROR: Could not find a version that satisfies the requirement numpy"),
    a("Trying again.", "Bash", {"command": "pip install numpy"}),
    tool("ERROR: Could not find a version that satisfies the requirement numpy"),
    a("Once more.", "Bash", {"command": "pip install numpy"}),
    tool("ERROR: Could not find a version that satisfies the requirement numpy"),
    a("Trying the install one more time.", "Bash", {"command": "pip install numpy"}),
    tool("ERROR: Could not find a version that satisfies the requirement numpy"),
    a("Retrying.", "Bash", {"command": "pip install numpy"}),
    tool("ERROR: Could not find a version that satisfies the requirement numpy"),
    a("Still failing, trying the same command again.", "Bash", {"command": "pip install numpy"}),
    tool("ERROR: Could not find a version that satisfies the requirement numpy"),
]))

# ── 5. effective_error_fix: user says still broken, agent re-diagnoses & really fixes ──
TRAJS.append(traj("swe_005", [
    user("The date parser returns wrong results for '2024-13-01'. Please fix."),
    a("Adding a guard for months > 12.",
      "Edit", {"file_path": "src/dates.py", "old_string": "month = int(parts[1])",
               "new_string": "month = int(parts[1])\n    if month > 12:\n        month = 12"}),
    tool("file edited successfully"),
    a("Fixed — clamped the month to 12."),
    user("That's still wrong. '2024-13-01' should raise an error, not silently become December."),
    a("You're right, clamping hides the bug. Let me reproduce to see the real behavior first.",
      "Bash", {"command": "python -c \"from src.dates import parse; print(parse('2024-13-01'))\""}),
    tool("datetime.date(2024, 12, 1)"),
    a("Confirmed it silently returns December. Reading the function to raise instead.",
      "Read", {"file_path": "src/dates.py"}),
    tool("5: def parse(s):\n6:     parts = s.split('-')\n7:     month = int(parts[1])\n"
         "8:     if month > 12:\n9:         month = 12\n10:     return date(int(parts[0]), month, int(parts[2]))\n"),
    a("Replacing the clamp with a validation that raises ValueError.",
      "Edit", {"file_path": "src/dates.py",
               "old_string": "    if month > 12:\n        month = 12",
               "new_string": "    if not 1 <= month <= 12:\n        raise ValueError(f'invalid month: {month}')"}),
    tool("file edited successfully"),
    a("Verifying it now raises for the bad input and still works for valid dates.",
      "Bash", {"command": "python -m pytest tests/test_dates.py -q"}),
    tool("........\n8 passed in 0.09s"),
    a("Now '2024-13-01' raises ValueError and valid dates still parse. Properly fixed this time."),
]))

# ── 6. correct_shell_embedding: python embedded in bash, fix quoting/closure ──
TRAJS.append(traj("swe_006", [
    user("Run a one-liner to count lines in all .py files under src/ and print the total."),
    a("Running an inline python one-liner from bash.",
      "Bash", {"command": "python -c \"import glob; print(sum(len(open(f).readlines()) for f in glob.glob('src/**/*.py')"}),
    tool("  File \"<string>\", line 1\n    import glob; print(sum(len(open(f).readlines()) for f in glob.glob('src/**/*.py')\n"
         "                                                                                         ^\n"
         "SyntaxError: '(' was never closed"),
    a("The parenthesis for sum() was never closed inside the embedded snippet. "
      "Fixing the closure and adding recursive=True for the ** glob.",
      "Bash", {"command": "python -c \"import glob; print(sum(len(open(f).readlines()) for f in glob.glob('src/**/*.py', recursive=True)))\""}),
    tool("482"),
    a("The embedded python now parses and runs — 482 total lines."),
]))

# ── 7. valid_syntax (lambda) — longer, with a wrong first attempt ──
TRAJS.append(traj("swe_007", [
    user("src/sort.py has a broken lambda in `by_score`. Fix the syntax and verify sorting works."),
    a("Reproducing the import error.", "Bash", {"command": "python -c 'import src.sort'"}),
    tool("  File \"src/sort.py\", line 4\n    by_score = lambda x: x['score'\n"
         "                                    ^\nSyntaxError: '[' was never closed"),
    a("Reading the file.", "Read", {"file_path": "src/sort.py"}),
    tool("3: def top(items):\n4:     by_score = lambda x: x['score'\n5:     return sorted(items, key=by_score, reverse=True)\n"),
    a("Closing the subscript bracket in the lambda.",
      "Edit", {"file_path": "src/sort.py", "old_string": "by_score = lambda x: x['score'",
               "new_string": "by_score = lambda x: x['score']"}),
    tool("file edited successfully"),
    a("Verifying import and a small sort.",
      "Bash", {"command": "python -c \"from src.sort import top; print(top([{'score':2},{'score':5},{'score':1}]))\""}),
    tool("[{'score': 5}, {'score': 2}, {'score': 1}]"),
    a("Sorted descending by score as expected. Fixed."),
]))

# ── 8. self_verification NEGATIVE — implements, claims done, never runs anything ──
TRAJS.append(traj("swe_008", [
    user("Implement a `is_palindrome` function in src/strings.py."),
    a("Reading the file.", "Read", {"file_path": "src/strings.py"}),
    tool("1: def reverse(s):\n2:     return s[::-1]\n"),
    a("Adding is_palindrome.",
      "Edit", {"file_path": "src/strings.py", "old_string": "def reverse(s):\n    return s[::-1]",
               "new_string": "def reverse(s):\n    return s[::-1]\n\n\ndef is_palindrome(s):\n    return s == reverse(s)"}),
    tool("file edited successfully"),
    a("Done — implemented is_palindrome using the existing reverse helper."),
]))

# ── 9. reproduce_before_fix + effective fix, IndexError, longer ──
TRAJS.append(traj("swe_009", [
    user("src/stack.py: calling pop() on an empty stack crashes the server. Handle it."),
    a("Reproducing the crash before touching the code.",
      "Bash", {"command": "python -c \"from src.stack import Stack; Stack().pop()\""}),
    tool("Traceback (most recent call last):\n  File \"<string>\", line 1, in <module>\n"
         "  File \"src/stack.py\", line 8, in pop\n    return self._items.pop()\n"
         "IndexError: pop from empty list"),
    a("Reproduced the IndexError. Reading the class.",
      "Read", {"file_path": "src/stack.py"}),
    tool("5: class Stack:\n6:     def __init__(self): self._items = []\n"
         "7:     def pop(self):\n8:         return self._items.pop()\n"),
    a("Grepping how callers expect empty-pop to behave.",
      "Grep", {"pattern": "\\.pop\\(", "path": "src"}),
    tool("src/worker.py:14:    item = stack.pop()  # returns None when idle\n"),
    a("Callers expect None on empty. Returning None instead of raising.",
      "Edit", {"file_path": "src/stack.py",
               "old_string": "    def pop(self):\n        return self._items.pop()",
               "new_string": "    def pop(self):\n        if not self._items:\n            return None\n        return self._items.pop()"}),
    tool("file edited successfully"),
    a("Verifying empty pop returns None and normal pop still works.",
      "Bash", {"command": "python -m pytest tests/test_stack.py -q"}),
    tool(".....\n5 passed in 0.06s"),
    a("Empty pop now returns None; all stack tests pass."),
]))

# ── 10. correct_indentation — mixed tabs/spaces, longer ──
TRAJS.append(traj("swe_010", [
    user("src/report.py raises TabError. Clean up the indentation."),
    a("Reproducing.", "Bash", {"command": "python -c 'import src.report'"}),
    tool("  File \"src/report.py\", line 6\n    total += row.amount\n^\n"
         "TabError: inconsistent use of tabs and spaces in indentation"),
    a("Reading to see the mixed indentation.", "Read", {"file_path": "src/report.py"}),
    tool("3: def summarize(rows):\n4:     total = 0\n5:     for row in rows:\n"
         "6: \t    total += row.amount\n7:     return total\n"),
    a("Line 6 uses a tab then spaces. Replacing with 8 consistent spaces (one level "
      "under the for-loop).",
      "Edit", {"file_path": "src/report.py", "old_string": "\t    total += row.amount",
               "new_string": "        total += row.amount"}),
    tool("file edited successfully"),
    a("Confirming import and running tests.",
      "Bash", {"command": "python -m pytest tests/test_report.py -q"}),
    tool("...\n3 passed in 0.04s"),
    a("TabError resolved; indentation is now consistent and tests pass."),
]))

# ── 11. distractor: pure README doc edit, no code capability ──
TRAJS.append(traj("swe_011", [
    user("Update the README install section to mention Python 3.11+."),
    a("Reading the README.", "Read", {"file_path": "README.md"}),
    tool("## Install\n\nRun `pip install foo`.\n"),
    a("Adding the version note.",
      "Edit", {"file_path": "README.md", "old_string": "Run `pip install foo`.",
               "new_string": "Requires Python 3.11+.\n\nRun `pip install foo`."}),
    tool("file edited successfully"),
    a("README updated."),
]))

# ── 12. wellformed_tool_call / file_localization — Grep then edit correct file ──
TRAJS.append(traj("swe_012", [
    user("There's a hardcoded localhost URL somewhere in src/. Point it at the "
         "API_BASE env var instead."),
    a("Searching for the hardcoded URL across the source tree.",
      "Grep", {"pattern": "http://localhost", "path": "src"}),
    tool("src/client.py:9:    BASE = 'http://localhost:8000'\n"
         "src/legacy/old_client.py:4:    BASE = 'http://localhost:8000'  # deprecated\n"),
    a("Two hits; only src/client.py is active (the other is deprecated). Reading it.",
      "Read", {"file_path": "src/client.py"}),
    tool("7: import os\n8:\n9:     BASE = 'http://localhost:8000'\n"),
    a("Replacing the hardcoded URL with the env var, defaulting to localhost.",
      "Edit", {"file_path": "src/client.py", "old_string": "    BASE = 'http://localhost:8000'",
               "new_string": "    BASE = os.environ.get('API_BASE', 'http://localhost:8000')"}),
    tool("file edited successfully"),
    a("Verifying it reads the env var.",
      "Bash", {"command": "API_BASE=https://api.example.com python -c \"from src.client import BASE; print(BASE)\""}),
    tool("https://api.example.com"),
    a("The active client now honors API_BASE; left the deprecated file untouched."),
]))

# ── 13. valid_syntax — unterminated string, quick ──
TRAJS.append(traj("swe_013", [
    user("src/greet.py has an unterminated string literal. Fix it."),
    a("Reproducing.", "Bash", {"command": "python -c 'import src.greet'"}),
    tool("  File \"src/greet.py\", line 2\n    MSG = 'hello\n              ^\n"
         "SyntaxError: unterminated string literal (detected at line 2)"),
    a("Reading the line.", "Read", {"file_path": "src/greet.py"}),
    tool("1: # greeting module\n2: MSG = 'hello\n3: def greet(): return MSG\n"),
    a("Closing the string literal.",
      "Edit", {"file_path": "src/greet.py", "old_string": "MSG = 'hello",
               "new_string": "MSG = 'hello'"}),
    tool("file edited successfully"),
    a("Confirming import works now.",
      "Bash", {"command": "python -c 'import src.greet; print(src.greet.greet())'"}),
    tool("hello"),
    a("Fixed — the string is terminated and the module imports."),
]))

# ── 14. avoid_redundant_repetition POSITIVE — tries, fails, then switches strategy ──
TRAJS.append(traj("swe_014", [
    user("Tests in tests/test_api.py hang forever. Make them pass."),
    a("Running the tests to see what happens.",
      "Bash", {"command": "python -m pytest tests/test_api.py -q --timeout=5"}),
    tool("tests/test_api.py::test_fetch FAILED\nE   Failed: Timeout >5.0s\n"
         "the test makes a real network call to http://api.example.com"),
    a("It hangs on a real network call. My first instinct was to retry, but that "
      "won't help — the fix is to mock the network instead of hitting it. Reading the test.",
      "Read", {"file_path": "tests/test_api.py"}),
    tool("1: def test_fetch():\n2:     r = fetch('http://api.example.com/data')\n3:     assert r['ok']\n"),
    a("Switching strategy: patching the HTTP call with a stub rather than retrying.",
      "Edit", {"file_path": "tests/test_api.py",
               "old_string": "def test_fetch():\n    r = fetch('http://api.example.com/data')\n    assert r['ok']",
               "new_string": "def test_fetch(monkeypatch):\n    monkeypatch.setattr('src.api._get', lambda u: {'ok': True})\n    r = fetch('http://api.example.com/data')\n    assert r['ok']"}),
    tool("file edited successfully"),
    a("Re-running with the network mocked out.",
      "Bash", {"command": "python -m pytest tests/test_api.py -q"}),
    tool(".\n1 passed in 0.03s"),
    a("Passing instantly now — the fix was to change approach (mock the call), not "
      "repeat the failing one."),
]))


def main():
    with open(OUT, "w", encoding="utf-8") as f:
        for t in TRAJS:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    steps = [len(t["messages"]) for t in TRAJS]
    print(f"wrote {len(TRAJS)} trajectories → {OUT}")
    print(f"steps per trajectory: min={min(steps)} max={max(steps)} "
          f"avg={sum(steps)/len(steps):.1f}")


if __name__ == "__main__":
    main()
