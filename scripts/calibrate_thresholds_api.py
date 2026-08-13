"""Measure cosine similarity between child/parent pairs under the active embedding backend.

calibrate_mount_threshold.py hardcodes the local Qwen model directly, so it can
only speak to Qwen vectors. This script goes through make_embedder() instead,
so it reports whatever UXFLOW_EMBED_BACKEND selects — 'fake', 'local', or 'api'.

It also measures deliberately unrelated (child, wrong-parent) pairs alongside
the true (child, parent) pairs. The true pairs alone only show where real
children land; the unrelated pairs establish a noise floor, and it's the GAP
between the two that a threshold actually has to sit inside.

Run:
    UXFLOW_EMBED_BACKEND=fake uv run python scripts/calibrate_thresholds_api.py
    UXFLOW_EMBED_BACKEND=api  uv run python scripts/calibrate_thresholds_api.py

Records data only. Does NOT change mount_threshold anywhere — retuning it from
this data is deliberately a separate, later step (see design spec §4.8).
"""
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 dengxuanliang
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))

import numpy as np
from dotenv import load_dotenv

load_dotenv(pathlib.Path(__file__).parent.parent / ".env")

from uxflow_runtime import make_embedder  # noqa: E402

# Copied verbatim from calibrate_mount_threshold.py so the two backends'
# numbers are directly comparable — same (child description, true parent
# description) pairs drawn from the taxonomy domain.
PAIRS = [
    ("修复运行时抛出的异常，如 TypeError/KeyError/ValueError", "从错误中恢复并修复问题"),
    ("修复导入模块失败的问题", "从错误中恢复并修复问题"),
    ("为函数补充单元测试", "验证代码正确性"),
]

# Noise floor: cross each child against a parent that is NOT its true parent.
# Built by crossing the existing children against the existing (wrong) parents
# above — no new vocabulary introduced, so this isolates "wrong pairing" as
# the only variable rather than also varying topic/domain. Two of the three
# PAIRS entries happen to share the same true parent ("从错误中恢复并修复问题"),
# so parents are de-duplicated first — otherwise the same (child, wrong_parent)
# combination would be counted twice while contributing no new information.
_UNIQUE_PARENTS = list(dict.fromkeys(parent for _, parent in PAIRS))
UNRELATED_PAIRS = [
    (child, wrong_parent)
    for child, true_parent in PAIRS
    for wrong_parent in _UNIQUE_PARENTS
    if wrong_parent != true_parent
]


def _cos(a, b):
    va, vb = np.asarray(a), np.asarray(b)
    return float(va @ vb / (np.linalg.norm(va) * np.linalg.norm(vb)))


def main():
    emb = make_embedder()
    print(f"backend: {type(emb).__name__}  dimension={emb.dimension}\n")

    print("-- true child→parent pairs --")
    true_sims = []
    for child, parent in PAIRS:
        s = _cos(emb.embed(child), emb.embed(parent))
        true_sims.append(s)
        print(f"cos={s:.3f}  child={child[:24]!r} parent={parent[:20]!r}")

    print("\n-- unrelated child→wrong-parent pairs (noise floor) --")
    noise_sims = []
    for child, wrong_parent in UNRELATED_PAIRS:
        s = _cos(emb.embed(child), emb.embed(wrong_parent))
        noise_sims.append(s)
        print(f"cos={s:.3f}  child={child[:24]!r} wrong_parent={wrong_parent[:20]!r}")

    true_min, true_max = min(true_sims), max(true_sims)
    noise_min, noise_max = min(noise_sims), max(noise_sims)
    print(
        f"\ntrue:   min={true_min:.3f} max={true_max:.3f} "
        f"mean={sum(true_sims) / len(true_sims):.3f}"
    )
    print(
        f"noise:  min={noise_min:.3f} max={noise_max:.3f} "
        f"mean={sum(noise_sims) / len(noise_sims):.3f}"
    )

    gap = true_min - noise_max
    print(f"\nseparation gap (true_min - noise_max) = {gap:.3f}")

    default_threshold = 0.50
    if gap > 0:
        print(
            f"→ clean separation: every true pair ({true_min:.3f}) scores above "
            f"every noise pair ({noise_max:.3f})."
        )
        if noise_max < default_threshold <= true_min:
            print(
                f"→ default mount_threshold={default_threshold} falls INSIDE the "
                f"gap [{noise_max:.3f}, {true_min:.3f}] — still separates true "
                f"pairs from noise under this backend."
            )
        else:
            print(
                f"→ default mount_threshold={default_threshold} falls OUTSIDE the "
                f"gap [{noise_max:.3f}, {true_min:.3f}] — does NOT reliably "
                f"separate true pairs from noise under this backend."
            )
    else:
        print(
            "→ NO clean separation: at least one noise pair scores at or above "
            "the lowest true pair. There is no midpoint here that would mean "
            "anything as a threshold; more data (or a different measure) is "
            "needed before retuning."
        )


if __name__ == "__main__":
    main()
