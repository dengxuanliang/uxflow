"""Measure real-Qwen cosine between child label descriptions and their true parents.

Prints the distribution so mount_threshold can be set from data, not guessed.
Run locally (needs the model): .venv/bin/python scripts/calibrate_mount_threshold.py
"""
# SPDX-License-Identifier: Apache-2.0
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
from module0.embedding import EmbeddingModel


# (child description, its intended parent description) pairs drawn from the taxonomy domain
PAIRS = [
    ("修复运行时抛出的异常，如 TypeError/KeyError/ValueError", "从错误中恢复并修复问题"),
    ("修复导入模块失败的问题", "从错误中恢复并修复问题"),
    ("为函数补充单元测试", "验证代码正确性"),
]


def _cos(a, b):
    va, vb = np.asarray(a), np.asarray(b)
    return float(va @ vb / (np.linalg.norm(va) * np.linalg.norm(vb)))


def main():
    emb = EmbeddingModel()
    sims = []
    for child, parent in PAIRS:
        s = _cos(emb.embed(child), emb.embed(parent))
        sims.append(s)
        print(f"cos={s:.3f}  child={child[:20]!r} parent={parent[:20]!r}")
    print(f"\nmin={min(sims):.3f} max={max(sims):.3f} mean={sum(sims)/len(sims):.3f}")
    print("→ set mount_threshold just BELOW min so true children mount; above unrelated noise.")


if __name__ == "__main__":
    main()
