"""Measure child→parent cosine separation under the active embedding backend.

calibrate_mount_threshold.py hardcodes the local Qwen model directly, so it can
only speak to Qwen vectors. This script goes through make_embedder() instead,
so it reports whatever UXFLOW_EMBED_BACKEND selects — 'fake', 'local', or 'api'.

Ground truth comes from fixtures/taxonomy_v0.json, which declares the real
root/child structure. Positives are each child against its declared parent;
negatives are each child against every root that is NOT its parent. Measuring
positives alone would only show where true children land — it's the overlap
between the two distributions that decides whether ANY single cutoff can work.

Run:
    UXFLOW_EMBED_BACKEND=fake uv run python scripts/calibrate_thresholds_api.py
    UXFLOW_EMBED_BACKEND=api  uv run python scripts/calibrate_thresholds_api.py

Records data only. Does NOT change mount_threshold anywhere — retuning it from
this data is deliberately a separate, later step (see design spec §4.8).
"""
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 dengxuanliang
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))

import numpy as np
from dotenv import load_dotenv

load_dotenv(pathlib.Path(__file__).parent.parent / ".env")

from uxflow_runtime import make_embedder  # noqa: E402

ROOT = pathlib.Path(__file__).parent.parent
TAXONOMY = ROOT / "fixtures" / "taxonomy_v0.json"

# The default this script is evaluating. Read-only: nothing here writes it back.
CURRENT_DEFAULT = 0.50


def _embed_text(entry: dict) -> str:
    """Label plus description — the root descriptions alone are very short."""
    return f"{entry['label']}: {entry['description']}"


def _cos(a, b) -> float:
    va, vb = np.asarray(a), np.asarray(b)
    return float(va @ vb / (np.linalg.norm(va) * np.linalg.norm(vb)))


def _stats(name: str, sims: list[float]) -> str:
    arr = np.asarray(sims)
    return (
        f"{name} n={len(sims):<3} min={arr.min():.3f} "
        f"p50={float(np.percentile(arr, 50)):.3f} max={arr.max():.3f}"
    )


def main():
    labels = json.loads(TAXONOMY.read_text(encoding="utf-8"))["labels"]
    roots = [entry for entry in labels if not entry.get("parent")]
    children = [entry for entry in labels if entry.get("parent")]

    emb = make_embedder()
    print(f"backend: {type(emb).__name__}  dimension={emb.dimension}")
    print(
        f"ground truth: {TAXONOMY.relative_to(ROOT)}  "
        f"{len(roots)} roots, {len(children)} children\n"
    )

    # One batch call for every label. The endpoint times out on roughly 30% of
    # requests, so 16 sequential single calls would very likely hit one.
    ordered = roots + children
    vectors = emb.embed_batch([_embed_text(entry) for entry in ordered])
    vec_by_label = {entry["label"]: vec for entry, vec in zip(ordered, vectors)}

    true_sims: list[float] = []
    false_sims: list[float] = []
    true_rows: list[tuple[float, str, str]] = []
    false_rows: list[tuple[float, str, str]] = []

    for child in children:
        for root in roots:
            s = _cos(vec_by_label[child["label"]], vec_by_label[root["label"]])
            if root["label"] == child["parent"]:
                true_sims.append(s)
                true_rows.append((s, child["label"], root["label"]))
            else:
                false_sims.append(s)
                false_rows.append((s, child["label"], root["label"]))

    print("-- true child→parent pairs --")
    for s, child, parent in sorted(true_rows):
        print(f"cos={s:.3f}  {child} → {parent}")

    print("\n-- worst false child→wrong-root pairs (top 5) --")
    for s, child, root in sorted(false_rows, reverse=True)[:5]:
        print(f"cos={s:.3f}  {child} → {root}  (WRONG)")

    print()
    print(_stats("TRUE ", true_sims))
    print(_stats("FALSE", false_sims))

    true_min, false_max = min(true_sims), max(false_sims)
    overlapping = not (true_min > false_max)
    print(
        f"\noverlap: true_min={true_min:.3f} vs false_max={false_max:.3f} -> "
        f"{'OVERLAPPING' if overlapping else 'SEPARABLE'}"
    )

    # Balanced accuracy = mean of true-positive and true-negative rate. Plain
    # accuracy would be misleading here: negatives outnumber positives 4:1, so a
    # threshold that rejects everything would still score 80%.
    print("\n thresh  miss_true  wrong_mount  balanced_acc")
    sweep = []
    for thresh in np.arange(0.200, 0.6001, 0.025):
        missed = sum(1 for s in true_sims if s < thresh)
        wrong = sum(1 for s in false_sims if s >= thresh)
        tpr = (len(true_sims) - missed) / len(true_sims)
        tnr = (len(false_sims) - wrong) / len(false_sims)
        bal = (tpr + tnr) / 2
        sweep.append((bal, float(thresh), missed, wrong))
        print(f"  {thresh:.3f}  {missed:>9}  {wrong:>11}  {bal:.3f}")

    best_bal, best_thresh, best_missed, best_wrong = max(
        sweep, key=lambda row: (row[0], -row[1])
    )
    cur_missed = sum(1 for s in true_sims if s < CURRENT_DEFAULT)
    cur_wrong = sum(1 for s in false_sims if s >= CURRENT_DEFAULT)

    print(
        f"\nbest balanced_acc={best_bal:.3f} at threshold={best_thresh:.3f} "
        f"(misses {best_missed}/{len(true_sims)} true, "
        f"wrongly mounts {best_wrong}/{len(false_sims)} false)"
    )
    print(
        f"current default mount_threshold={CURRENT_DEFAULT} "
        f"misses {cur_missed}/{len(true_sims)} true pairs, "
        f"wrongly mounts {cur_wrong}/{len(false_sims)} false"
    )

    # evolution.py:61-68 does argmax over roots FIRST, then applies the
    # threshold. So the cutoff only decides whether to mount at all; which
    # parent gets picked is decided by ranking. That makes argmax accuracy the
    # threshold-independent ceiling, and the wrong_mount column above an
    # overstatement of real error (a negative above the cutoff is only actually
    # mounted if it also beats the true parent).
    argmax_correct = 0
    argmax_misses = []
    for child in children:
        ranked = sorted(
            roots,
            key=lambda r: _cos(
                vec_by_label[child["label"]], vec_by_label[r["label"]]
            ),
            reverse=True,
        )
        if ranked[0]["label"] == child["parent"]:
            argmax_correct += 1
        else:
            argmax_misses.append(
                f"{child['label']}: picked {ranked[0]['label']}, "
                f"true {child['parent']}"
            )

    print(
        f"\nargmax-over-roots (ignoring any threshold): "
        f"{argmax_correct}/{len(children)} children rank their true parent #1"
    )
    for miss in argmax_misses:
        print(f"  wrong: {miss}")

    if overlapping:
        print(
            f"\n→ The two distributions OVERLAP: some children are more similar "
            f"to a wrong root ({false_max:.3f}) than others are to their real "
            f"parent ({true_min:.3f}). NO single threshold separates them — the "
            f"best-balanced-accuracy value above is a least-bad compromise, not "
            f"a fix. Treat this as a signal that the criterion itself needs to "
            f"change (relative ranking instead of an absolute cutoff, richer "
            f"embedding text, or LLM adjudication with embeddings as prefilter) "
            f"rather than as a number to retune."
        )
    else:
        print(
            f"\n→ Clean separation in [{false_max:.3f}, {true_min:.3f}]. A cutoff "
            f"inside that interval classifies every measured pair correctly."
        )
    print("\n(measurement only — this script changes no threshold)")


if __name__ == "__main__":
    main()
