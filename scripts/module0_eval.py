"""Module 0 Prompt Evaluation Harness.

Runs the golden eval set through QueryCompiler N times, computes:
- Label accuracy (target_capability match)
- Route accuracy (pass/drop correct)
- Drop reason accuracy (when drop, reason correct)
- Drift rate (N runs same input, consistency %)

Usage:
    python scripts/module0_eval.py [--runs N] [--output results.json]
"""

import asyncio
import json
import os
import pathlib
import sys
import time
from collections import defaultdict

from dotenv import load_dotenv
load_dotenv(pathlib.Path(__file__).parent.parent / ".env")

from llm_gateway import LLMGateway, GatewayConfig
from module0 import QueryCompiler, Taxonomy
from module0.parsing import ParseError

FIXTURES_DIR = pathlib.Path(__file__).parent.parent / "fixtures"
GOLDEN_SET_PATH = FIXTURES_DIR / "golden_eval_set.json"
TAXONOMY_PATH = FIXTURES_DIR / "taxonomy_v0.json"


def load_golden_set():
    with open(GOLDEN_SET_PATH) as f:
        return json.load(f)


def score_single_run(actual_spec, actual_dropped, expected_items):
    """Score one run against expected items.

    Returns dict with per-sub-problem scores.
    """
    # Combine pass and drop results for matching
    actual_all = []
    for sp in actual_spec.sub_problems:
        actual_all.append({
            "target_capability": sp.target_capability,
            "route": "pass",
            "confidence": sp.confidence,
        })
    for d in actual_dropped:
        actual_all.append({
            "target_capability": d.target_capability,
            "route": "drop",
            "drop_reason": d.drop_reason,
        })

    scores = {
        "label_correct": 0,
        "label_total": 0,
        "route_correct": 0,
        "route_total": 0,
        "drop_reason_correct": 0,
        "drop_reason_total": 0,
        "sub_problem_count_match": len(actual_all) == len(expected_items),
    }

    # Match by position (best effort — LLM may reorder)
    # For route accuracy, check if counts match
    expected_pass = [e for e in expected_items if e.get("route") == "pass"]
    expected_drop = [e for e in expected_items if e.get("route") == "drop"]
    actual_pass = [a for a in actual_all if a["route"] == "pass"]
    actual_drop = [a for a in actual_all if a["route"] == "drop"]

    # Route accuracy: did we get the right number of pass vs drop?
    scores["route_total"] = len(expected_items)
    scores["route_correct"] = (
        min(len(actual_pass), len(expected_pass)) +
        min(len(actual_drop), len(expected_drop))
    )

    # Label accuracy: for each expected pass item, find best match in actual
    for exp in expected_pass:
        scores["label_total"] += 1
        exp_labels = set(exp.get("target_capability", []))
        if "__NEW__" in exp_labels:
            # Accept any taxonomy_extension label
            if actual_pass:
                scores["label_correct"] += 1
            continue
        # Find an actual pass whose labels overlap
        for actual in actual_pass:
            actual_labels = set(actual.get("target_capability", []))
            if exp_labels & actual_labels:  # intersection non-empty
                scores["label_correct"] += 1
                break

    # Drop reason accuracy
    for exp in expected_drop:
        if "drop_reason" in exp:
            scores["drop_reason_total"] += 1
            exp_reason = exp["drop_reason"]
            for actual in actual_drop:
                if actual.get("drop_reason") == exp_reason:
                    scores["drop_reason_correct"] += 1
                    break

    return scores


def compute_drift(run_results):
    """Compute drift: how consistent are multiple runs of same input.

    run_results: list of (spec, dropped) tuples for same input.
    Returns consistency score 0-1 (1 = all identical).
    """
    if len(run_results) <= 1:
        return 1.0

    # Compare by: route pattern + label sets
    signatures = []
    for spec, dropped in run_results:
        sig = []
        for sp in spec.sub_problems:
            sig.append(("pass", tuple(sorted(sp.target_capability))))
        for d in dropped:
            sig.append(("drop", d.drop_reason))
        signatures.append(tuple(sorted(sig)))

    # Consistency = fraction of runs matching the mode
    from collections import Counter
    counts = Counter(signatures)
    most_common_count = counts.most_common(1)[0][1]
    return most_common_count / len(signatures)


async def run_eval(num_runs=1, model=None):
    golden = load_golden_set()
    taxonomy = Taxonomy.load(TAXONOMY_PATH)
    model = model or os.environ.get("MODULE0_TEST_MODEL", "gpt-4o-mini")

    config = GatewayConfig(
        litellm_base=os.environ.get("LITELLM_BASE", "http://localhost:4000/v1"),
        litellm_key=os.environ.get("LITELLM_KEY", ""),
        concurrency=5,
        transport_stuck_seconds=0,
    )

    all_scores = defaultdict(list)
    all_drift_data = defaultdict(list)  # case_id -> list of (spec, dropped)
    errors = []

    async with LLMGateway(config) as gw:
        for run_idx in range(num_runs):
            print(f"\n{'='*60}")
            print(f"Run {run_idx + 1}/{num_runs}")
            print(f"{'='*60}")

            for case in golden["cases"]:
                case_id = case["id"]
                raw_input = case["raw_input"]
                expected = case["expected"]

                print(f"  {case_id}: \"{raw_input[:30]}...\"", end=" ", flush=True)
                start = time.time()

                try:
                    compiler = QueryCompiler(
                        gateway=gw, taxonomy=taxonomy, model=model, embedding_model=None
                    )
                    spec = await compiler.compile(raw_input)
                    dropped = compiler.dropped_records

                    scores = score_single_run(spec, dropped, expected)
                    all_scores[case_id].append(scores)
                    all_drift_data[case_id].append((spec, dropped))

                    elapsed = time.time() - start
                    label_ok = scores["label_correct"] == scores["label_total"]
                    route_ok = scores["route_correct"] == scores["route_total"]
                    status = "✓" if (label_ok and route_ok) else "✗"
                    print(f"{status} ({elapsed:.1f}s) labels={scores['label_correct']}/{scores['label_total']} route={scores['route_correct']}/{scores['route_total']}")

                except (ParseError, Exception) as e:
                    elapsed = time.time() - start
                    print(f"ERROR ({elapsed:.1f}s): {type(e).__name__}: {e}")
                    errors.append({"case_id": case_id, "run": run_idx, "error": str(e)})

    # Aggregate metrics
    print(f"\n{'='*60}")
    print("AGGREGATE RESULTS")
    print(f"{'='*60}")

    total_label_correct = 0
    total_label_total = 0
    total_route_correct = 0
    total_route_total = 0
    total_drop_correct = 0
    total_drop_total = 0

    for case_id, score_list in all_scores.items():
        for s in score_list:
            total_label_correct += s["label_correct"]
            total_label_total += s["label_total"]
            total_route_correct += s["route_correct"]
            total_route_total += s["route_total"]
            total_drop_correct += s["drop_reason_correct"]
            total_drop_total += s["drop_reason_total"]

    label_acc = total_label_correct / max(total_label_total, 1)
    route_acc = total_route_correct / max(total_route_total, 1)
    drop_acc = total_drop_correct / max(total_drop_total, 1)

    # Drift (only meaningful if num_runs > 1)
    drift_scores = []
    if num_runs > 1:
        for case_id, runs in all_drift_data.items():
            drift_scores.append(compute_drift(runs))
        avg_drift = sum(drift_scores) / len(drift_scores) if drift_scores else 1.0
    else:
        avg_drift = None

    results = {
        "num_runs": num_runs,
        "num_cases": len(golden["cases"]),
        "model": model,
        "metrics": {
            "label_accuracy": round(label_acc, 3),
            "route_accuracy": round(route_acc, 3),
            "drop_reason_accuracy": round(drop_acc, 3),
            "consistency_rate": round(avg_drift, 3) if avg_drift is not None else "N/A (single run)",
        },
        "totals": {
            "label_correct": total_label_correct,
            "label_total": total_label_total,
            "route_correct": total_route_correct,
            "route_total": total_route_total,
            "drop_reason_correct": total_drop_correct,
            "drop_reason_total": total_drop_total,
        },
        "errors": errors,
    }

    print(f"\n  Label accuracy:      {label_acc:.1%} ({total_label_correct}/{total_label_total})")
    print(f"  Route accuracy:      {route_acc:.1%} ({total_route_correct}/{total_route_total})")
    print(f"  Drop reason accuracy: {drop_acc:.1%} ({total_drop_correct}/{total_drop_total})")
    if avg_drift is not None:
        print(f"  Consistency rate:    {avg_drift:.1%}")
    if errors:
        print(f"  Parse errors:        {len(errors)}")
    print()

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Module 0 prompt evaluation")
    parser.add_argument("--runs", type=int, default=1, help="Number of runs per case (>1 for drift measurement)")
    parser.add_argument("--output", type=str, default=None, help="Save results to JSON file")
    parser.add_argument("--model", type=str, default=None, help="Override model name")
    args = parser.parse_args()

    results = asyncio.run(run_eval(num_runs=args.runs, model=args.model))

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
