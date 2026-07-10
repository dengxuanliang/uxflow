"""Module 0.5 ④ — async targeted backfill (queue-schedulable execution body)."""
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 dengxuanliang

from __future__ import annotations

from module0.taxonomy import TaxonomyLabel
from module0_5.models import BackfillResult

__all__ = ["run_backfill"]


async def run_backfill(
    new_label: TaxonomyLabel,
    store,
    judge,
    *,
    top_k: int = 200,
    judge_batch_size: int = 30,
) -> BackfillResult:
    """4 步回填一个标签。幂等：已标该 label 的切片跳过。失败可返回不抛穿。

    Step 1 构造查询（零LLM）：label.keywords → BM25；description_embedding → 向量锚。
    Step 2 粗筛（零LLM）：store.recall(...) 取 top_k，跳过已标该 label 的。
    Step 3 精判（LLM）：judge.judge_batch 判该切片是否演示此能力。
    Step 4 写回：judged_true 的 → store.update_labels（合并语义，幂等）。

    judge_batch_size is a reserved contract seam; batching is currently owned
    by the injected Judge, so this param is not consumed in V1.
    """
    label = new_label.label
    errors: list[str] = []

    query_embeddings = [new_label.description_embedding] if new_label.description_embedding else []
    try:
        hits = store.recall(
            structured_filters={},
            keywords=new_label.keywords,
            query_embeddings=query_embeddings,
            top_n=top_k,
        )
        candidates, slices = [], []
        for hit in hits:
            sig = hit.signature
            if label in (sig.capability_labels or []):
                continue  # idempotence: skip already-labeled
            sl = store.get_slice(sig.trajectory_id, sig.slice_index)
            if sl is not None:
                candidates.append(sig)
                slices.append(sl)
    except Exception as e:
        return BackfillResult(label=label, candidates_screened=0,
                              judged_true=0, slices_written=0, errors=[f"recall: {e}"])

    if not slices:
        return BackfillResult(label=label, candidates_screened=0,
                              judged_true=0, slices_written=0, errors=errors)

    try:
        results = await judge.judge_batch(
            slices=slices,
            target_capability=[label],
            trajectory_signal=new_label.description,
        )
    except Exception as e:
        return BackfillResult(label=label, candidates_screened=len(slices),
                              judged_true=0, slices_written=0, errors=[f"judge: {e}"])

    judged_true = 0
    written = 0
    for sig, res in zip(candidates, results):
        if res.match:
            judged_true += 1
            try:
                store.update_labels(sig.trajectory_id, sig.slice_index, [label])
                written += 1
            except Exception as e:
                errors.append(f"writeback {sig.trajectory_id}:{sig.slice_index}: {e}")

    return BackfillResult(
        label=label, candidates_screened=len(slices),
        judged_true=judged_true, slices_written=written, errors=errors,
    )
