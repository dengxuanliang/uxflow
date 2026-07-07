"""QueryCompiler — orchestrates Call 1/2/3/2' into a Problem Spec.

Flow (spec §3.2):
  Call 1 (decompose) → Call 2 (label + self-eval)
    ├── route==pass → Problem Spec
    ├── route==drop, reason!=ambiguous → dropped (audit)
    └── route==drop, reason==ambiguous → Call 3 (clarify) → Call 2' (re-eval)
          ├── confidence >= 0.8 → Problem Spec (origin=clarified)
          └── confidence < 0.8 → dropped (final)

Hard invariant: at most 4 LLM calls, Call 3+2' at most once, no recursion.
"""

from __future__ import annotations

from module0.parsing import (
    parse_call1_response,
    parse_call2_response,
    parse_call3_response,
)
from module0.prompts import (
    build_call1_messages,
    build_call2_messages,
    build_call3_messages,
    build_call2_prime_messages,
)
from module0.schema import (
    ProblemSpec,
    SubProblem,
    DroppedSubProblem,
    StructuredFilters,
)
from module0.taxonomy import Taxonomy

__all__ = ["QueryCompiler"]

_PASS_THRESHOLD = 0.8  # spec Part 6 分流规则


class QueryCompiler:
    """Compile a natural language problem description into a Problem Spec."""

    def __init__(self, *, gateway, taxonomy: Taxonomy, model: str,
                 embedding_model=None, max_tokens: int = 4000):
        self._gateway = gateway
        self._taxonomy = taxonomy
        self._model = model
        self._embedding_model = embedding_model
        self._max_tokens = max_tokens
        self.dropped_records: list[DroppedSubProblem] = []

    async def compile(self, raw_input: str) -> ProblemSpec:
        """Run the full compilation pipeline. At most 4 LLM calls."""
        self.dropped_records = []

        # ── Call 1: decompose ──
        c1_text, _ = await self._gateway.call(
            build_call1_messages(raw_input), self._model, max_tokens=self._max_tokens,
        )
        sub_problems_raw = parse_call1_response(c1_text)

        # ── Call 2: label + self-eval ──
        c2_text, _ = await self._gateway.call(
            build_call2_messages(sub_problems_raw, self._taxonomy),
            self._model, max_tokens=self._max_tokens,
        )
        c2_results = parse_call2_response(c2_text)

        # Index raw sub-problems by id for merging
        raw_by_id = {sp["id"]: sp for sp in sub_problems_raw}

        passed: list[SubProblem] = []
        ambiguous: list[dict] = []

        for result in c2_results:
            sp_id = result["id"]
            raw = raw_by_id.get(sp_id, {})
            route = result.get("route")
            confidence = result.get("confidence", 0)

            if route == "pass" and confidence >= _PASS_THRESHOLD:
                passed.append(self._build_sub_problem(result, raw, origin="original", parent_id=None))
            else:
                reason = result.get("drop_reason", "other")
                self.dropped_records.append(self._build_dropped(result, raw, reason))
                if reason == "ambiguous":
                    ambiguous.append({
                        "id": sp_id,
                        "raw_text": raw.get("raw_text", ""),
                        "failure_summary": raw.get("failure_summary", ""),
                        "target_capability": result.get("target_capability", []),
                    })

        # ── Call 3 + Call 2' (conditional, at most once, no recursion) ──
        if ambiguous:
            passed.extend(await self._clarify_and_reeval(ambiguous))

        # ── Embedding (optional) ──
        if self._embedding_model is not None:
            for sp in passed:
                self._embedding_model.embed_batch(sp.hyde_positive)

        return ProblemSpec(raw_input=raw_input, domain="agentic_swe", sub_problems=passed)

    async def _clarify_and_reeval(self, ambiguous: list[dict]) -> list[SubProblem]:
        """Call 3 (disambiguate) + Call 2' (re-eval). At most once, no recursion."""
        # ── Call 3 ──
        c3_text, _ = await self._gateway.call(
            build_call3_messages(ambiguous), self._model, max_tokens=self._max_tokens,
        )
        c3_results = parse_call3_response(c3_text)

        # Flatten clarified sub-problems, track parent_id
        clarified_raw: list[dict] = []
        parent_map: dict[str, str] = {}
        for group in c3_results:
            original_id = group["original_id"]
            for c in group["clarified"]:
                clarified_raw.append(c)
                parent_map[c["id"]] = original_id

        if not clarified_raw:
            return []

        # ── Call 2' ──
        c2p_text, _ = await self._gateway.call(
            build_call2_prime_messages(clarified_raw, self._taxonomy),
            self._model, max_tokens=self._max_tokens,
        )
        c2p_results = parse_call2_response(c2p_text)

        clarified_by_id = {c["id"]: c for c in clarified_raw}
        recovered: list[SubProblem] = []
        for result in c2p_results:
            sp_id = result["id"]
            raw = clarified_by_id.get(sp_id, {})
            if result.get("confidence", 0) >= _PASS_THRESHOLD:
                recovered.append(self._build_sub_problem(
                    result, raw, origin="clarified", parent_id=parent_map.get(sp_id),
                ))
        return recovered

    def _build_sub_problem(self, result: dict, raw: dict, *, origin: str,
                           parent_id: str | None) -> SubProblem:
        return SubProblem(
            id=result["id"],
            origin=origin,
            parent_id=parent_id,
            raw_text=raw.get("raw_text", ""),
            failure_summary=raw.get("failure_summary", ""),
            target_capability=result["target_capability"],
            trajectory_signal=result["trajectory_signal"],
            hyde_positive=result["hyde_positive"],
            keywords=result["keywords"],
            structured_filters=self._build_filters(result.get("structured_filters")),
            confidence=result["confidence"],
            route="pass",
        )

    def _build_dropped(self, result: dict, raw: dict, reason: str) -> DroppedSubProblem:
        return DroppedSubProblem(
            id=result["id"],
            origin="original",
            parent_id=None,
            raw_text=raw.get("raw_text", ""),
            failure_summary=raw.get("failure_summary", ""),
            target_capability=result.get("target_capability", ["unknown"]),
            trajectory_signal=result.get("trajectory_signal", ""),
            hyde_positive=result.get("hyde_positive", ["", ""]),
            keywords=result.get("keywords", []),
            structured_filters=self._build_filters(result.get("structured_filters")),
            confidence=result.get("confidence", 0.0),
            route="drop",
            drop_reason=reason,
        )

    def _build_filters(self, d: dict | None) -> StructuredFilters:
        if not d:
            return StructuredFilters()
        return StructuredFilters(
            languages=d.get("languages"),
            tools_used=d.get("tools_used"),
            outcome_transition=d.get("outcome_transition"),
            min_turns=d.get("min_turns"),
            has_verification_step=d.get("has_verification_step"),
        )
