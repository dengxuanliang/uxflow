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
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 dengxuanliang

from __future__ import annotations

from module0.parsing import (
    ParseError,
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

__all__ = ["QueryCompiler", "CompileError"]

_PASS_THRESHOLD = 0.8  # spec Part 6 分流规则


class CompileError(Exception):
    """Raised when compilation cannot proceed (Call 1 failed after retries)."""


class RetryableParseError(ParseError):
    """Internal signal: this step should be retried (empty response)."""


class _StepFailed(Exception):
    """Internal: a step exhausted its retries. Caught by compile() to degrade."""

    def __init__(self, step_name: str, last_error: Exception | None):
        self.step_name = step_name
        self.last_error = last_error
        super().__init__(f"{step_name} failed after retries: {last_error}")


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
        # Side-channel: HyDE query-anchor embeddings per sub-problem id.
        # Not part of the Problem Spec (contract §4 — computed query anchors,
        # consumed by the downstream retrieval layer / Module 1).
        self.hyde_embeddings: dict[str, list[list[float]]] = {}
        self._robustness = {
            "retries": {"call1": 0, "call2": 0, "call3": 0, "call2prime": 0},
            "degraded": [],
        }

    @property
    def robustness_report(self) -> dict:
        """Retry/degrade stats for the last compile() run."""
        return self._robustness

    async def compile(self, raw_input: str) -> ProblemSpec:
        """Run the full compilation pipeline. At most 4 LLM calls."""
        self.dropped_records = []
        self.hyde_embeddings = {}
        self._robustness = {
            "retries": {"call1": 0, "call2": 0, "call3": 0, "call2prime": 0},
            "degraded": [],
        }

        # ── Call 1: decompose ──
        try:
            sub_problems_raw = await self._call_and_parse(
                lambda: build_call1_messages(raw_input),
                parse_call1_response, step_name="call1",
            )
        except _StepFailed as e:
            raise CompileError("Call 1 failed after retries") from e

        # ── Call 2: label + self-eval ──
        try:
            c2_results = await self._call_and_parse(
                lambda: build_call2_messages(sub_problems_raw, self._taxonomy),
                parse_call2_response, step_name="call2",
            )
        except _StepFailed:
            c2_results = []
            self._robustness["degraded"].append("call2")

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
                # Per-item isolation: a schema-invalid pass item (e.g. LLM
                # returned 4 labels or 1 hyde segment) degrades to a drop
                # instead of crashing the whole batch.
                try:
                    passed.append(self._build_sub_problem(result, raw, origin="original", parent_id=None))
                except ValueError:
                    self._record_dropped(result, raw, "other")
            else:
                reason = result.get("drop_reason", "other")
                self._record_dropped(result, raw, reason)
                if reason == "ambiguous":
                    ambiguous.append({
                        "id": sp_id,
                        "raw_text": raw.get("raw_text", ""),
                        "failure_summary": raw.get("failure_summary", ""),
                        "target_capability": result.get("target_capability", []),
                    })

        # ── Call 3 + Call 2' (conditional, at most once, no recursion) ──
        if ambiguous:
            try:
                passed.extend(await self._clarify_and_reeval(ambiguous))
            except _StepFailed:
                self._robustness["degraded"].append("clarify")
                # Clarify path failed: ambiguous items stay in dropped_records,
                # main flow continues (no recover, no crash).

        # ── Embedding (optional) ──
        if self._embedding_model is not None:
            for sp in passed:
                vecs = self._embedding_model.embed_batch(sp.hyde_positive)
                self.hyde_embeddings[sp.id] = vecs

        return ProblemSpec(raw_input=raw_input, domain="agentic_swe", sub_problems=passed)

    async def _call_and_parse(
        self,
        build_messages_fn,
        parser,
        *,
        step_name: str,
        max_attempts: int = 2,
    ):
        """Call the LLM and parse; retry once on empty/malformed response.

        Raises _StepFailed when all attempts are exhausted; the caller decides
        how to degrade. Parsing stays strict — this only adds retry/None-guard.
        """
        last_err = None
        for attempt in range(max_attempts):
            content, _ = await self._gateway.call(
                build_messages_fn(), self._model, max_tokens=self._max_tokens
            )
            try:
                if not content or not content.strip():
                    raise RetryableParseError(f"{step_name}: empty response")
                return parser(content)
            except ParseError as e:  # RetryableParseError is a subclass
                last_err = e
                if attempt + 1 < max_attempts:
                    self._robustness["retries"][step_name] += 1
                continue
        raise _StepFailed(step_name, last_err)

    async def _clarify_and_reeval(self, ambiguous: list[dict]) -> list[SubProblem]:
        """Call 3 (disambiguate) + Call 2' (re-eval). At most once, no recursion."""
        # ── Call 3 ──
        c3_results = await self._call_and_parse(
            lambda: build_call3_messages(ambiguous),
            parse_call3_response, step_name="call3",
        )

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
        c2p_results = await self._call_and_parse(
            lambda: build_call2_prime_messages(clarified_raw, self._taxonomy),
            parse_call2_response, step_name="call2prime",
        )

        clarified_by_id = {c["id"]: c for c in clarified_raw}
        recovered: list[SubProblem] = []
        for result in c2p_results:
            sp_id = result["id"]
            raw = clarified_by_id.get(sp_id, {})
            if result.get("confidence", 0) >= _PASS_THRESHOLD:
                try:
                    recovered.append(self._build_sub_problem(
                        result, raw, origin="clarified", parent_id=parent_map.get(sp_id),
                    ))
                except ValueError:
                    pass  # schema-invalid clarified item silently dropped
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
        # Dropped items may carry invalid enums (that's often why they were
        # dropped). Use lenient filters so recording an audit trail never fails.
        try:
            filters = self._build_filters(result.get("structured_filters"))
        except ValueError:
            filters = StructuredFilters()
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
            structured_filters=filters,
            confidence=result.get("confidence", 0.0),
            route="drop",
            drop_reason=reason,
        )

    def _record_dropped(self, result: dict, raw: dict, reason: str) -> None:
        """Append a dropped record, tolerating schema-invalid input.

        DroppedSubProblem validation is minimal (only route + drop_reason), but
        an out-of-range drop_reason from the LLM would still raise. Fall back to
        'other' so a bad reason never crashes the compile.
        """
        try:
            self.dropped_records.append(self._build_dropped(result, raw, reason))
        except ValueError:
            self.dropped_records.append(self._build_dropped(result, raw, "other"))

    def _build_filters(self, d: dict | None) -> StructuredFilters:
        if not d:
            return StructuredFilters()
        return StructuredFilters(
            languages=d.get("languages"),
            tools_used=d.get("tools_used"),
            min_turns=d.get("min_turns"),
            has_verification_step=d.get("has_verification_step"),
        )
