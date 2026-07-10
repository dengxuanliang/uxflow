"""Module 0.5 — label system self-evolution."""
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 dengxuanliang

from module0_5.models import LabelProposal, BackfillResult
from module0_5.evolution import resolve_proposal, ProposalResolution, ingest_proposal
from module0_5.inheritance import rerank_with_inheritance
from module0_5.backfill import run_backfill

__all__ = ["LabelProposal", "BackfillResult", "resolve_proposal", "ProposalResolution", "ingest_proposal", "rerank_with_inheritance", "run_backfill"]
