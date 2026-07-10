"""Module 0.5 — label system self-evolution."""
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 dengxuanliang

from module0_5.models import LabelProposal, BackfillResult
from module0_5.evolution import resolve_proposal, ProposalResolution

__all__ = ["LabelProposal", "BackfillResult", "resolve_proposal", "ProposalResolution"]
