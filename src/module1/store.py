"""Storage abstraction shared by modules 2/3/0.5."""
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 dengxuanliang

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from module1.models import Slice, TrajectorySignature

__all__ = ["RecallHit", "SliceStore"]


@dataclass
class RecallHit:
    """A recalled signature plus its RRF fusion score."""

    signature: TrajectorySignature
    rrf_score: float


@runtime_checkable
class SliceStore(Protocol):
    """Read and mutable-writeback interface over slice signatures."""

    def add_batch(self, sigs: list[TrajectorySignature]) -> None: ...

    def recall(
        self,
        *,
        structured_filters: dict,
        keywords: list[str],
        query_embeddings: list[list[float]],
        top_n: int = 20,
    ) -> list[RecallHit]: ...

    def get_slice(self, trajectory_id: str, slice_index: int) -> Slice | None: ...

    def set_slice_source(
        self, trajectory_id: str, slice_index: int, slice_obj: Slice
    ) -> None: ...

    def update_labels(
        self, trajectory_id: str, slice_index: int, labels: list[str]
    ) -> None: ...
