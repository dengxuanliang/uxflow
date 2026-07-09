"""Module 3 dataset composition and general-data hook."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

__all__ = ["GeneralDataConfig", "compose_dataset"]

_log = logging.getLogger(__name__)


@dataclass
class GeneralDataConfig:
    ratio: float = 0.25
    source_path: str | None = None


def compose_dataset(
    targeted: list[Any],
    *,
    general_config: GeneralDataConfig,
) -> dict:
    """Return targeted data plus optional general data and a manifest."""
    general: list[Any] = []
    if general_config.source_path is None:
        _log.info(
            "No general data source configured (ratio=%.2f intended); "
            "returning targeted slices only.",
            general_config.ratio,
        )
    else:
        _log.warning(
            "general_config.source_path set (%s) but general-data loading is "
            "not implemented in this batch; skipping.",
            general_config.source_path,
        )

    return {
        "targeted": targeted,
        "general": general,
        "manifest": {
            "targeted_count": len(targeted),
            "general_count": len(general),
            "general_ratio": general_config.ratio,
        },
    }
