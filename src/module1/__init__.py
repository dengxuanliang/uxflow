"""Module 1: Trajectory Processing Pipeline.

Processes backflow trajectories to find positive capability demonstrations
and output loss mask spans for SFT training.
"""
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 dengxuanliang

from module1.models import (
    Step,
    Trajectory,
    Slice,
    TrajectorySignature,
    JudgeResult,
    SFTCandidate,
)
from module1.loader import load_trajectories, parse_trajectory
from module1.slicer import slice_trajectory
from module1.signature import extract_signature
from module1.index import MemoryIndex
from module1.summarizer import summarize_slice
from module1.judge import Judge
from module1.pipeline import TrajectoryPipeline, PipelineConfig

__all__ = [
    # Models
    "Step",
    "Trajectory",
    "Slice",
    "TrajectorySignature",
    "JudgeResult",
    "SFTCandidate",
    # Components
    "load_trajectories",
    "parse_trajectory",
    "slice_trajectory",
    "extract_signature",
    "MemoryIndex",
    "summarize_slice",
    "Judge",
    # Pipeline
    "TrajectoryPipeline",
    "PipelineConfig",
]
