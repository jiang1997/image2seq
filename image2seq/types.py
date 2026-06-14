"""Shared types for image2seq."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

import numpy as np

Point = Tuple[int, int]


@dataclass
class RenderConfig:
    """Configuration for block-coverage string art rendering."""

    center: Point = (100, 100)
    radius: int = 100
    num_anchors: int = 180
    block_size: int = 8
    line_width: int = 2
    max_iterations: int = 5000
    metric: str = "l2"
    importance_weight: float = 0.0
    progress_interval: int = 500
    strategy: str = "beam"
    beam_width: int = 8
    num_starts: int = 4
    candidate_pool_size: int = 60
    min_improvement: float = 0.01
    min_lines: int = 0
    random_seed: int | None = 42


@dataclass
class RenderResult:
    """Output of a rendering run."""

    sequence: np.ndarray
    coverage: np.ndarray
    render_image: np.ndarray
    target_image: np.ndarray
    iterations: int
    final_error: float
    anchors: list[Point] = field(default_factory=list)
