"""Candidate anchor sampling for optimizers."""

from __future__ import annotations

import numpy as np

from image2seq.types import Point


def sample_candidates(
    current: Point,
    anchors: list[Point],
    pool_size: int,
    rng: np.random.Generator,
) -> list[Point]:
    """Return a subset of anchors to evaluate from the current position."""
    others = [anchor for anchor in anchors if anchor != current]
    if pool_size <= 0 or pool_size >= len(others):
        return others

    indices = rng.choice(len(others), size=pool_size, replace=False)
    return [others[int(i)] for i in indices]
