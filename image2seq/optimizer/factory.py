"""Optimizer factory for strategy selection."""

from __future__ import annotations

import numpy as np

from image2seq.optimizer.beam import BeamSearchOptimizer
from image2seq.optimizer.greedy import GreedyOptimizer
from image2seq.optimizer.multi_start import MultiStartOptimizer
from image2seq.optimizer.protocol import Optimizer
from image2seq.physics.protocol import LineSimulator
from image2seq.types import Point, RenderConfig


def create_optimizer(
    simulator: LineSimulator,
    anchors: list[Point],
    config: RenderConfig,
) -> Optimizer:
    """Build the configured optimizer strategy, optionally wrapped for multi-start."""
    rng = np.random.default_rng(config.random_seed)

    if config.strategy == "greedy":
        core: Optimizer = GreedyOptimizer(simulator, anchors)
    elif config.strategy == "beam":
        core = BeamSearchOptimizer(simulator, anchors, rng=rng)
    else:
        raise ValueError(f"unknown strategy: {config.strategy}")

    if config.num_starts > 1:
        return MultiStartOptimizer(core, anchors, config.num_starts)
    return core
