"""Multi-start wrapper that tries several starting anchors."""

from __future__ import annotations

import numpy as np

from image2seq.optimizer.protocol import Optimizer
from image2seq.types import Point, RenderConfig, RenderResult


def select_start_anchors(
    anchors: list[Point],
    num_starts: int,
) -> list[Point]:
    """Pick evenly spaced anchors as starting positions."""
    if num_starts <= 1:
        return [anchors[0]]
    if num_starts >= len(anchors):
        return list(anchors)

    indices = np.linspace(0, len(anchors) - 1, num_starts, dtype=int)
    return [anchors[int(i)] for i in indices]


class MultiStartOptimizer(Optimizer):
    """Run an inner optimizer from multiple starts and keep the best result."""

    def __init__(
        self,
        inner: Optimizer,
        anchors: list[Point],
        num_starts: int,
    ) -> None:
        self.inner = inner
        self.anchors = anchors
        self.num_starts = num_starts

    def run(self, config: RenderConfig) -> RenderResult:
        starts = select_start_anchors(self.anchors, self.num_starts)
        best_result: RenderResult | None = None

        for index, start in enumerate(starts):
            print(f"start {index + 1}/{len(starts)}: anchor={start}")
            if hasattr(self.inner, "start_anchor"):
                self.inner.start_anchor = start
            result = self.inner.run(config)
            if best_result is None or result.final_error < best_result.final_error:
                best_result = result

        assert best_result is not None
        return best_result
