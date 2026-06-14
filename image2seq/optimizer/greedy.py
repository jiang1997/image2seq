"""Greedy string-art optimizer decoupled from the physical simulator."""

from __future__ import annotations

import numpy as np

from image2seq.optimizer.candidates import sample_candidates
from image2seq.optimizer.protocol import Optimizer
from image2seq.physics.protocol import LineSimulator
from image2seq.types import Point, RenderConfig, RenderResult


class GreedyOptimizer(Optimizer):
    """
    Sequential greedy search over anchor-to-anchor segments.

    The optimizer only depends on the LineSimulator interface, so the
    physical model can be swapped without changing this code.
    """

    def __init__(self, simulator: LineSimulator, anchors: list[Point]) -> None:
        self.simulator = simulator
        self.anchors = anchors
        self.start_anchor: Point | None = None

    def _best_next(
        self,
        current: Point,
        candidates: list[Point],
    ) -> tuple[Point, float]:
        best_point = current
        best_score = 0.0

        for candidate in candidates:
            if not self.simulator.has_line(current, candidate):
                continue
            score = self.simulator.evaluate_line(current, candidate)
            if score > best_score:
                best_score = score
                best_point = candidate

        return best_point, best_score

    def run(self, config: RenderConfig) -> RenderResult:
        self.simulator.reset()

        start = self.start_anchor if self.start_anchor is not None else self.anchors[0]
        current = start
        sequence: list[Point] = [current]
        iterations = 0
        rng = np.random.default_rng(config.random_seed)

        target_image = getattr(self.simulator, "target_image", None)
        if target_image is None:
            target_image = self.simulator.render_image()

        for step in range(config.max_iterations):
            candidates = sample_candidates(
                current,
                self.anchors,
                config.candidate_pool_size,
                rng,
            )
            next_point, improvement = self._best_next(current, candidates)
            lines_placed = len(sequence) - 1

            if next_point == current or improvement <= 0.0:
                break
            if improvement <= config.min_improvement and lines_placed >= config.min_lines:
                break

            self.simulator.apply_line(current, next_point)
            current = next_point
            sequence.append(current)
            iterations = step + 1

            if config.progress_interval > 0 and iterations % config.progress_interval == 0:
                err = self.simulator.total_error()
                print(f"step {iterations}: error={err:.2f}")

        return RenderResult(
            sequence=np.array(sequence, dtype=np.int32),
            coverage=self.simulator.coverage_map(),
            render_image=self.simulator.render_image(),
            target_image=np.asarray(target_image, dtype=np.uint8),
            iterations=iterations,
            final_error=self.simulator.total_error(),
            anchors=self.anchors,
        )
