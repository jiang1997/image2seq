"""Beam search optimizer over anchor-to-anchor segments."""

from __future__ import annotations

import numpy as np

from image2seq.optimizer.candidates import sample_candidates
from image2seq.optimizer.protocol import Optimizer
from image2seq.physics.protocol import LineSimulator
from image2seq.types import Point, RenderConfig, RenderResult


class BeamSearchOptimizer(Optimizer):
    """Maintain top partial paths ranked by total coverage error."""

    def __init__(
        self,
        simulator: LineSimulator,
        anchors: list[Point],
        rng: np.random.Generator | None = None,
    ) -> None:
        self.simulator = simulator
        self.anchors = anchors
        self.rng = rng if rng is not None else np.random.default_rng()
        self.start_anchor: Point | None = None

    def run(self, config: RenderConfig) -> RenderResult:
        self.simulator.reset()
        rng = (
            np.random.default_rng(config.random_seed)
            if config.random_seed is not None
            else self.rng
        )

        start = self.start_anchor if self.start_anchor is not None else self.anchors[0]
        initial_snapshot = self.simulator.snapshot()
        beams: list[tuple[list[Point], tuple, float]] = [
            ([start], initial_snapshot, self.simulator.total_error())
        ]
        best_sequence = [start]
        best_snapshot = initial_snapshot
        best_error = beams[0][2]

        target_image = getattr(self.simulator, "target_image", None)
        if target_image is None:
            target_image = self.simulator.render_image()

        for step in range(config.max_iterations):
            expansions: list[tuple[list[Point], tuple, float]] = []

            any_extended = False

            for sequence, snapshot, error in beams:
                self.simulator.restore(snapshot)
                current = sequence[-1]
                lines_placed = len(sequence) - 1
                candidates = sample_candidates(
                    current,
                    self.anchors,
                    config.candidate_pool_size,
                    rng,
                )

                branch_extended = False
                for candidate in candidates:
                    if not self.simulator.has_line(current, candidate):
                        continue
                    improvement = self.simulator.evaluate_line(current, candidate)
                    if improvement <= 0.0:
                        continue
                    if (
                        improvement <= config.min_improvement
                        and lines_placed >= config.min_lines
                    ):
                        continue

                    self.simulator.apply_line(current, candidate)
                    child_error = self.simulator.total_error()
                    expansions.append(
                        (sequence + [candidate], self.simulator.snapshot(), child_error)
                    )
                    branch_extended = True
                    any_extended = True
                    self.simulator.restore(snapshot)

                if not branch_extended:
                    expansions.append((sequence, snapshot, error))

            if not any_extended:
                break

            if not expansions:
                break

            expansions.sort(key=lambda item: item[2])
            beams = expansions[: config.beam_width]

            if beams[0][2] < best_error:
                best_sequence, best_snapshot, best_error = beams[0]

            iterations = step + 1
            if config.progress_interval > 0 and iterations % config.progress_interval == 0:
                print(f"step {iterations}: error={best_error:.2f}")

        self.simulator.restore(best_snapshot)

        return RenderResult(
            sequence=np.array(best_sequence, dtype=np.int32),
            coverage=self.simulator.coverage_map(),
            render_image=self.simulator.render_image(),
            target_image=np.asarray(target_image, dtype=np.uint8),
            iterations=len(best_sequence) - 1,
            final_error=best_error,
            anchors=self.anchors,
        )
