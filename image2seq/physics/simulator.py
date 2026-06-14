"""Block-coverage simulator with opaque line union."""

from __future__ import annotations

import numpy as np

from image2seq.physics.lines import rasterize_line_indices
from image2seq.physics.protocol import LineSimulator
from image2seq.types import Point


class BlockCoverageSimulator(LineSimulator):
    """
    Physical model for string art using opaque thread coverage.

    Tone is represented by the fraction of covered sub-pixels inside each
    low-resolution block, not by stacking gray values on the same pixel.
    """

    def __init__(
        self,
        target_image: np.ndarray,
        anchors: list[Point],
        block_size: int = 8,
        line_width: int = 2,
        metric: str = "l2",
        importance_map: np.ndarray | None = None,
    ) -> None:
        if target_image.ndim != 2:
            raise ValueError("target_image must be a 2D grayscale array")

        self.block_size = block_size
        self.line_width = line_width
        self.metric = metric

        self.target_image = target_image.astype(np.float64)
        self.low_h, self.low_w = self.target_image.shape
        self.high_h = self.low_h * block_size
        self.high_w = self.low_w * block_size
        self.block_area = float(block_size * block_size)

        self.target_cov = 1.0 - self.target_image / 255.0

        if importance_map is None:
            self.weights = np.ones((self.low_h, self.low_w), dtype=np.float64)
        else:
            self.weights = importance_map.astype(np.float64)

        self.anchors = list(anchors)
        self._line_cache: dict[tuple[Point, Point], np.ndarray] = {}

        self.covered = np.zeros(self.high_h * self.high_w, dtype=bool)
        self.block_counts = np.zeros((self.low_h, self.low_w), dtype=np.int32)

    def _to_high_res(self, point: Point) -> Point:
        scale = self.block_size
        return (point[0] * scale + scale // 2, point[1] * scale + scale // 2)

    def _mask_key(self, p1: Point, p2: Point) -> tuple[Point, Point]:
        return (self._to_high_res(p1), self._to_high_res(p2))

    def _build_line_indices(self, p1: Point, p2: Point) -> np.ndarray:
        key = self._mask_key(p1, p2)
        cached = self._line_cache.get(key)
        if cached is not None:
            return cached

        indices = rasterize_line_indices(
            key[0],
            key[1],
            self.high_h,
            self.high_w,
            self.line_width,
        )
        self._line_cache[key] = indices
        return indices

    def _lookup_indices(self, p1: Point, p2: Point) -> np.ndarray | None:
        if p1 == p2:
            return None
        return self._build_line_indices(p1, p2)

    def has_line(self, p1: Point, p2: Point) -> bool:
        return p1 != p2

    def reset(self) -> None:
        self.covered.fill(False)
        self.block_counts.fill(0)

    def snapshot(self) -> tuple:
        return (self.covered.copy(), self.block_counts.copy())

    def restore(self, snapshot: tuple) -> None:
        covered, block_counts = snapshot
        self.covered[:] = covered
        self.block_counts[:] = block_counts

    def _block_error(self, block_y: int, block_x: int, coverage: float) -> float:
        target = self.target_cov[block_y, block_x]
        weight = self.weights[block_y, block_x]
        diff = target - coverage
        if self.metric == "l1":
            return weight * abs(diff)
        return weight * diff * diff

    def _coverage_from_count(self, count: int) -> float:
        return min(1.0, count / self.block_area)

    def _new_pixels_per_block(self, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        uncovered = indices[~self.covered[indices]]
        if uncovered.size == 0:
            return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32)

        ys = uncovered // self.high_w
        xs = uncovered % self.high_w
        block_ids = (ys // self.block_size) * self.low_w + (xs // self.block_size)
        return np.unique(block_ids, return_counts=True)

    def _evaluate_indices(self, indices: np.ndarray) -> float:
        block_ids, counts = self._new_pixels_per_block(indices)
        if block_ids.size == 0:
            return 0.0

        improvement = 0.0
        for block_id, n_new in zip(block_ids, counts):
            by = int(block_id // self.low_w)
            bx = int(block_id % self.low_w)
            old_cov = self._coverage_from_count(int(self.block_counts[by, bx]))
            new_cov = self._coverage_from_count(int(self.block_counts[by, bx]) + int(n_new))
            improvement += self._block_error(by, bx, old_cov) - self._block_error(
                by, bx, new_cov
            )
        return improvement

    def evaluate_line(self, p1: Point, p2: Point) -> float:
        indices = self._lookup_indices(p1, p2)
        if indices is None or indices.size == 0:
            return 0.0
        return self._evaluate_indices(indices)

    def apply_line(self, p1: Point, p2: Point) -> int:
        indices = self._lookup_indices(p1, p2)
        if indices is None or indices.size == 0:
            return 0

        uncovered = indices[~self.covered[indices]]
        if uncovered.size == 0:
            return 0

        self.covered[uncovered] = True
        ys = uncovered // self.high_w
        xs = uncovered % self.high_w
        block_ids = (ys // self.block_size) * self.low_w + (xs // self.block_size)
        unique_ids, counts = np.unique(block_ids, return_counts=True)
        for block_id, count in zip(unique_ids, counts):
            by = int(block_id // self.low_w)
            bx = int(block_id % self.low_w)
            self.block_counts[by, bx] += int(count)

        return int(uncovered.size)

    def total_error(self) -> float:
        coverage = self.block_counts.astype(np.float64) / self.block_area
        diff = self.target_cov - coverage
        if self.metric == "l1":
            return float(np.sum(self.weights * np.abs(diff)))
        return float(np.sum(self.weights * diff * diff))

    def coverage_map(self) -> np.ndarray:
        return self.block_counts.astype(np.float64) / self.block_area

    def render_image(self) -> np.ndarray:
        coverage = self.coverage_map()
        return np.clip((1.0 - coverage) * 255.0, 0, 255).astype(np.uint8)
