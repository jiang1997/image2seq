"""Abstract interface between optimizers and physical simulators."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from image2seq.types import Point


class LineSimulator(ABC):
    """Physical model for opaque line placement and coverage evaluation."""

    @abstractmethod
    def reset(self) -> None:
        """Clear all placed lines and restore an empty canvas."""

    @abstractmethod
    def evaluate_line(self, p1: Point, p2: Point) -> float:
        """Return expected error reduction if line (p1, p2) is applied (>0 is better)."""

    @abstractmethod
    def apply_line(self, p1: Point, p2: Point) -> int:
        """Apply an opaque line. Returns count of newly covered sub-pixels."""

    @abstractmethod
    def total_error(self) -> float:
        """Current mismatch between target coverage and simulated coverage."""

    @abstractmethod
    def coverage_map(self) -> np.ndarray:
        """Low-resolution per-block coverage in [0, 1]."""

    @abstractmethod
    def render_image(self) -> np.ndarray:
        """Low-resolution grayscale preview in [0, 255]."""

    @abstractmethod
    def has_line(self, p1: Point, p2: Point) -> bool:
        """Whether a precomputed mask exists for the segment."""

    @abstractmethod
    def snapshot(self) -> tuple:
        """Capture mutable simulator state for branching search."""

    @abstractmethod
    def restore(self, snapshot: tuple) -> None:
        """Restore state previously returned by snapshot()."""
