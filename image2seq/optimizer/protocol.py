"""Abstract interface for string-art optimizers."""

from __future__ import annotations

from abc import ABC, abstractmethod

from image2seq.types import RenderConfig, RenderResult


class Optimizer(ABC):
    """Search strategy that produces a rendered line sequence."""

    @abstractmethod
    def run(self, config: RenderConfig) -> RenderResult:
        """Execute the optimizer and return the best render result."""
