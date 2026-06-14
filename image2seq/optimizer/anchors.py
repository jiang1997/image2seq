"""Anchor point generation."""

from __future__ import annotations

import numpy as np

from image2seq.types import Point


def circular_anchors(
    center: Point,
    radius: int,
    num_points: int,
) -> list[Point]:
    """Place anchors evenly around a circle."""
    cy, cx = center
    anchors: list[Point] = []

    for i in range(num_points):
        angle = 2 * np.pi * i / num_points
        dy = int(round(np.cos(angle) * radius))
        dx = int(round(np.sin(angle) * radius))
        anchors.append((cy + dy, cx + dx))

    return anchors
