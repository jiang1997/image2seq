"""Line rasterization utilities."""

from __future__ import annotations

import numpy as np

from image2seq.types import Point


def bresenham_line(p1: Point, p2: Point) -> list[Point]:
    """Return integer (row, col) pixels on the line from p1 to p2."""
    y0, x0 = p1
    y1, x1 = p2
    points: list[Point] = []

    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy

    while True:
        points.append((y0, x0))
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy

    return points


def rasterize_line_indices(
    p1: Point,
    p2: Point,
    height: int,
    width: int,
    line_width: int = 1,
) -> np.ndarray:
    """
    Return flat indices for an opaque line without allocating a full mask.

    Coordinates are (row, col) in the same space as height/width.
    """
    centerline = bresenham_line(p1, p2)
    if not centerline:
        return np.empty(0, dtype=np.int32)

    if line_width <= 1:
        indices = [
            y * width + x
            for y, x in centerline
            if 0 <= y < height and 0 <= x < width
        ]
        return np.asarray(indices, dtype=np.int32)

    radius = max(0, line_width // 2)
    seen: set[int] = set()
    for y, x in centerline:
        y0 = max(0, y - radius)
        y1 = min(height, y + radius + 1)
        x0 = max(0, x - radius)
        x1 = min(width, x + radius + 1)
        for yy in range(y0, y1):
            base = yy * width
            for xx in range(x0, x1):
                seen.add(base + xx)

    return np.fromiter(seen, dtype=np.int32, count=len(seen))
