"""Image loading, saving, and importance-map helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from image2seq.types import RenderResult


def rgb_to_gray(rgb: np.ndarray) -> np.ndarray:
    if rgb.ndim == 2:
        return rgb.astype(np.uint8)
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)


def load_grayscale_image(path: str | Path, size: int) -> np.ndarray:
    """Load and resize an image to (size, size) grayscale."""
    image = Image.open(path).convert("L")
    image = image.resize((size, size), Image.Resampling.LANCZOS)
    return np.array(image, dtype=np.uint8)


def make_importance_map(target: np.ndarray, edge_weight: float) -> np.ndarray:
    """Edge-weighted importance map for coverage error."""
    if edge_weight <= 0:
        return np.ones(target.shape, dtype=np.float64)

    grad_y = np.zeros_like(target, dtype=np.float64)
    grad_x = np.zeros_like(target, dtype=np.float64)
    grad_y[1:-1, :] = np.abs(target[2:, :].astype(np.float64) - target[:-2, :])
    grad_x[:, 1:-1] = np.abs(target[:, 2:].astype(np.float64) - target[:, :-2])
    grad = grad_y + grad_x
    max_grad = float(grad.max())
    if max_grad > 0:
        grad /= max_grad
    return 1.0 + edge_weight * grad


def save_result(result: RenderResult, output_dir: str | Path) -> None:
    """Save sequence, coverage map, and rendered preview."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / "sequence.npy", result.sequence)
    np.save(output_dir / "coverage.npy", result.coverage)

    Image.fromarray(result.render_image).save(output_dir / "render.png")
    Image.fromarray(result.target_image).save(output_dir / "target.png")

    with open(output_dir / "summary.txt", "w", encoding="utf-8") as fh:
        fh.write(f"iterations: {result.iterations}\n")
        fh.write(f"final_error: {result.final_error:.6f}\n")
        fh.write(f"sequence_length: {len(result.sequence)}\n")
