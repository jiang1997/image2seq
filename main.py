#!/usr/bin/env python3
"""CLI entry point for image2seq string art rendering."""

from __future__ import annotations

import argparse
from pathlib import Path

from image2seq.io import load_grayscale_image, make_importance_map, save_result
from image2seq.optimizer import circular_anchors, create_optimizer
from image2seq.physics import BlockCoverageSimulator
from image2seq.types import RenderConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate string art with block-coverage opaque simulation.",
    )
    parser.add_argument("input", type=Path, help="Input image path")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("output"),
        help="Output directory (default: output)",
    )
    parser.add_argument("--radius", type=int, default=100, help="Anchor circle radius")
    parser.add_argument("--anchors", type=int, default=180, help="Number of anchor points")
    parser.add_argument("--block-size", type=int, default=8, help="Coverage block scale")
    parser.add_argument("--line-width", type=int, default=2, help="Thread width in pixels")
    parser.add_argument("--max-iter", type=int, default=5000, help="Maximum line count")
    parser.add_argument(
        "--metric",
        choices=["l1", "l2"],
        default="l2",
        help="Coverage error metric",
    )
    parser.add_argument(
        "--importance",
        type=float,
        default=3.0,
        help="Edge importance weight (0 disables)",
    )
    parser.add_argument(
        "--progress",
        type=int,
        default=500,
        help="Progress print interval (0 disables)",
    )
    parser.add_argument(
        "--strategy",
        choices=["greedy", "beam"],
        default="beam",
        help="Optimizer strategy (default: beam)",
    )
    parser.add_argument(
        "--beam-width",
        type=int,
        default=8,
        help="Beam width for beam search (default: 8)",
    )
    parser.add_argument(
        "--num-starts",
        type=int,
        default=4,
        help="Number of starting anchors to try (default: 4)",
    )
    parser.add_argument(
        "--candidate-pool",
        type=int,
        default=60,
        help="Candidate anchors sampled per step (0 = all)",
    )
    parser.add_argument(
        "--min-improvement",
        type=float,
        default=0.01,
        help="Minimum error reduction to continue (default: 0.01)",
    )
    parser.add_argument(
        "--min-lines",
        type=int,
        default=0,
        help="Minimum lines before low-improvement stopping",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for candidate sampling (default: 42)",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    size = args.radius * 2 + 1

    config = RenderConfig(
        center=(args.radius, args.radius),
        radius=args.radius,
        num_anchors=args.anchors,
        block_size=args.block_size,
        line_width=args.line_width,
        max_iterations=args.max_iter,
        metric=args.metric,
        importance_weight=args.importance,
        progress_interval=args.progress,
        strategy=args.strategy,
        beam_width=args.beam_width,
        num_starts=args.num_starts,
        candidate_pool_size=args.candidate_pool,
        min_improvement=args.min_improvement,
        min_lines=args.min_lines,
        random_seed=args.seed,
    )

    target = load_grayscale_image(args.input, size)
    anchors = circular_anchors(config.center, config.radius, config.num_anchors)
    importance = make_importance_map(target, config.importance_weight)

    simulator = BlockCoverageSimulator(
        target_image=target,
        anchors=anchors,
        block_size=config.block_size,
        line_width=config.line_width,
        metric=config.metric,
        importance_map=importance,
    )

    optimizer = create_optimizer(simulator, anchors, config)
    result = optimizer.run(config)

    save_result(result, args.output)
    print(
        f"done: {result.iterations} lines, "
        f"error={result.final_error:.4f}, saved to {args.output}"
    )


if __name__ == "__main__":
    main()
