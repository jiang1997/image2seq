# image2seq

**English** | [中文](README.zh-CN.md)

Approximate a grayscale image as string art by drawing chords between anchor points on a circle.

<img src="portrait.jpg" alt="input" width="200"/> <img src="portrait_string_art.jpg" alt="string art result" width="200"/>

## Overview

1. Place anchors evenly on a circle around the canvas (180 by default)
2. From the current anchor, draw a chord to the next anchor
3. Repeat until done, producing an anchor visit sequence `sequence`

The physical model uses **block coverage with opaque union** (inspired by string art literature):

- Real black thread is **opaque**: overlapping lines do not get darker at the same pixel
- Tone comes from **local coverage**: the fraction of covered sub-pixels in each block approximates the target gray level
- High-resolution simulation (`block_size` scale) compared against a low-resolution target image

Optimization (default: **beam search + multi-start**):

- **Greedy**: pick the chord with the largest error reduction each step — fast
- **Beam**: keep the top-k partial paths to reduce greedy myopia
- **Multi-start**: run from several starting anchors and keep the best result

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python main.py portrait.jpg -o output
```

## CLI options

| Option | Default | Description |
|--------|---------|-------------|
| `input` | — | Input image path |
| `-o, --output` | `output` | Output directory |
| `--radius` | `100` | Anchor circle radius (image size = 2r+1) |
| `--anchors` | `180` | Number of anchors on the circle |
| `--block-size` | `8` | Coverage block scale factor |
| `--line-width` | `2` | Thread width in high-res pixels |
| `--max-iter` | `5000` | Maximum number of lines |
| `--metric` | `l2` | Coverage error metric: `l1` / `l2` |
| `--importance` | `3.0` | Edge importance weight (0 disables) |
| `--strategy` | `beam` | Optimizer: `greedy` / `beam` |
| `--beam-width` | `8` | Beam search width |
| `--num-starts` | `4` | Number of starting anchors |
| `--candidate-pool` | `60` | Candidates sampled per step (0 = all) |
| `--min-improvement` | `0.01` | Stop when improvement falls below this |
| `--min-lines` | `0` | Minimum lines before early stopping |
| `--seed` | `42` | Random seed |
| `--progress` | `500` | Progress print interval (0 disables) |

### Examples

```bash
# Quality-first (default settings)
python main.py portrait.jpg -o output --strategy beam --beam-width 6 --num-starts 2

# Speed-first
python main.py portrait.jpg -o output --strategy greedy --num-starts 1 --candidate-pool 0
```

## Output

| File | Description |
|------|-------------|
| `render.png` | Rendered preview |
| `target.png` | Resized target image |
| `sequence.npy` | Anchor sequence `(N, 2)` as `(row, col)` |
| `coverage.npy` | Low-resolution coverage map |
| `summary.txt` | Iteration count and final error |

## Project layout

```
image2seq/
├── main.py                 # CLI entry point
├── requirements.txt
└── image2seq/
    ├── physics/            # How lines cover the canvas
    │   ├── protocol.py     # LineSimulator interface
    │   ├── lines.py        # Bresenham rasterization
    │   └── simulator.py    # BlockCoverageSimulator
    ├── optimizer/          # Which line to draw next
    │   ├── greedy.py
    │   ├── beam.py
    │   ├── multi_start.py
    │   └── factory.py
    ├── types.py            # Config and result types
    └── io.py               # Image I/O and result saving
```

Physics and optimization are decoupled via the `LineSimulator` interface, so either layer can be swapped independently.

## References

- [Birsak et al. 2018 — String Art](https://doi.org/10.1111/cgf.13359)
- [Demoussel et al. 2022 — A Greedy Algorithm for Generative String Art](https://hal.science/hal-03901755)
