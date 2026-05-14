from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from .renderer import RenderOptions, render_sketch_gif

# 解析命令行参数
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sketch-gen",
        description="Generate a progressive sketch-and-color GIF from an image.",
    )
    parser.add_argument("input_path", type=Path, help="Source image path.")
    parser.add_argument("output_path", type=Path, help="Destination .gif path.")
    parser.add_argument("--frames", type=int, default=24, help="Total GIF frames.")
    parser.add_argument(
        "--duration",
        type=int,
        default=80,
        help="Duration of each GIF frame in milliseconds.",
    )
    parser.add_argument(
        "--blur-kernel",
        type=int,
        default=5,
        help="Odd GaussianBlur kernel size.",
    )
    parser.add_argument(
        "--canny-low",
        type=int,
        default=80,
        help="Lower Canny threshold.",
    )
    parser.add_argument(
        "--canny-high",
        type=int,
        default=160,
        help="Upper Canny threshold.",
    )
    parser.add_argument(
        "--color-frames-ratio",
        type=float,
        default=0.35,
        help="Fraction of frames used for color fill.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    options = RenderOptions(
        frames=args.frames,
        duration=args.duration,
        blur_kernel=args.blur_kernel,
        canny_low=args.canny_low,
        canny_high=args.canny_high,
        color_frames_ratio=args.color_frames_ratio,
    )

    try:
        result = render_sketch_gif(args.input_path, args.output_path, options)
    except ValueError as error:
        # Keep failures easy for NodeJS to consume: stdout stays empty, stderr
        # receives a human-readable message, and the exit code is non-zero.
        print(str(error), file=sys.stderr)
        return 2

    # stdout is intentionally only JSON. A future npm wrapper can parse this
    # directly without filtering progress bars or debug logs.
    print(
        json.dumps(
            {
                "output_path": str(result.output_path),
                "frames": result.frames,
                "width": result.width,
                "height": result.height,
            },
            ensure_ascii=True,
        )
    )
    return 0
