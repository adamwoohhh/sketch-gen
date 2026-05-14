from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
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
    parser.add_argument(
        "output_dir",
        nargs="?",
        type=Path,
        help="Destination directory. Defaults to the input image directory.",
    )
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
    parser.add_argument(
        "--log",
        action="store_true",
        help="Write a process log file under ./logs.",
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
        log_path = setup_logging(args.log)
        logging.info("render start input=%s output_dir=%s", args.input_path, args.output_dir)
        output_path = build_output_path(args.input_path, args.output_dir)
        result = render_sketch_gif(args.input_path, output_path, options)
        logging.info("render complete output=%s frames=%s", result.output_path, result.frames)
    except ValueError as error:
        # Keep failures easy for NodeJS to consume: stdout stays empty, stderr
        # receives a human-readable message, and the exit code is non-zero.
        print(str(error), file=sys.stderr)
        return 2

    # stdout is intentionally only JSON. A future npm wrapper can parse this
    # directly without filtering progress bars or debug logs.
    payload = {
        "output_path": str(result.output_path),
        "frames": result.frames,
        "width": result.width,
        "height": result.height,
    }
    if log_path is not None:
        payload["log_path"] = str(log_path)

    print(json.dumps(payload, ensure_ascii=True))
    return 0


def setup_logging(enabled: bool) -> Path | None:
    if not enabled:
        logging.disable(logging.CRITICAL)
        return None

    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    log_path = log_dir / f"sketch-gen-{timestamp}.log"

    # 日志只写文件，不写 stdout/stderr。这样 NodeJS 调用时 stdout 仍然是纯 JSON，
    # 不会因为调试文本混入而导致 JSON.parse 失败。
    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        force=True,
    )
    return log_path


def build_output_path(input_path: Path, output_dir: Path | None) -> Path:
    # The CLI accepts a directory instead of a final filename so callers do not
    # need to invent unique GIF names. This also keeps the future NodeJS wrapper
    # simple: pass an input path and optionally an output directory.
    input_path = Path(input_path)
    target_dir = Path(output_dir) if output_dir is not None else input_path.parent

    # Use the original file stem plus a compact timestamp. Example:
    # photo.png -> photo-20260514105523.gif
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    return target_dir / f"{input_path.stem}-{timestamp}.gif"
