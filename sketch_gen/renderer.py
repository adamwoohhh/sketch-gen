from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


@dataclass(frozen=True)
class RenderOptions:
    frames: int = 24
    duration: int = 80
    blur_kernel: int = 5
    canny_low: int = 80
    canny_high: int = 160
    color_frames_ratio: float = 0.35


@dataclass(frozen=True)
class RenderResult:
    output_path: Path
    frames: int
    width: int
    height: int


def validate_options(options: RenderOptions) -> None:
    if options.frames < 2:
        raise ValueError("--frames must be at least 2")
    if options.duration <= 0:
        raise ValueError("--duration must be greater than 0")
    if options.blur_kernel <= 0 or options.blur_kernel % 2 == 0:
        raise ValueError("--blur-kernel must be a positive odd integer")
    if options.canny_low < 0 or options.canny_high < 0:
        raise ValueError("Canny thresholds must be non-negative")
    if options.canny_low > options.canny_high:
        raise ValueError("--canny-low must be less than or equal to --canny-high")
    if not 0 < options.color_frames_ratio < 1:
        raise ValueError("--color-frames-ratio must be greater than 0 and less than 1")


def render_sketch_gif(
    input_path: str | Path,
    output_path: str | Path,
    options: RenderOptions | None = None,
) -> RenderResult:
    options = options or RenderOptions()
    validate_options(options)

    input_path = Path(input_path)
    output_path = Path(output_path)

    if output_path.suffix.lower() != ".gif":
        raise ValueError("output path must end with .gif")

    # OpenCV reads images as BGR, not RGB. Most image libraries and browsers use
    # RGB, so we convert after loading to keep color blending intuitive later.
    source_bgr = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
    if source_bgr is None:
        raise ValueError(f"input image could not be read: {input_path}")
    source_rgb = cv2.cvtColor(source_bgr, cv2.COLOR_BGR2RGB)

    height, width = source_rgb.shape[:2]

    # Canny works on a single brightness channel. Grayscale removes color while
    # preserving light/dark structure, which is what edge detection needs.
    gray = cv2.cvtColor(source_bgr, cv2.COLOR_BGR2GRAY)

    # Gaussian blur softens tiny image noise before edge detection. Without this
    # step, Canny tends to produce many speckles that look unlike hand sketching.
    blurred = cv2.GaussianBlur(gray, (options.blur_kernel, options.blur_kernel), 0)

    # Canny returns a binary image: edge pixels are 255, non-edge pixels are 0.
    # The two thresholds control how strict edge detection is.
    edges = cv2.Canny(blurred, options.canny_low, options.canny_high)

    frames = _build_animation_frames(source_rgb, edges, options)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=options.duration,
        loop=0,
    )

    return RenderResult(
        output_path=output_path,
        frames=options.frames,
        width=width,
        height=height,
    )


def _build_animation_frames(
    source_rgb: np.ndarray,
    edges: np.ndarray,
    options: RenderOptions,
) -> list[Image.Image]:
    color_frame_count = max(1, round(options.frames * options.color_frames_ratio))
    line_frame_count = max(1, options.frames - color_frame_count)

    line_frames = _build_line_reveal_frames(edges, line_frame_count)
    final_sketch = np.array(line_frames[-1], dtype=np.uint8)
    color_frames = _build_color_fill_frames(source_rgb, final_sketch, color_frame_count)

    return line_frames + color_frames


def _build_line_reveal_frames(edges: np.ndarray, frame_count: int) -> list[Image.Image]:
    height, width = edges.shape
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    edge_rows, edge_cols = np.where(edges > 0)

    # Reveal edge pixels from top to bottom. This deterministic ordering is less
    # artistic than real stroke reconstruction, but it is stable and easy to test.
    order = np.lexsort((edge_cols, edge_rows))
    edge_rows = edge_rows[order]
    edge_cols = edge_cols[order]

    frames: list[Image.Image] = []
    total_edges = len(edge_rows)
    for index in range(frame_count):
        progress = (index + 1) / frame_count
        visible_edges = round(total_edges * progress)

        frame = canvas.copy()
        if visible_edges > 0:
            frame[edge_rows[:visible_edges], edge_cols[:visible_edges]] = (25, 25, 25)
        frames.append(Image.fromarray(frame, mode="RGB"))

    return frames


def _build_color_fill_frames(
    source_rgb: np.ndarray,
    sketch_rgb: np.ndarray,
    frame_count: int,
) -> list[Image.Image]:
    frames: list[Image.Image] = []
    line_mask = np.any(sketch_rgb < 250, axis=2)

    for index in range(frame_count):
        # Blend moves from the completed black-line sketch to the original color.
        # A low alpha keeps the canvas mostly white; alpha=1 is the full source.
        alpha = (index + 1) / frame_count
        blended = (
            sketch_rgb.astype(np.float32) * (1 - alpha)
            + source_rgb.astype(np.float32) * alpha
        ).astype(np.uint8)

        # Re-apply dark line pixels so the final color image still has visible
        # sketch outlines instead of becoming just the original picture.
        blended[line_mask] = (25, 25, 25)
        frames.append(Image.fromarray(blended, mode="RGB"))

    return frames
