from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageSequence

from sketch_gen.renderer import RenderOptions, _order_edge_pixels, render_sketch_gif


def _write_synthetic_image(path: Path) -> None:
    image = np.full((32, 32, 3), 255, dtype=np.uint8)
    cv2.rectangle(image, (8, 8), (24, 24), (20, 120, 220), thickness=-1)
    cv2.line(image, (4, 28), (28, 4), (220, 40, 40), thickness=2)
    cv2.imwrite(str(path), image)


def test_render_sketch_gif_creates_requested_frame_count(tmp_path):
    input_path = tmp_path / "input.png"
    output_path = tmp_path / "output.gif"
    _write_synthetic_image(input_path)

    result = render_sketch_gif(
        input_path,
        output_path,
        RenderOptions(frames=8, duration=40),
    )

    assert result.output_path == output_path
    assert result.frames == 8
    assert result.width == 32
    assert result.height == 32
    assert output_path.exists()

    with Image.open(output_path) as gif:
        frames = list(ImageSequence.Iterator(gif))

    assert len(frames) == 8


def test_order_edge_pixels_keeps_connected_strokes_together():
    edges = np.zeros((5, 8), dtype=np.uint8)
    edges[:, 1] = 255
    edges[:, 6] = 255

    rows, cols = _order_edge_pixels(edges)
    ordered_points = list(zip(rows.tolist(), cols.tolist()))
    left_stroke = {(row, 1) for row in range(5)}
    right_stroke = {(row, 6) for row in range(5)}

    first_stroke_points = set(ordered_points[:5])
    second_stroke_points = set(ordered_points[5:])

    assert first_stroke_points in (left_stroke, right_stroke)
    assert second_stroke_points in (left_stroke, right_stroke)
    assert first_stroke_points != second_stroke_points
