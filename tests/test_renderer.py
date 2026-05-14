from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageSequence

from sketch_gen.renderer import (
    RenderOptions,
    _build_color_fill_chunks,
    _build_color_fill_frames,
    _order_edge_pixels,
    render_sketch_gif,
)


def _write_synthetic_image(path: Path) -> None:
    image = np.full((32, 32, 3), 230, dtype=np.uint8)
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


def test_color_fill_frames_paint_connected_color_regions_before_background():
    source_rgb = np.full((6, 8, 3), (240, 220, 120), dtype=np.uint8)
    source_rgb[1:3, 1:3] = (220, 40, 40)
    source_rgb[3:5, 5:7] = (40, 80, 220)
    sketch_rgb = np.full((6, 8, 3), 255, dtype=np.uint8)

    frames = _build_color_fill_frames(source_rgb, sketch_rgb, frame_count=3)
    frame_arrays = [np.array(frame, dtype=np.uint8) for frame in frames]
    changed_masks = [
        np.any(frame != sketch_rgb, axis=2)
        for frame in frame_arrays
    ]

    red_mask = np.zeros((6, 8), dtype=bool)
    red_mask[1:3, 1:3] = True
    blue_mask = np.zeros((6, 8), dtype=bool)
    blue_mask[3:5, 5:7] = True
    background_mask = ~(red_mask | blue_mask)

    first_new_pixels = changed_masks[0]
    second_new_pixels = changed_masks[1] & ~changed_masks[0]
    third_new_pixels = changed_masks[2] & ~changed_masks[1]

    assert first_new_pixels.sum() == 4
    assert second_new_pixels.sum() == 4
    assert first_new_pixels.tolist() in (red_mask.tolist(), blue_mask.tolist())
    assert second_new_pixels.tolist() in (red_mask.tolist(), blue_mask.tolist())
    assert not np.any(changed_masks[1] & background_mask)
    assert third_new_pixels.sum() == background_mask.sum()
    assert np.array_equal(frame_arrays[-1], source_rgb)


def test_color_fill_chunks_spread_many_regions_across_frames():
    source_rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    for row in range(4):
        for col in range(4):
            source_rgb[row, col] = (row * 64, col * 64, 128)
    line_mask = np.zeros((4, 4), dtype=bool)

    chunks = _build_color_fill_chunks(source_rgb, line_mask, frame_count=4)
    chunk_sizes = [int(np.count_nonzero(chunk)) for chunk in chunks]

    assert len(chunks) == 4
    assert sum(chunk_sizes) == 16
    assert max(chunk_sizes) <= 6
    assert min(chunk_sizes) >= 2
