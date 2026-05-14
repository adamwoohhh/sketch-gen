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
    color_frames_ratio: float = 0.65


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

    # 检查输出文件必须是.gif格式
    if output_path.suffix.lower() != ".gif":
        raise ValueError("output path must end with .gif")

    # OpenCV reads images as BGR, not RGB. Most image libraries and browsers use
    # RGB, so we convert after loading to keep color blending intuitive later.
    source_bgr = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
    if source_bgr is None:
        raise ValueError(f"input image could not be read: {input_path}")
    # 将图片信息中的颜色转换为RGB格式
    source_rgb = cv2.cvtColor(source_bgr, cv2.COLOR_BGR2RGB)

    # 从图片信息中取出宽高数据
    height, width = source_rgb.shape[:2]

    # Canny works on a single brightness channel. Grayscale removes color while
    # preserving light/dark structure, which is what edge detection needs.
    gray = cv2.cvtColor(source_bgr, cv2.COLOR_BGR2GRAY)

    # 高斯模糊
    # Gaussian blur softens tiny image noise before edge detection. Without this
    # step, Canny tends to produce many speckles that look unlike hand sketching.
    blurred = cv2.GaussianBlur(gray, (options.blur_kernel, options.blur_kernel), 0)

    # 边缘检测
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
    # frames 为 gif 总帧数，color_frames_ratio 比例的帧数用于颜色填充，剩余的帧数用于线条绘制
    color_frame_count = max(1, round(options.frames * options.color_frames_ratio))
    line_frame_count = max(1, options.frames - color_frame_count)

    # 生成绘制线条的帧
    line_frames = _build_line_reveal_frames(edges, line_frame_count)
    # 取线稿的最后一帧作为填色的起始帧
    final_sketch = np.array(line_frames[-1], dtype=np.uint8)
    # 生成填色的帧
    color_frames = _build_color_fill_frames(source_rgb, final_sketch, color_frame_count)

    return line_frames + color_frames


def _build_line_reveal_frames(edges: np.ndarray, frame_count: int) -> list[Image.Image]:
    height, width = edges.shape
    # 根据边缘图的宽高创建一个白色的画布
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    # 取出所有边缘像素的行列索引，并尽量按照“同一笔划连续绘制”的顺序排列。
    edge_rows, edge_cols = _order_edge_pixels(edges)

    # 按照既定的帧数，将像素点按顺序填充到每一帧
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

# 边缘像素点排序，连续的笔划优先绘制，分叉时按固定顺序走，保证动画稳定可复现。
def _order_edge_pixels(edges: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Canny 生成的线稿变成 0/1 二值图，方便后续连通组件分析。边缘像素是 255，非边缘像素是 0。
    edge_mask = (edges > 0).astype(np.uint8)
    # 计算图片的连接组件
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        edge_mask,
        # 寻找周围 8 个相邻的像素
        connectivity=8,
    )

    ordered_points: list[tuple[int, int]] = []
    # 组件 0 是背景，所以这里从 1 开始，提取出所有的边
    component_ids = range(1, component_count)

    # 计算边的顺序
    # 按照组件的上边界坐标和左边界坐标排序。这样动画会先画上边靠上的笔划，再画下边的；同一水平线上的笔划会先画左边的，再画右边的。
    sorted_component_ids = sorted(
        component_ids,
        # lamda 用于定义临时函数
        # 这里通过临时函数返回了一个元组，元组的第一个元素是组件的上边界坐标，第二个元素是组件的左边界坐标。
        key=lambda label: (
            stats[label, cv2.CC_STAT_TOP],
            stats[label, cv2.CC_STAT_LEFT],
        ),
    )

    for label in sorted_component_ids:
        component_rows, component_cols = np.where(labels == label)
        component_points = set(zip(component_rows.tolist(), component_cols.tolist()))
        # 计算边内部的像素点顺序
        ordered_points.extend(_walk_connected_stroke(component_points))

    if not ordered_points:
        empty = np.array([], dtype=np.int64)
        return empty, empty

    rows, cols = zip(*ordered_points)
    return np.array(rows, dtype=np.int64), np.array(cols, dtype=np.int64)


# 计算边内部的像素点顺序
def _walk_connected_stroke(
    component_points: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    remaining = set(component_points)
    ordered: list[tuple[int, int]] = []

    # 不断从剩余的像素点中选择一个起始点，优先选择端点（只有一个或没有相邻点的像素）。
    # 将起始点加入有序列表并从剩余集合中移除，然后寻找相邻像素点，优先选择行坐标较小的；行坐标相同再选择列坐标较小的。重复这个过程，直到没有剩余像素点。
    while remaining:
        current = _choose_stroke_start(remaining)

        while current in remaining:
            ordered.append(current)
            remaining.remove(current)

            neighbors = [
                point
                for point in _neighbor_points(current)
                if point in remaining
            ]
            if not neighbors:
                break

            # 在同一连通笔划内部，优先走到相邻像素。遇到分叉时使用固定排序，
            # 这样动画稳定可复现，而不是每次随机选择一个方向。
            current = min(neighbors, key=lambda point: (point[0], point[1]))

    return ordered


# 找出笔划的起始点。
# 通常是端点（只有一个或没有相邻点的像素），如果没有端点（比如一个闭环），就从任意像素开始。多个候选时，选择行坐标较小的；行坐标相同再选择列坐标较小的。
def _choose_stroke_start(points: set[tuple[int, int]]) -> tuple[int, int]:
    # 找出所有端点（相邻像素不超过 1 个的像素）
    endpoints = [
        point
        for point in points
        if _remaining_neighbor_count(point, points) <= 1
    ]
    # 如果没有端点（比如一个闭环），就从任意像素开始。多个候选时，选择行坐标较小的；行坐标相同再选择列坐标较小的。
    candidates = endpoints or list(points)
    return min(candidates, key=lambda point: (point[0], point[1]))


# 计算 point 的 8 个相邻点中还有多少个在 points 里。这个函数用来判断一个像素点是笔划的末端（只有一个或没有相邻点）还是中间（有两个或更多相邻点）。
def _remaining_neighbor_count(
    point: tuple[int, int],
    points: set[tuple[int, int]],
) -> int:
    return sum(neighbor in points for neighbor in _neighbor_points(point))


# 给定的点的八个邻居坐标，从左上角开始顺时针方向排列。
def _neighbor_points(point: tuple[int, int]) -> list[tuple[int, int]]:
    row, col = point
    return [
        (row - 1, col - 1),
        (row - 1, col),
        (row - 1, col + 1),
        (row, col - 1),
        (row, col + 1),
        (row + 1, col - 1),
        (row + 1, col),
        (row + 1, col + 1),
    ]


def _build_color_fill_frames(
    source_rgb: np.ndarray,
    sketch_rgb: np.ndarray,
    frame_count: int,
) -> list[Image.Image]:
    frames: list[Image.Image] = []
    current_frame = sketch_rgb.copy()
    # 找出所有线条所在的像素（小于250视为线条，255为纯白）
    line_mask = np.any(sketch_rgb < 250, axis=2)

    # 根据原图+线稿，计算出所有的色块
    fill_chunks = _build_color_fill_chunks(source_rgb, line_mask, frame_count)

    # 颜色填充不再整张图一起淡入，而是每一帧只涂一个连续像素块。这样看起来
    # 更像画笔一笔一笔把色块补上；背景区域会被排在最后填充。
    for chunk_mask in fill_chunks:
        if np.any(chunk_mask):
            current_frame[chunk_mask] = source_rgb[chunk_mask]

        # 线稿像素始终压在颜色之上，避免填色时把线条盖掉。
        frame = current_frame.copy()
        frame[line_mask] = (25, 25, 25)
        frames.append(Image.fromarray(frame, mode="RGB"))

    while len(frames) < frame_count:
        frames.append(frames[-1].copy())

    return frames


def _build_color_fill_chunks(
    source_rgb: np.ndarray,
    line_mask: np.ndarray,
    frame_count: int,
) -> list[np.ndarray]:
    regions = _find_color_regions(source_rgb, line_mask)
    if not regions:
        empty_mask = np.zeros_like(line_mask, dtype=bool)
        return [empty_mask.copy() for _ in range(frame_count)]

    foreground_regions = [
        region
        for region, is_background in regions
        if not is_background
    ]
    background_regions = [
        region
        for region, is_background in regions
        if is_background
    ]

    background_mask = np.zeros_like(line_mask, dtype=bool)
    for region in background_regions:
        background_mask |= region

    foreground_frame_count = frame_count - 1 if np.any(background_mask) else frame_count
    if foreground_frame_count <= 0:
        return [background_mask]

    foreground_chunks = _build_region_chunks(
        foreground_regions,
        foreground_frame_count,
        line_mask,
    )
    if np.any(background_mask):
        return foreground_chunks + [background_mask]
    return foreground_chunks


def _build_region_chunks(
    regions: list[np.ndarray],
    frame_count: int,
    template_mask: np.ndarray,
) -> list[np.ndarray]:
    if not regions:
        return [np.zeros_like(template_mask, dtype=bool) for _ in range(frame_count)]

    # 先保证每个色块至少有机会单独出现；如果帧数更多，再把大色块拆成多笔。
    chunks: list[np.ndarray] = []
    extra_frames = max(0, frame_count - len(regions))
    region_sizes = [int(np.count_nonzero(region)) for region in regions]
    total_pixels = sum(region_sizes)

    for index, region in enumerate(regions):
        extra_for_region = 0
        if total_pixels > 0 and extra_frames > 0:
            if index == len(regions) - 1:
                extra_for_region = extra_frames
            else:
                extra_for_region = round(
                    extra_frames * np.count_nonzero(region) / total_pixels
                )
                extra_for_region = min(extra_for_region, extra_frames)
            extra_frames -= extra_for_region

        split_count = 1 + extra_for_region
        chunks.extend(_split_region_mask(region, split_count))

    if len(chunks) <= frame_count:
        return _ensure_chunk_count(chunks, frame_count)

    return _ensure_chunk_count(_pack_chunks_evenly(chunks, frame_count), frame_count)


def _ensure_chunk_count(
    chunks: list[np.ndarray],
    frame_count: int,
) -> list[np.ndarray]:
    chunks = [chunk for chunk in chunks if np.any(chunk)]
    if not chunks:
        return chunks

    while len(chunks) < frame_count:
        largest_index = max(
            range(len(chunks)),
            key=lambda index: int(np.count_nonzero(chunks[index])),
        )
        largest = chunks[largest_index]
        if int(np.count_nonzero(largest)) <= 1:
            break
        first_half, second_half = _split_region_mask(largest, 2)
        chunks[largest_index : largest_index + 1] = [first_half, second_half]

    return chunks


def _pack_chunks_evenly(
    chunks: list[np.ndarray],
    frame_count: int,
) -> list[np.ndarray]:
    total_pixels = sum(int(np.count_nonzero(chunk)) for chunk in chunks)
    if total_pixels == 0:
        return chunks[:frame_count]

    packed_chunks: list[np.ndarray] = []
    current_mask = np.zeros_like(chunks[0], dtype=bool)
    current_pixels = 0
    target_pixels = max(1, round(total_pixels / frame_count))

    for chunk in chunks:
        chunk_pixels = int(np.count_nonzero(chunk))
        if (
            len(packed_chunks) < frame_count - 1
            and current_pixels > 0
            and current_pixels + chunk_pixels > target_pixels
        ):
            packed_chunks.append(current_mask)
            current_mask = np.zeros_like(chunk, dtype=bool)
            current_pixels = 0

        current_mask |= chunk
        current_pixels += chunk_pixels

    if current_pixels > 0:
        packed_chunks.append(current_mask)

    if len(packed_chunks) > frame_count:
        tail_mask = np.zeros_like(chunks[0], dtype=bool)
        for chunk in packed_chunks[frame_count - 1 :]:
            tail_mask |= chunk
        packed_chunks = packed_chunks[: frame_count - 1] + [tail_mask]

    return packed_chunks


def _find_color_regions(
    source_rgb: np.ndarray,
    line_mask: np.ndarray,
) -> list[tuple[np.ndarray, bool]]:
    height, width = line_mask.shape
    # 对线稿的像素点取反，即所有非线条的像素点都被视为可填色区域。
    paintable_mask = ~line_mask
    # 真实图片里同一色块会有轻微噪声，所以先把颜色按 64 级量化。相近颜色会
    # 被归为同一组，再按连通组件切分，得到“同色且连续”的填色区域。
    quantized = source_rgb // 64

    regions: list[tuple[np.ndarray, bool, tuple[int, int], int]] = []
    # 只对出现最多的颜色桶做精细连通分块。真实照片可能有大量细小颜色变化，
    # 如果每个颜色桶都跑一次连通组件会很慢；主色块负责呈现涂抹过程，剩余
    # 像素会在最后的兜底区域补完。
    paintable_colors = quantized[paintable_mask]
    # 从 paintable_colors 中找出所有颜色和每个颜色像素点数量
    unique_colors, counts = np.unique(paintable_colors, axis=0, return_counts=True)
    max_color_buckets = 16
    # 按大到小取前 max_color_buckets 个颜色桶的索引
    top_indexes = np.argsort(counts)[-max_color_buckets:]
    # 前 max_color_buckets 个颜色桶的颜色值列表
    seen_keys = [
        tuple(color.tolist())
        for color in unique_colors[top_indexes]
    ]
    # 记录已经被主颜色桶覆盖的像素，剩下的零碎颜色后面合并处理。
    processed_mask = np.zeros_like(paintable_mask, dtype=bool)

    for color_key in seen_keys:
        color_mask = np.all(quantized == color_key, axis=2) & paintable_mask
        processed_mask |= color_mask
        component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
            color_mask.astype(np.uint8),
            connectivity=8,
        )

        for label in range(1, component_count):
            rows, cols = np.where(labels == label)
            if len(rows) == 0:
                continue

            is_background = _touches_image_border(rows, cols, height, width)
            # 色块本身已经由 OpenCV 保证是连通区域。这里用布尔 mask 表示整块，
            # 后续可以用 NumPy 一次性赋值，避免把百万像素转成 Python tuple。
            region_mask = labels == label
            top_left = (
                stats[label, cv2.CC_STAT_TOP],
                stats[label, cv2.CC_STAT_LEFT],
            )
            area = stats[label, cv2.CC_STAT_AREA]
            regions.append((region_mask, is_background, top_left, area))

    remaining_mask = paintable_mask & ~processed_mask
    if np.any(remaining_mask):
        regions.extend(
            _build_regions_from_mask(remaining_mask, height, width, force_background=False)
        )

    # 非背景色块先填，接触图片边缘的背景区域最后填。面积大的区域排后一点，
    # 小的主体色块会先一笔笔出现。
    sorted_regions = sorted(
        regions,
        key=lambda item: (item[1], item[3], item[2][0], item[2][1]),
    )
    return [
        (points, is_background)
        for points, is_background, _top_left, _area in sorted_regions
    ]


def _build_regions_from_mask(
    mask: np.ndarray,
    height: int,
    width: int,
    force_background: bool,
) -> list[tuple[np.ndarray, bool, tuple[int, int], int]]:
    regions: list[tuple[np.ndarray, bool, tuple[int, int], int]] = []
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8),
        connectivity=8,
    )

    for label in range(1, component_count):
        rows, cols = np.where(labels == label)
        if len(rows) == 0:
            continue

        is_background = force_background or _touches_image_border(
            rows,
            cols,
            height,
            width,
        )
        region_mask = labels == label
        top_left = (
            stats[label, cv2.CC_STAT_TOP],
            stats[label, cv2.CC_STAT_LEFT],
        )
        area = stats[label, cv2.CC_STAT_AREA]
        regions.append((region_mask, is_background, top_left, area))

    return regions


def _touches_image_border(
    rows: np.ndarray,
    cols: np.ndarray,
    height: int,
    width: int,
) -> bool:
    return bool(
        np.any(rows == 0)
        or np.any(cols == 0)
        or np.any(rows == height - 1)
        or np.any(cols == width - 1)
    )


def _split_region_mask(
    region_mask: np.ndarray,
    split_count: int,
) -> list[np.ndarray]:
    if split_count <= 1 or np.count_nonzero(region_mask) <= 1:
        return [region_mask]

    rows, cols = np.where(region_mask)
    order = np.lexsort((cols, rows))
    rows = rows[order]
    cols = cols[order]

    chunks: list[np.ndarray] = []
    for row_chunk, col_chunk in zip(
        np.array_split(rows, split_count),
        np.array_split(cols, split_count),
    ):
        if len(row_chunk) == 0:
            continue
        chunk_mask = np.zeros_like(region_mask, dtype=bool)
        chunk_mask[row_chunk, col_chunk] = True
        chunks.append(chunk_mask)
    return chunks
