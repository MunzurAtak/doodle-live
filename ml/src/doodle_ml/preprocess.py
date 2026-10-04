"""Stroke preprocessing (SPEC §4). Must stay identical to ``web/src/ml/preprocess.ts``.

Pipeline for one drawing (a list of strokes, each a list of (x, y) points):

1. normalise   shift to the origin, scale uniformly so the longer side is 255
2. resample    insert points so consecutive points are at most ``resample_spacing`` apart
3. simplify    Ramer-Douglas-Peucker with ``rdp_epsilon``
4. stroke-3    rows of (dx, dy, pen_lift); the first offset is from (0, 0)
5. scale       dx, dy divided by ``offset_scale``
6. pad         truncate / zero-pad to ``max_len`` rows, plus a 0/1 mask

Cross-language rules, so Python and TypeScript produce the same floats:
- plain scalar float64 arithmetic in the same order as the TypeScript code (no numpy
  vectorisation, no ``math.hypot``: distances use ``sqrt(dx * dx + dy * dy)``);
- RDP picks the *first* point with the largest distance and keeps it only if that
  distance is strictly greater than epsilon; distance is point-to-segment;
- strokes without points are dropped; zero-length segments are dropped during resampling.

Steps 1-3 (:func:`simplify_drawing`) are separate from steps 4-6 (:func:`encode`) so
training can cache the simplified drawing and apply augmentation in between.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

Point = tuple[float, float]
Stroke = list[Point]
Drawing = list[Stroke]

TARGET_SIZE = 255.0


@dataclass(frozen=True)
class PreprocessConfig:
    max_len: int = 200
    rdp_epsilon: float = 2.0
    resample_spacing: float = 1.0
    offset_scale: float = 1.0


def to_drawing(strokes: Sequence[Sequence[Sequence[float]]]) -> Drawing:
    """Convert any nested sequence / (n, 2) arrays to plain float tuples, dropping
    empty strokes."""
    drawing: Drawing = []
    for stroke in strokes:
        points = [(float(p[0]), float(p[1])) for p in stroke]
        if points:
            drawing.append(points)
    return drawing


def normalize(drawing: Drawing) -> Drawing:
    """Step 1: shift to the origin and scale the longer side to 255."""
    if not drawing:
        return []
    min_x = min(x for stroke in drawing for x, _ in stroke)
    min_y = min(y for stroke in drawing for _, y in stroke)
    max_x = max(x for stroke in drawing for x, _ in stroke)
    max_y = max(y for stroke in drawing for _, y in stroke)
    extent = max(max_x - min_x, max_y - min_y)
    if extent == 0.0:
        return [[(x - min_x, y - min_y) for x, y in stroke] for stroke in drawing]
    # Multiply before dividing so the longest side maps to exactly 255.0.
    return [
        [
            ((x - min_x) * TARGET_SIZE / extent, (y - min_y) * TARGET_SIZE / extent)
            for x, y in stroke
        ]
        for stroke in drawing
    ]


def resample_stroke(stroke: Stroke, spacing: float) -> Stroke:
    """Step 2: linear interpolation so consecutive points are at most ``spacing`` apart."""
    if not stroke:
        return []
    out: Stroke = [stroke[0]]
    for i in range(1, len(stroke)):
        x0, y0 = out[-1]
        x1, y1 = stroke[i]
        dx = x1 - x0
        dy = y1 - y0
        dist = math.sqrt(dx * dx + dy * dy)
        if dist == 0.0:
            continue
        n = math.ceil(dist / spacing)
        for k in range(1, n):
            t = k / n
            out.append((x0 + dx * t, y0 + dy * t))
        out.append((x1, y1))
    return out


def _segment_distance(p: Point, a: Point, b: Point) -> float:
    """Distance from ``p`` to the segment ``a``-``b``."""
    px, py = p
    ax, ay = a
    bx, by = b
    vx = bx - ax
    vy = by - ay
    wx = px - ax
    wy = py - ay
    length_sq = vx * vx + vy * vy
    if length_sq == 0.0:
        return math.sqrt(wx * wx + wy * wy)
    t = (wx * vx + wy * vy) / length_sq
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0
    cx = ax + vx * t - px
    cy = ay + vy * t - py
    return math.sqrt(cx * cx + cy * cy)


def rdp(stroke: Stroke, epsilon: float) -> Stroke:
    """Step 3: Ramer-Douglas-Peucker simplification (iterative, keeps endpoints)."""
    n = len(stroke)
    if n <= 2:
        return list(stroke)
    keep = [False] * n
    keep[0] = True
    keep[n - 1] = True
    stack = [(0, n - 1)]
    while stack:
        start, end = stack.pop()
        max_dist = -1.0
        index = -1
        for i in range(start + 1, end):
            d = _segment_distance(stroke[i], stroke[start], stroke[end])
            if d > max_dist:
                max_dist = d
                index = i
        if index != -1 and max_dist > epsilon:
            keep[index] = True
            stack.append((start, index))
            stack.append((index, end))
    return [p for p, k in zip(stroke, keep, strict=True) if k]


def simplify_drawing(
    strokes: Sequence[Sequence[Sequence[float]]],
    config: PreprocessConfig = PreprocessConfig(),  # noqa: B008 (frozen dataclass)
) -> Drawing:
    """Steps 1-3: normalise, resample, simplify. Returns absolute coordinates."""
    drawing = normalize(to_drawing(strokes))
    return [rdp(resample_stroke(s, config.resample_spacing), config.rdp_epsilon) for s in drawing]


def to_stroke3(drawing: Drawing) -> list[tuple[float, float, float]]:
    """Step 4: (dx, dy, pen_lift) rows; pen_lift is 1 on the last point of each stroke."""
    rows: list[tuple[float, float, float]] = []
    prev_x = 0.0
    prev_y = 0.0
    for stroke in drawing:
        last = len(stroke) - 1
        for i, (x, y) in enumerate(stroke):
            rows.append((x - prev_x, y - prev_y, 1.0 if i == last else 0.0))
            prev_x = x
            prev_y = y
    return rows


def encode(
    drawing: Drawing,
    config: PreprocessConfig = PreprocessConfig(),  # noqa: B008
) -> tuple[np.ndarray, np.ndarray]:
    """Steps 4-6 on an already simplified drawing. Returns float32 ``(max_len, 3)``
    stroke rows and a float32 ``(max_len,)`` mask."""
    rows = to_stroke3(drawing)[: config.max_len]
    strokes = np.zeros((config.max_len, 3), dtype=np.float32)
    mask = np.zeros(config.max_len, dtype=np.float32)
    for i, (dx, dy, lift) in enumerate(rows):
        strokes[i, 0] = dx / config.offset_scale
        strokes[i, 1] = dy / config.offset_scale
        strokes[i, 2] = lift
        mask[i] = 1.0
    return strokes, mask


def preprocess(
    strokes: Sequence[Sequence[Sequence[float]]],
    config: PreprocessConfig = PreprocessConfig(),  # noqa: B008
) -> tuple[np.ndarray, np.ndarray]:
    """Full pipeline (steps 1-6), as used on the live canvas."""
    return encode(simplify_drawing(strokes, config), config)
