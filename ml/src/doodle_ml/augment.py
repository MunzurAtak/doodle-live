"""Training-time augmentation and a fast (vectorised) stroke-3 encoder.

Drawings here are lists of float ``(n, 2)`` arrays that already went through
preprocessing steps 1-3 (see :mod:`doodle_ml.cache`).

Order for one training sample:

1. prefix   with probability ``prefix_prob`` keep only the first k strokes,
            k ~ U{1..n_strokes} (teaches the model to guess early, SPEC §5)
2. affine   independent x/y scale and a small rotation
3. normalise again, exactly like the live canvas does for whatever is drawn so far
4. encode   stroke-3 rows scaled by ``offset_scale`` (steps 4-6, unpadded)

Steps 1-3 change the drawing *after* simplification, so a prefix is not re-simplified at
its new scale. That's a small approximation; evaluation uses the exact pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from doodle_ml.preprocess import TARGET_SIZE

ArrayDrawing = list[np.ndarray]


@dataclass(frozen=True)
class AugmentConfig:
    prefix_prob: float = 0.5
    scale_min: float = 0.9
    scale_max: float = 1.1
    rotate_deg: float = 10.0


def take_prefix(drawing: ArrayDrawing, k: int) -> ArrayDrawing:
    return drawing[: max(1, k)]


def random_prefix(drawing: ArrayDrawing, rng: np.random.Generator, prob: float) -> ArrayDrawing:
    if len(drawing) > 1 and rng.random() < prob:
        return take_prefix(drawing, int(rng.integers(1, len(drawing) + 1)))
    return drawing


def affine(drawing: ArrayDrawing, rng: np.random.Generator, config: AugmentConfig) -> ArrayDrawing:
    sx, sy = rng.uniform(config.scale_min, config.scale_max, size=2)
    theta = np.deg2rad(rng.uniform(-config.rotate_deg, config.rotate_deg))
    cos, sin = np.cos(theta), np.sin(theta)
    matrix = np.array([[sx * cos, -sy * sin], [sx * sin, sy * cos]])
    return [stroke.astype(np.float64) @ matrix.T for stroke in drawing]


def normalize_arrays(drawing: ArrayDrawing) -> ArrayDrawing:
    """Vectorised equivalent of :func:`doodle_ml.preprocess.normalize`."""
    if not drawing:
        return []
    points = np.concatenate(drawing).astype(np.float64)
    mins = points.min(axis=0)
    extent = float((points.max(axis=0) - mins).max())
    if extent == 0.0:
        return [s - mins for s in drawing]
    return [(s - mins) * TARGET_SIZE / extent for s in drawing]


def encode_rows(drawing: ArrayDrawing, offset_scale: float, max_len: int) -> np.ndarray:
    """Vectorised steps 4-6 without padding: float32 ``(min(n, max_len), 3)`` rows."""
    if not drawing:
        return np.zeros((0, 3), dtype=np.float32)
    points = np.concatenate(drawing).astype(np.float64)
    rows = np.zeros((len(points), 3), dtype=np.float64)
    rows[:, :2] = np.diff(points, axis=0, prepend=np.zeros((1, 2)))
    rows[:, :2] /= offset_scale
    rows[np.cumsum([len(s) for s in drawing]) - 1, 2] = 1.0
    return rows[:max_len].astype(np.float32)


def augment(drawing: ArrayDrawing, rng: np.random.Generator, config: AugmentConfig) -> ArrayDrawing:
    drawing = random_prefix(drawing, rng, config.prefix_prob)
    drawing = affine(drawing, rng, config)
    return normalize_arrays(drawing)
