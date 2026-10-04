"""Tiny synthetic, *learnable* drawing sets for training / evaluation tests."""

from __future__ import annotations

import numpy as np

from doodle_ml.dataset import DrawingSet

SHAPES = ["hline", "vline", "square"]


def shape_drawing(kind: str, rng: np.random.Generator) -> list[np.ndarray]:
    jitter = lambda pts: np.clip(np.array(pts, float) + rng.normal(0, 4, (len(pts), 2)), 0, 255)  # noqa: E731
    if kind == "hline":
        return [jitter([[0, 128], [255, 128]])]
    if kind == "vline":
        return [jitter([[128, 0], [128, 255]])]
    return [jitter([[0, 0], [255, 0], [255, 255]]), jitter([[255, 255], [0, 255], [0, 0]])]


def synthetic_set(per_class: int, seed: int = 0, dtype: type = np.uint8) -> DrawingSet:
    rng = np.random.default_rng(seed)
    drawings, labels = [], []
    for label, kind in enumerate(SHAPES):
        for _ in range(per_class):
            drawings.append([s.round() for s in shape_drawing(kind, rng)])
            labels.append(label)
    return DrawingSet.from_drawings(drawings, labels, SHAPES, dtype)
