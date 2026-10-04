"""Cache preprocessing steps 1-3 (normalise, resample, simplify) for every split and
compute ``offset_scale`` on the training set (SPEC §4 step 5).

Simplification runs in pure Python (to match the TypeScript implementation exactly), so
it is done once here in parallel instead of in every training epoch.

Writes to ``ml/data/processed/``:
- ``simplified_{train,val,test}.npz``  float32 :class:`DrawingSet`
- ``preprocess_meta.json``             preprocessing config incl. ``offset_scale``

Usage (from ``ml/``)::

    uv run python -m doodle_ml.cache
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from doodle_ml.config import PROCESSED_DIR
from doodle_ml.dataset import SPLITS, Drawing, DrawingSet
from doodle_ml.parallel import chunked_map
from doodle_ml.preprocess import PreprocessConfig, simplify_drawing

META_FILE = "preprocess_meta.json"


def simplified_path(split: str, processed_dir: Path = PROCESSED_DIR) -> Path:
    return processed_dir / f"simplified_{split}.npz"


def simplify_many(drawings: Sequence[Drawing]) -> list[Drawing]:
    """Top-level (picklable) worker: steps 1-3 for a chunk of drawings."""
    config = PreprocessConfig()
    out: list[Drawing] = []
    for drawing in drawings:
        simplified = simplify_drawing(drawing, config)
        out.append([np.asarray(stroke, dtype=np.float32) for stroke in simplified])
    return out


def simplify_set(ds: DrawingSet, workers: int | None = None, desc: str | None = None) -> DrawingSet:
    drawings = [ds.strokes(i) for i in range(len(ds))]
    simplified = chunked_map(simplify_many, drawings, workers=workers, desc=desc)
    return DrawingSet.from_drawings(simplified, ds.labels.tolist(), ds.categories, np.float32)


def stroke3_offsets(ds: DrawingSet) -> np.ndarray:
    """All (dx, dy) offsets of the stroke-3 encoding of every drawing in ``ds``
    (the first offset of each drawing is measured from the origin)."""
    starts = ds.drawing_point_starts()
    points = ds.points.astype(np.float64)
    offsets = np.diff(points, axis=0, prepend=np.zeros((1, 2)))
    offsets[starts] = points[starts]
    return offsets


def compute_offset_scale(ds: DrawingSet) -> float:
    """Standard deviation of all dx and dy offsets pooled together."""
    offsets = stroke3_offsets(ds)
    if offsets.size == 0:
        return 1.0
    std = float(offsets.std())
    return std if std > 0 else 1.0


def load_preprocess_config(processed_dir: Path = PROCESSED_DIR) -> PreprocessConfig:
    data = json.loads((processed_dir / META_FILE).read_text(encoding="utf-8"))
    return PreprocessConfig(**data["preprocess"])


def load_simplified(split: str, processed_dir: Path = PROCESSED_DIR) -> DrawingSet:
    path = simplified_path(split, processed_dir)
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; run: uv run python -m doodle_ml.cache")
    return DrawingSet.load(path)


def build_cache(
    processed_dir: Path = PROCESSED_DIR, workers: int | None = None
) -> PreprocessConfig:
    simplified: dict[str, DrawingSet] = {}
    for split in SPLITS:
        raw = DrawingSet.load(processed_dir / f"{split}.npz")
        simplified[split] = simplify_set(raw, workers, desc=f"Simplifying {split}")
        simplified[split].save(simplified_path(split, processed_dir))
    config = replace(PreprocessConfig(), offset_scale=compute_offset_scale(simplified["train"]))
    meta = {
        "preprocess": asdict(config),
        "points": {s: int(len(ds.points)) for s, ds in simplified.items()},
    }
    (processed_dir / META_FILE).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Cache simplified drawings + offset_scale.")
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args(argv)
    config = build_cache(args.processed_dir, args.workers)
    print(f"offset_scale = {config.offset_scale:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
