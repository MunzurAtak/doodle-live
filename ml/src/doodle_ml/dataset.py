"""Turn the downloaded raw ndjson files into deterministic train/val/test splits.

Drawings are ragged (variable number of strokes, variable points per stroke), so each
split is stored as flat arrays in one compressed ``.npz``:

- ``points``              (P, 2) uint8   all (x, y) points, concatenated
- ``stroke_lengths``      (S,)   int32   points per stroke
- ``strokes_per_drawing`` (N,)   int32   strokes per drawing
- ``labels``              (N,)   int16   index into ``categories``
- ``categories``          (C,)   str

:class:`DrawingSet` wraps these and gives back a drawing as a list of ``(n, 2)`` arrays.

Usage (from ``ml/``)::

    uv run python -m doodle_ml.dataset              # build splits + print summary
    uv run python -m doodle_ml.dataset --summary-only
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from doodle_ml.config import PROCESSED_DIR, RAW_DIR, SEED, category_slug, load_categories

SPLITS = ("train", "val", "test")

Drawing = list[np.ndarray]  # list of strokes, each an (n, 2) array of x, y


@dataclass(frozen=True)
class SplitSizes:
    train: int = 5000
    val: int = 500
    test: int = 500

    @property
    def total(self) -> int:
        return self.train + self.val + self.test

    @classmethod
    def from_total(cls, n: int) -> SplitSizes:
        """10:1:1 split, e.g. 6000 -> 5000 / 500 / 500."""
        held_out = n // 12
        if held_out == 0:
            raise ValueError(f"need at least 12 drawings per class, got {n}")
        return cls(train=n - 2 * held_out, val=held_out, test=held_out)


def parse_drawing(raw: Sequence[Sequence[Sequence[float]]]) -> Drawing:
    """Convert Quick, Draw! ``[[xs], [ys]]`` strokes to (n, 2) uint8 arrays.

    Empty strokes are dropped. Raises ValueError on malformed or out-of-range data
    (the simplified dataset is scaled to 0..255).
    """
    strokes: Drawing = []
    for stroke in raw:
        if len(stroke) < 2:
            raise ValueError("stroke must contain x and y lists")
        xs, ys = stroke[0], stroke[1]
        if len(xs) != len(ys):
            raise ValueError("x and y lists differ in length")
        if len(xs) == 0:
            continue
        arr = np.column_stack([xs, ys])
        if arr.min() < 0 or arr.max() > 255:
            raise ValueError("coordinates outside 0..255")
        strokes.append(arr.astype(np.uint8))
    return strokes


def read_raw_category(path: Path) -> list[Drawing]:
    """Read one raw ndjson file, skipping drawings that are empty or malformed."""
    drawings: list[Drawing] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                drawing = parse_drawing(json.loads(line)["drawing"])
            except (ValueError, KeyError, TypeError):
                continue
            if drawing:
                drawings.append(drawing)
    return drawings


@dataclass
class DrawingSet:
    points: np.ndarray
    stroke_lengths: np.ndarray
    strokes_per_drawing: np.ndarray
    labels: np.ndarray
    categories: list[str]
    _stroke_starts: np.ndarray = field(init=False, repr=False)
    _drawing_starts: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if len(self.labels) != len(self.strokes_per_drawing):
            raise ValueError("labels and strokes_per_drawing differ in length")
        if int(self.strokes_per_drawing.sum()) != len(self.stroke_lengths):
            raise ValueError("strokes_per_drawing does not sum to the number of strokes")
        if int(self.stroke_lengths.sum()) != len(self.points):
            raise ValueError("stroke_lengths does not sum to the number of points")
        zero = np.zeros(1, dtype=np.int64)
        self._stroke_starts = np.concatenate([zero, np.cumsum(self.stroke_lengths)])
        self._drawing_starts = np.concatenate([zero, np.cumsum(self.strokes_per_drawing)])

    def __len__(self) -> int:
        return len(self.labels)

    def strokes(self, i: int) -> Drawing:
        first, last = self._drawing_starts[i], self._drawing_starts[i + 1]
        starts = self._stroke_starts
        return [self.points[starts[j] : starts[j + 1]] for j in range(first, last)]

    def points_per_drawing(self) -> np.ndarray:
        return np.diff(self._stroke_starts[self._drawing_starts])

    def __getitem__(self, i: int) -> tuple[Drawing, int]:
        return self.strokes(i), int(self.labels[i])

    @classmethod
    def from_drawings(
        cls, drawings: Sequence[Drawing], labels: Sequence[int], categories: list[str]
    ) -> DrawingSet:
        if len(drawings) != len(labels):
            raise ValueError("drawings and labels differ in length")
        all_strokes = [stroke for drawing in drawings for stroke in drawing]
        points = (
            np.concatenate(all_strokes).astype(np.uint8)
            if all_strokes
            else np.zeros((0, 2), dtype=np.uint8)
        )
        return cls(
            points=points,
            stroke_lengths=np.array([len(s) for s in all_strokes], dtype=np.int32),
            strokes_per_drawing=np.array([len(d) for d in drawings], dtype=np.int32),
            labels=np.asarray(labels, dtype=np.int16),
            categories=list(categories),
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            points=self.points,
            stroke_lengths=self.stroke_lengths,
            strokes_per_drawing=self.strokes_per_drawing,
            labels=self.labels,
            categories=np.array(self.categories),
        )

    @classmethod
    def load(cls, path: Path) -> DrawingSet:
        with np.load(path, allow_pickle=False) as data:
            return cls(
                points=data["points"],
                stroke_lengths=data["stroke_lengths"],
                strokes_per_drawing=data["strokes_per_drawing"],
                labels=data["labels"],
                categories=[str(c) for c in data["categories"]],
            )


def split_indices(n: int, sizes: SplitSizes, seed: int) -> dict[str, np.ndarray]:
    """Deterministic, disjoint index sets for one category."""
    if n < sizes.total:
        raise ValueError(f"need {sizes.total} drawings, got {n}")
    order = np.random.default_rng(seed).permutation(n)
    bounds = np.cumsum([sizes.train, sizes.val, sizes.test])
    return {
        "train": order[: bounds[0]],
        "val": order[bounds[0] : bounds[1]],
        "test": order[bounds[1] : bounds[2]],
    }


def build_splits(
    categories: list[str],
    raw_dir: Path = RAW_DIR,
    sizes: SplitSizes = SplitSizes(),  # noqa: B008 (frozen dataclass, safe default)
    seed: int = SEED,
) -> dict[str, DrawingSet]:
    drawings: dict[str, list[Drawing]] = {s: [] for s in SPLITS}
    labels: dict[str, list[int]] = {s: [] for s in SPLITS}
    for label, category in enumerate(categories):
        path = raw_dir / f"{category_slug(category)}.ndjson"
        if not path.exists():
            raise FileNotFoundError(f"{path} missing; run doodle_ml.download first")
        category_drawings = read_raw_category(path)
        try:
            indices = split_indices(len(category_drawings), sizes, seed + label)
        except ValueError as exc:
            raise ValueError(f"{category}: {exc}") from exc
        for split, idx in indices.items():
            drawings[split].extend(category_drawings[i] for i in idx)
            labels[split].extend([label] * len(idx))
    return {s: DrawingSet.from_drawings(drawings[s], labels[s], categories) for s in SPLITS}


def save_splits(sets: dict[str, DrawingSet], out_dir: Path, sizes: SplitSizes, seed: int) -> None:
    for split, ds in sets.items():
        ds.save(out_dir / f"{split}.npz")
    meta = {
        "categories": sets["train"].categories,
        "sizes": {"train": sizes.train, "val": sizes.val, "test": sizes.test},
        "seed": seed,
        "source": "Quick, Draw! simplified ndjson (CC BY 4.0), recognized drawings only",
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def load_splits(out_dir: Path = PROCESSED_DIR) -> dict[str, DrawingSet]:
    return {s: DrawingSet.load(out_dir / f"{s}.npz") for s in SPLITS}


def summarize(sets: dict[str, DrawingSet]) -> str:
    header = f"{'split':<6} {'drawings':>9} {'classes':>8} {'min/class':>10} "
    header += f"{'avg strokes':>12} {'avg points':>11} {'max points':>11}"
    rows = [header, "-" * len(header)]
    for split, ds in sets.items():
        drawing_points = ds.points_per_drawing()
        per_class = np.bincount(ds.labels, minlength=len(ds.categories))
        rows.append(
            f"{split:<6} {len(ds):>9,} {len(ds.categories):>8} {per_class.min():>10,} "
            f"{ds.strokes_per_drawing.mean():>12.2f} {drawing_points.mean():>11.1f} "
            f"{drawing_points.max():>11,}"
        )
    return "\n".join(rows)


def main(argv: list[str] | None = None, categories: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build train/val/test splits.")
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--out-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--per-class", type=int, default=SplitSizes().total)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args(argv)

    if args.summary_only:
        sets = load_splits(args.out_dir)
    else:
        sizes = SplitSizes.from_total(args.per_class)
        sets = build_splits(categories or load_categories(), args.raw_dir, sizes, args.seed)
        save_splits(sets, args.out_dir, sizes, args.seed)
        print(f"Saved splits to {args.out_dir}")
    print(summarize(sets))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
