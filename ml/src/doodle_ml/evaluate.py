"""Evaluate the best checkpoint on the test set (SPEC §5) and write ``reports/``.

- ``metrics.json``                 top-1, top-3, per-class accuracy, model info
- ``accuracy_vs_strokes.{json,png}`` accuracy when the model only sees the first k strokes
- ``confusion_top_pairs.json``     the 15 most frequent (true -> predicted) mistakes

Partial drawings go through the *exact* live pipeline (raw prefix -> normalise ->
resample -> simplify), so the stroke curve reflects what the game will see.
(ONNX latency is measured in the export step.)

Usage (from ``ml/``)::

    uv run python -m doodle_ml.evaluate
    uv run python -m doodle_ml.evaluate --per-class 100   # faster, on a subset
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from doodle_ml.cache import load_simplified, simplify_many
from doodle_ml.config import ML_ROOT, PROCESSED_DIR
from doodle_ml.data import StrokeDataset, collate
from doodle_ml.dataset import Drawing, DrawingSet
from doodle_ml.model import count_parameters
from doodle_ml.parallel import chunked_map
from doodle_ml.preprocess import PreprocessConfig
from doodle_ml.train import CHECKPOINT_DIR, load_model

REPORTS_DIR = ML_ROOT.parent / "reports"

# Reference palette (light surface): categorical slots 1-2 and text tokens.
SURFACE = "#fcfcfb"
SERIES = ("#2a78d6", "#eb6834")
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"


@torch.no_grad()
def predict(
    model: torch.nn.Module,
    drawings: DrawingSet,
    preprocess: PreprocessConfig,
    device: torch.device | str = "cpu",
    batch_size: int = 512,
) -> np.ndarray:
    """Softmax probabilities ``(N, C)`` in the original order of ``drawings``."""
    ds = StrokeDataset(drawings, preprocess, augment_config=None)
    order = np.argsort(ds.lengths(), kind="stable")
    probs = np.zeros((len(ds), len(drawings.categories)), dtype=np.float32)
    model.eval()
    for start in range(0, len(order), batch_size):
        idx = order[start : start + batch_size]
        strokes, mask, _ = collate([ds[int(i)] for i in idx])
        logits = model(strokes.to(device), mask.to(device))
        probs[idx] = logits.softmax(-1).cpu().numpy()
    return probs


def topk_accuracy(probs: np.ndarray, labels: np.ndarray, k: int) -> float:
    topk = np.argsort(-probs, axis=1)[:, :k]
    return float((topk == labels[:, None]).any(axis=1).mean())


def per_class_accuracy(probs: np.ndarray, labels: np.ndarray, categories: list[str]) -> dict:
    preds = probs.argmax(1)
    return {
        c: float((preds[labels == i] == i).mean()) if (labels == i).any() else float("nan")
        for i, c in enumerate(categories)
    }


def confusion_top_pairs(
    probs: np.ndarray, labels: np.ndarray, categories: list[str], n: int = 15
) -> list[dict[str, Any]]:
    preds = probs.argmax(1)
    counts = np.zeros((len(categories), len(categories)), dtype=np.int64)
    np.add.at(counts, (labels, preds), 1)
    np.fill_diagonal(counts, 0)
    support = np.bincount(labels, minlength=len(categories))
    flat = np.argsort(-counts, axis=None, kind="stable")[:n]
    pairs = []
    for t, p in zip(*np.unravel_index(flat, counts.shape), strict=True):
        if counts[t, p] == 0:
            break
        pairs.append(
            {
                "true": categories[t],
                "predicted": categories[p],
                "count": int(counts[t, p]),
                "rate": float(counts[t, p] / support[t]),
            }
        )
    return pairs


def simplify_prefixes(raw: DrawingSet, k: int) -> DrawingSet:
    """Exact live pipeline on the first ``k`` strokes of every drawing."""
    prefixes: list[Drawing] = [raw.strokes(i)[:k] for i in range(len(raw))]
    simplified = chunked_map(simplify_many, prefixes, desc=f"Prefixes k={k}")
    return DrawingSet.from_drawings(simplified, raw.labels.tolist(), raw.categories, np.float32)


def accuracy_vs_strokes(
    model: torch.nn.Module,
    raw: DrawingSet,
    preprocess: PreprocessConfig,
    max_strokes: int,
    device: torch.device | str = "cpu",
) -> list[dict[str, float]]:
    """For k = 1..max_strokes: accuracy seeing only the first k strokes (drawings with
    fewer strokes are shown complete)."""
    labels = raw.labels.astype(np.int64)
    rows = []
    for k in range(1, max_strokes + 1):
        probs = predict(model, simplify_prefixes(raw, k), preprocess, device)
        rows.append(
            {
                "strokes": k,
                "top1": topk_accuracy(probs, labels, 1),
                "top3": topk_accuracy(probs, labels, 3),
                "share_complete": float((raw.strokes_per_drawing <= k).mean()),
            }
        )
    return rows


def plot_accuracy_vs_strokes(rows: Sequence[dict[str, float]], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ks = [r["strokes"] for r in rows]
    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    for (key, name), color in zip(
        [("top3", "Top-3"), ("top1", "Top-1")], SERIES[::-1], strict=True
    ):
        values = [100 * r[key] for r in rows]
        ax.plot(
            ks,
            values,
            color=color,
            linewidth=2,
            marker="o",
            markersize=5,
            markeredgecolor=SURFACE,
            markeredgewidth=1.5,
            label=name,
            zorder=3,
        )
        ax.annotate(
            f"{name} {values[-1]:.0f}%",
            (ks[-1], values[-1]),
            xytext=(8, 0),
            textcoords="offset points",
            va="center",
            color=TEXT_PRIMARY,
            fontsize=9,
        )
    ax.set_xlabel("Strokes seen", color=TEXT_SECONDARY)
    ax.set_ylabel("Test accuracy (%)", color=TEXT_SECONDARY)
    ax.set_title(
        "Accuracy climbs with every stroke",
        loc="left",
        color=TEXT_PRIMARY,
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xticks(ks)
    ax.set_ylim(0, 100)
    ax.set_xlim(ks[0] - 0.3, ks[-1] + 1.2)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.tick_params(colors=TEXT_SECONDARY, length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.legend(frameon=False, loc="lower right", labelcolor=TEXT_PRIMARY)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def run(
    checkpoint: Path,
    processed_dir: Path,
    reports_dir: Path,
    per_class: int | None,
    max_strokes: int,
) -> dict[str, Any]:
    device = "cpu"
    model, ckpt = load_model(checkpoint, device)
    preprocess = PreprocessConfig(**ckpt["preprocess"])
    categories: list[str] = ckpt["categories"]

    raw = DrawingSet.load(processed_dir / "test.npz")
    simplified = load_simplified("test", processed_dir)
    if per_class:
        raw, simplified = raw.first_per_class(per_class), simplified.first_per_class(per_class)
    labels = simplified.labels.astype(np.int64)

    probs = predict(model, simplified, preprocess, device)
    metrics = {
        "n_test": int(len(labels)),
        "top1": topk_accuracy(probs, labels, 1),
        "top3": topk_accuracy(probs, labels, 3),
        "parameters": count_parameters(model),
        "checkpoint_epoch": int(ckpt["epoch"]),
        "val_metrics": ckpt["metrics"],
        "offset_scale": preprocess.offset_scale,
        "per_class_top1": per_class_accuracy(probs, labels, categories),
    }
    pairs = confusion_top_pairs(probs, labels, categories)
    curve = accuracy_vs_strokes(model, raw, preprocess, max_strokes, device)

    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (reports_dir / "confusion_top_pairs.json").write_text(
        json.dumps(pairs, indent=2), encoding="utf-8"
    )
    (reports_dir / "accuracy_vs_strokes.json").write_text(
        json.dumps(curve, indent=2), encoding="utf-8"
    )
    plot_accuracy_vs_strokes(curve, reports_dir / "accuracy_vs_strokes.png")
    return {"metrics": metrics, "pairs": pairs, "curve": curve}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate the classifier on the test set.")
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_DIR / "best.pt")
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--reports-dir", type=Path, default=REPORTS_DIR)
    parser.add_argument("--per-class", type=int, default=None, help="subset for speed")
    parser.add_argument("--max-strokes", type=int, default=8)
    args = parser.parse_args(argv)

    out = run(
        args.checkpoint, args.processed_dir, args.reports_dir, args.per_class, args.max_strokes
    )
    m = out["metrics"]
    print(f"test top-1 {m['top1']:.3f}  top-3 {m['top3']:.3f}  (n={m['n_test']:,})")
    for row in out["curve"]:
        print(f"  {row['strokes']} strokes: top-1 {row['top1']:.3f}  top-3 {row['top3']:.3f}")
    print("most confused:", ", ".join(f"{p['true']}->{p['predicted']}" for p in out["pairs"][:5]))
    print(f"wrote {args.reports_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
