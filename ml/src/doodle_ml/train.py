"""Train the stroke classifier (SPEC §5).

Usage (from ``ml/``)::

    uv run python -m doodle_ml.cache          # once: simplified drawings + offset_scale
    uv run python -m doodle_ml.train --quick  # few-minute sanity run
    uv run python -m doodle_ml.train          # full run
    uv run tensorboard --logdir runs

Saves ``checkpoints/best.pt`` (best validation top-1) and ``checkpoints/last.pt``.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from doodle_ml.augment import AugmentConfig
from doodle_ml.cache import load_preprocess_config, load_simplified
from doodle_ml.config import CONFIG_DIR, ML_ROOT, PROCESSED_DIR
from doodle_ml.data import BucketBatchSampler, StrokeDataset, collate
from doodle_ml.dataset import DrawingSet
from doodle_ml.model import ModelConfig, StrokeTransformer, count_parameters
from doodle_ml.preprocess import PreprocessConfig

CHECKPOINT_DIR = ML_ROOT / "checkpoints"
RUNS_DIR = ML_ROOT / "runs"
TRAIN_CONFIG = CONFIG_DIR / "train.yaml"


@dataclass
class DataSection:
    train_per_class: int | None = None
    val_per_class: int | None = None


@dataclass
class TrainSection:
    epochs: int = 15
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 0.01
    warmup_epochs: float = 1.0
    label_smoothing: float = 0.1
    grad_clip: float | None = 1.0
    num_workers: int = 2
    num_threads: int | None = None
    seed: int = 42


@dataclass
class ModelSection:
    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 4
    d_ff: int = 256
    dropout: float = 0.1


@dataclass
class RunConfig:
    data: DataSection = field(default_factory=DataSection)
    model: ModelSection = field(default_factory=ModelSection)
    augment: AugmentConfig = field(default_factory=AugmentConfig)
    train: TrainSection = field(default_factory=TrainSection)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> RunConfig:
        return cls(
            data=DataSection(**raw.get("data", {})),
            model=ModelSection(**raw.get("model", {})),
            augment=AugmentConfig(**raw.get("augment", {})),
            train=TrainSection(**raw.get("train", {})),
        )


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def load_run_config(path: Path = TRAIN_CONFIG, quick: bool = False) -> RunConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    quick_overrides = raw.pop("quick", {}) or {}
    if quick:
        raw = _merge(raw, quick_overrides)
    return RunConfig.from_dict(raw)


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002 (seed any library code using the global RNG)
    torch.manual_seed(seed)


def warmup_cosine(step: int, warmup_steps: int, total_steps: int) -> float:
    """LR multiplier: linear warmup to 1, then cosine decay to 0."""
    if warmup_steps > 0 and step < warmup_steps:
        return (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def make_loader(
    ds: StrokeDataset, batch_size: int, shuffle: bool, seed: int, num_workers: int
) -> tuple[DataLoader, BucketBatchSampler]:
    sampler = BucketBatchSampler(ds.lengths(), batch_size, shuffle=shuffle, seed=seed)
    loader = DataLoader(
        ds,
        batch_sampler=sampler,
        collate_fn=collate,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
    )
    return loader, sampler


@torch.no_grad()
def evaluate_loader(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    correct1 = correct3 = total = 0
    loss_sum = 0.0
    for strokes, mask, labels in loader:
        strokes, mask, labels = strokes.to(device), mask.to(device), labels.to(device)
        logits = model(strokes, mask)
        loss_sum += nn.functional.cross_entropy(logits, labels, reduction="sum").item()
        top3 = logits.topk(3, dim=-1).indices
        correct1 += (top3[:, 0] == labels).sum().item()
        correct3 += (top3 == labels[:, None]).any(dim=-1).sum().item()
        total += labels.numel()
    return {"loss": loss_sum / total, "top1": correct1 / total, "top3": correct3 / total}


def build_model(config: RunConfig, n_classes: int, max_len: int) -> StrokeTransformer:
    return StrokeTransformer(
        ModelConfig(n_classes=n_classes, max_len=max_len, **asdict(config.model))
    )


def save_checkpoint(
    path: Path,
    model: StrokeTransformer,
    config: RunConfig,
    preprocess: PreprocessConfig,
    categories: list[str],
    epoch: int,
    metrics: dict[str, float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "model_config": asdict(model.config),
            "run_config": asdict(config),
            "preprocess": asdict(preprocess),
            "categories": categories,
            "epoch": epoch,
            "metrics": metrics,
        },
        path,
    )


def load_model(path: Path, device: torch.device | str = "cpu") -> tuple[StrokeTransformer, dict]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = StrokeTransformer(ModelConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()
    return model, checkpoint


def train(
    config: RunConfig,
    train_set: DrawingSet,
    val_set: DrawingSet,
    preprocess: PreprocessConfig,
    checkpoint_dir: Path = CHECKPOINT_DIR,
    log_dir: Path | None = None,
    device: torch.device | None = None,
    verbose: bool = True,
) -> dict[str, float]:
    """Train and return the best validation metrics."""
    tc = config.train
    seed_everything(tc.seed)
    if tc.num_threads:
        torch.set_num_threads(tc.num_threads)
    device = device or pick_device()

    train_ds = StrokeDataset(train_set, preprocess, config.augment, seed=tc.seed)
    val_ds = StrokeDataset(val_set, preprocess, augment_config=None)
    train_loader, train_sampler = make_loader(
        train_ds, tc.batch_size, True, tc.seed, tc.num_workers
    )
    # Validation has no augmentation, so worker processes would only cost memory.
    val_loader, _ = make_loader(val_ds, tc.batch_size * 2, False, tc.seed, num_workers=0)

    model = build_model(config, len(train_set.categories), preprocess.max_len).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=tc.lr, weight_decay=tc.weight_decay)
    steps_per_epoch = len(train_sampler)
    total_steps = steps_per_epoch * tc.epochs
    warmup_steps = int(tc.warmup_epochs * steps_per_epoch)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: warmup_cosine(step, warmup_steps, total_steps)
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=tc.label_smoothing)
    writer = SummaryWriter(log_dir) if log_dir else None

    if verbose:
        print(
            f"device={device}  params={count_parameters(model):,}  train={len(train_set):,}  "
            f"val={len(val_set):,}  steps/epoch={steps_per_epoch}  offset_scale="
            f"{preprocess.offset_scale:.3f}"
        )

    best: dict[str, float] = {"top1": -1.0}
    step = 0
    start = time.perf_counter()
    for epoch in range(1, tc.epochs + 1):
        model.train()
        train_sampler.set_epoch(epoch)
        epoch_start = time.perf_counter()
        loss_sum = 0.0
        correct = seen = 0
        for strokes, mask, labels in train_loader:
            strokes, mask, labels = strokes.to(device), mask.to(device), labels.to(device)
            logits = model(strokes, mask)
            loss = criterion(logits, labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if tc.grad_clip:
                nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip)
            optimizer.step()
            scheduler.step()
            step += 1
            loss_sum += loss.item() * labels.numel()
            correct += (logits.argmax(-1) == labels).sum().item()
            seen += labels.numel()
            if writer and step % 50 == 0:
                writer.add_scalar("train/loss_step", loss.item(), step)
                writer.add_scalar("train/lr", scheduler.get_last_lr()[0], step)

        val = evaluate_loader(model, val_loader, device)
        epoch_time = time.perf_counter() - epoch_start
        metrics = {
            "epoch": float(epoch),
            "train_loss": loss_sum / seen,
            "train_top1_augmented": correct / seen,
            "val_loss": val["loss"],
            "val_top1": val["top1"],
            "val_top3": val["top3"],
            "epoch_seconds": epoch_time,
        }
        if writer:
            for name, value in metrics.items():
                if name != "epoch":
                    writer.add_scalar(f"epoch/{name}", value, epoch)
        is_best = val["top1"] > best["top1"]
        if is_best:
            best = {"epoch": float(epoch), **val}
            save_checkpoint(
                checkpoint_dir / "best.pt",
                model,
                config,
                preprocess,
                train_set.categories,
                epoch,
                metrics,
            )
        save_checkpoint(
            checkpoint_dir / "last.pt",
            model,
            config,
            preprocess,
            train_set.categories,
            epoch,
            metrics,
        )
        if verbose:
            remaining = (time.perf_counter() - start) / epoch * (tc.epochs - epoch)
            print(
                f"epoch {epoch:>2}/{tc.epochs}  loss {metrics['train_loss']:.3f}  "
                f"val top1 {val['top1']:.3f} top3 {val['top3']:.3f}  "
                f"{epoch_time / 60:.1f} min  (~{remaining / 60:.0f} min left)"
                + ("  *best" if is_best else "")
            )
    if writer:
        writer.close()
    return best


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train the stroke classifier.")
    parser.add_argument("--config", type=Path, default=TRAIN_CONFIG)
    parser.add_argument("--quick", action="store_true", help="small sanity run")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--train-per-class", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--num-threads", type=int)
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--checkpoint-dir", type=Path, default=CHECKPOINT_DIR)
    args = parser.parse_args(argv)

    config = load_run_config(args.config, quick=args.quick)
    for name, section, attr in [
        ("epochs", config.train, "epochs"),
        ("train_per_class", config.data, "train_per_class"),
        ("batch_size", config.train, "batch_size"),
        ("num_workers", config.train, "num_workers"),
        ("num_threads", config.train, "num_threads"),
    ]:
        value = getattr(args, name)
        if value is not None:
            setattr(section, attr, value)

    preprocess = load_preprocess_config(args.processed_dir)
    train_set = load_simplified("train", args.processed_dir)
    val_set = load_simplified("val", args.processed_dir)
    if config.data.train_per_class:
        train_set = train_set.first_per_class(config.data.train_per_class)
    if config.data.val_per_class:
        val_set = val_set.first_per_class(config.data.val_per_class)

    run_name = time.strftime("%Y%m%d-%H%M%S") + ("-quick" if args.quick else "")
    best = train(config, train_set, val_set, preprocess, args.checkpoint_dir, RUNS_DIR / run_name)
    print(json.dumps({"best": best}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
