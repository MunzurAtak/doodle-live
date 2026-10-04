import json
from pathlib import Path

import numpy as np
import pytest
import torch

from doodle_ml.cache import simplify_set
from doodle_ml.evaluate import confusion_top_pairs, run, topk_accuracy
from doodle_ml.preprocess import PreprocessConfig
from doodle_ml.train import (
    ModelSection,
    RunConfig,
    TrainSection,
    load_model,
    load_run_config,
    train,
    warmup_cosine,
)
from helpers import synthetic_set

TINY_MODEL = ModelSection(d_model=32, n_layers=1, n_heads=2, d_ff=64, dropout=0.0)


def test_warmup_cosine_schedule() -> None:
    assert warmup_cosine(0, 10, 100) == pytest.approx(0.1)
    assert warmup_cosine(9, 10, 100) == pytest.approx(1.0)
    assert warmup_cosine(10, 10, 100) == pytest.approx(1.0)
    assert warmup_cosine(55, 10, 100) == pytest.approx(0.5)
    assert warmup_cosine(100, 10, 100) == pytest.approx(0.0)


def test_load_run_config_quick_overrides(tmp_path: Path) -> None:
    path = tmp_path / "t.yaml"
    path.write_text("train:\n  epochs: 15\nquick:\n  train:\n    epochs: 2\n")
    assert load_run_config(path).train.epochs == 15
    assert load_run_config(path, quick=True).train.epochs == 2


def test_repo_train_config_parses() -> None:
    config = load_run_config()
    assert config.train.epochs == 15 and config.model.d_model == 128
    assert load_run_config(quick=True).data.train_per_class == 300


def test_overfits_tiny_dataset_and_checkpoint_roundtrip(tmp_path: Path) -> None:
    """1-batch overfit smoke test: the model must be able to learn 3 trivial shapes."""
    train_set = simplify_set(synthetic_set(16, seed=0), workers=1)
    preprocess = PreprocessConfig(offset_scale=50.0)
    config = RunConfig(
        model=TINY_MODEL,
        train=TrainSection(epochs=40, batch_size=64, lr=3e-3, warmup_epochs=0, num_workers=0),
    )
    config.augment = type(config.augment)(prefix_prob=0.0, rotate_deg=0.0)
    best = train(
        config,
        train_set,
        train_set,
        preprocess,
        tmp_path,
        device=torch.device("cpu"),
        verbose=False,
    )
    assert best["top1"] == 1.0

    model, ckpt = load_model(tmp_path / "best.pt")
    assert ckpt["categories"] == ["hline", "vline", "square"]
    assert ckpt["preprocess"]["offset_scale"] == 50.0
    assert not model.training


def test_topk_and_confusion_pairs() -> None:
    probs = np.array([[0.7, 0.2, 0.1], [0.1, 0.3, 0.6], [0.2, 0.5, 0.3], [0.6, 0.3, 0.1]])
    labels = np.array([0, 1, 1, 1])
    assert topk_accuracy(probs, labels, 1) == 0.5
    assert topk_accuracy(probs, labels, 2) == 1.0
    pairs = confusion_top_pairs(probs, labels, ["a", "b", "c"])
    assert {(p["true"], p["predicted"]) for p in pairs} == {("b", "c"), ("b", "a")}
    assert all(p["rate"] == pytest.approx(1 / 3) for p in pairs)


def test_evaluate_run_writes_reports(tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    raw_test = synthetic_set(6, seed=5)
    raw_test.save(processed / "test.npz")
    simplify_set(raw_test, workers=1).save(processed / "simplified_test.npz")
    train_set = simplify_set(synthetic_set(8), workers=1)
    config = RunConfig(model=TINY_MODEL, train=TrainSection(epochs=1, batch_size=16, num_workers=0))
    train(
        config,
        train_set,
        train_set,
        PreprocessConfig(offset_scale=50.0),
        tmp_path / "ckpt",
        device=torch.device("cpu"),
        verbose=False,
    )

    reports = tmp_path / "reports"
    run(tmp_path / "ckpt" / "best.pt", processed, reports, per_class=None, max_strokes=3)
    metrics = json.loads((reports / "metrics.json").read_text())
    assert metrics["n_test"] == 18 and 0.0 <= metrics["top1"] <= metrics["top3"] <= 1.0
    curve = json.loads((reports / "accuracy_vs_strokes.json").read_text())
    assert [r["strokes"] for r in curve] == [1, 2, 3]
    assert curve[-1]["share_complete"] == 1.0
    assert (reports / "accuracy_vs_strokes.png").stat().st_size > 1000
