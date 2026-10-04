import json
from pathlib import Path

import numpy as np
import pytest

from doodle_ml.cache import (
    build_cache,
    compute_offset_scale,
    load_preprocess_config,
    load_simplified,
    simplify_set,
    stroke3_offsets,
)
from doodle_ml.dataset import DrawingSet
from doodle_ml.preprocess import simplify_drawing
from helpers import synthetic_set


def test_stroke3_offsets_restart_at_origin_per_drawing() -> None:
    a = [np.array([[1.0, 2.0], [4.0, 6.0]]), np.array([[10.0, 10.0]])]
    b = [np.array([[3.0, 3.0]])]
    ds = DrawingSet.from_drawings([a, b], [0, 0], ["x"], np.float32)
    np.testing.assert_allclose(stroke3_offsets(ds), [[1, 2], [3, 4], [6, 4], [3, 3]])


def test_compute_offset_scale_is_pooled_std() -> None:
    ds = DrawingSet.from_drawings([[np.array([[0.0, 0.0], [2.0, 0.0]])]], [0], ["x"], np.float32)
    assert compute_offset_scale(ds) == pytest.approx(np.std([0, 0, 2, 0]))


def test_simplify_set_matches_reference(tmp_path: Path) -> None:
    raw = synthetic_set(3)
    simplified = simplify_set(raw, workers=1)
    assert simplified.points.dtype == np.float32
    for i in range(len(raw)):
        expected = simplify_drawing(raw.strokes(i))
        got = simplified.strokes(i)
        assert [len(s) for s in got] == [len(s) for s in expected]
        for g, e in zip(got, expected, strict=True):
            np.testing.assert_allclose(g, np.array(e), atol=1e-4)


def test_build_cache_end_to_end(tmp_path: Path) -> None:
    for split, seed in [("train", 0), ("val", 1), ("test", 2)]:
        synthetic_set(4, seed=seed).save(tmp_path / f"{split}.npz")
    config = build_cache(tmp_path, workers=1)
    assert config.offset_scale > 1.0
    assert load_preprocess_config(tmp_path) == config
    assert len(load_simplified("val", tmp_path)) == 12
    meta = json.loads((tmp_path / "preprocess_meta.json").read_text())
    assert meta["preprocess"]["max_len"] == 200


def test_load_simplified_missing_gives_helpful_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="doodle_ml.cache"):
        load_simplified("train", tmp_path)
