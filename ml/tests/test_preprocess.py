import json
import math
from pathlib import Path

import numpy as np
import pytest

from doodle_ml.fixtures import PREPROCESS_FIXTURE, build_cases, write_preprocess_fixture
from doodle_ml.preprocess import (
    PreprocessConfig,
    encode,
    normalize,
    preprocess,
    rdp,
    resample_stroke,
    simplify_drawing,
    to_drawing,
    to_stroke3,
)

# normalize ----------------------------------------------------------------------------


def test_normalize_shifts_to_origin_and_scales_longest_side() -> None:
    out = normalize([[(10.0, 20.0), (60.0, 30.0)]])  # 50 wide, 10 high
    assert out == [[(0.0, 0.0), (255.0, 51.0)]]


def test_normalize_preserves_aspect_ratio_for_tall_drawings() -> None:
    out = normalize([[(0.0, 0.0), (10.0, 100.0)]])
    assert out[0][1] == pytest.approx((25.5, 255.0))


def test_normalize_zero_extent_uses_scale_one() -> None:
    assert normalize([[(7.0, 9.0)], [(7.0, 9.0)]]) == [[(0.0, 0.0)], [(0.0, 0.0)]]


def test_normalize_empty() -> None:
    assert normalize([]) == []


def test_to_drawing_drops_empty_strokes_and_accepts_arrays() -> None:
    drawing = to_drawing([np.array([[1, 2], [3, 4]], dtype=np.uint8), [], [[5, 6]]])
    assert drawing == [[(1.0, 2.0), (3.0, 4.0)], [(5.0, 6.0)]]


# resample -----------------------------------------------------------------------------


def test_resample_inserts_points_at_most_spacing_apart() -> None:
    out = resample_stroke([(0.0, 0.0), (3.5, 0.0)], spacing=1.0)
    assert out == [(0.0, 0.0), (0.875, 0.0), (1.75, 0.0), (2.625, 0.0), (3.5, 0.0)]


def test_resample_keeps_short_segments_and_drops_duplicates() -> None:
    assert resample_stroke([(0.0, 0.0), (0.5, 0.0), (0.5, 0.0)], 1.0) == [(0.0, 0.0), (0.5, 0.0)]


def test_resample_single_point_and_empty() -> None:
    assert resample_stroke([(1.0, 1.0)], 1.0) == [(1.0, 1.0)]
    assert resample_stroke([], 1.0) == []


@pytest.mark.parametrize("seed", range(5))
def test_resample_max_gap_property(seed: int) -> None:
    rng = np.random.default_rng(seed)
    stroke = [tuple(p) for p in rng.uniform(0, 255, (20, 2)).tolist()]
    out = resample_stroke(stroke, 1.0)
    gaps = [math.dist(a, b) for a, b in zip(out, out[1:], strict=False)]
    assert max(gaps) <= 1.0 + 1e-9


# rdp ----------------------------------------------------------------------------------


def test_rdp_removes_collinear_points() -> None:
    line = [(float(i), 0.0) for i in range(50)]
    assert rdp(line, 2.0) == [(0.0, 0.0), (49.0, 0.0)]


def test_rdp_keeps_corner() -> None:
    stroke = [(0.0, 0.0), (5.0, 0.0), (10.0, 0.0), (10.0, 5.0), (10.0, 10.0)]
    assert rdp(stroke, 2.0) == [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0)]


def test_rdp_threshold_is_strict() -> None:
    assert rdp([(0.0, 0.0), (5.0, 2.0), (10.0, 0.0)], 2.0) == [(0.0, 0.0), (10.0, 0.0)]
    assert len(rdp([(0.0, 0.0), (5.0, 2.001), (10.0, 0.0)], 2.0)) == 3


def test_rdp_closed_stroke_uses_point_distance() -> None:
    square = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0), (0.0, 0.0)]
    assert rdp(square, 2.0) == square


# stroke-3 / encode --------------------------------------------------------------------


def test_to_stroke3_offsets_and_pen_lifts() -> None:
    rows = to_stroke3([[(1.0, 2.0), (4.0, 6.0)], [(10.0, 10.0)]])
    assert rows == [(1.0, 2.0, 0.0), (3.0, 4.0, 1.0), (6.0, 4.0, 1.0)]


def test_encode_pads_scales_and_masks() -> None:
    config = PreprocessConfig(max_len=4, offset_scale=2.0)
    strokes, mask = encode([[(2.0, 4.0), (6.0, 4.0)]], config)
    assert strokes.shape == (4, 3) and strokes.dtype == np.float32
    np.testing.assert_array_equal(strokes[:2], [[1.0, 2.0, 0.0], [2.0, 0.0, 1.0]])
    np.testing.assert_array_equal(strokes[2:], 0.0)
    np.testing.assert_array_equal(mask, [1, 1, 0, 0])


def test_encode_truncates_to_max_len() -> None:
    drawing = [[(float(i), 0.0)] for i in range(10)]
    strokes, mask = encode(drawing, PreprocessConfig(max_len=3))
    assert mask.sum() == 3
    np.testing.assert_array_equal(strokes[:, 2], 1.0)


def test_preprocess_empty_drawing_is_all_zero() -> None:
    strokes, mask = preprocess([])
    assert strokes.shape == (200, 3) and not strokes.any() and not mask.any()


def test_preprocess_is_translation_and_scale_invariant() -> None:
    base = [[(0.0, 0.0), (40.0, 10.0), (80.0, 60.0)], [(10.0, 70.0), (70.0, 70.0)]]
    moved = [[(x * 3.0 + 500.0, y * 3.0 - 20.0) for x, y in s] for s in base]
    a, ma = preprocess(base)
    b, mb = preprocess(moved)
    np.testing.assert_array_equal(ma, mb)
    np.testing.assert_allclose(a, b, atol=1e-4)


def test_simplified_dataset_drawing_is_nearly_unchanged() -> None:
    drawing = [[(0.0, 0.0), (255.0, 0.0), (255.0, 120.0)], [(30.0, 200.0), (90.0, 240.0)]]
    assert simplify_drawing(drawing) == drawing


# golden fixtures (the cross-language contract) ----------------------------------------


def _load_fixture() -> dict:
    return json.loads(PREPROCESS_FIXTURE.read_text(encoding="utf-8"))


def test_fixture_file_is_up_to_date() -> None:
    """Fails if preprocess.py changed without regenerating the shared fixtures.
    Fix with: uv run python -m doodle_ml.fixtures"""
    fixture = _load_fixture()
    offset_scale = fixture["cases"][0]["config"]["offset_scale"]
    expected = build_cases(PreprocessConfig(offset_scale=offset_scale))
    assert json.loads(json.dumps(expected)) == fixture["cases"]


@pytest.mark.parametrize("case", _load_fixture()["cases"], ids=lambda c: c["name"])
def test_preprocess_matches_fixture(case: dict) -> None:
    config = PreprocessConfig(**case["config"])
    strokes, mask = preprocess(case["input"], config)
    length = case["expected"]["length"]
    assert int(mask.sum()) == length
    expected_rows = np.asarray(case["expected"]["rows"], dtype=np.float64).reshape(-1, 3)
    np.testing.assert_allclose(strokes[:length], expected_rows, atol=1e-5)
    assert not strokes[length:].any()


def test_write_fixture_roundtrip(tmp_path: Path) -> None:
    path = write_preprocess_fixture(tmp_path / "f.json", PreprocessConfig(offset_scale=3.0))
    data = json.loads(path.read_text())
    assert len(data["cases"]) >= 20
    assert {c["config"]["offset_scale"] for c in data["cases"]} == {3.0, 2.5}
