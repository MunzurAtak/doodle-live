import numpy as np
import pytest

from doodle_ml.augment import (
    AugmentConfig,
    affine,
    augment,
    encode_rows,
    normalize_arrays,
    random_prefix,
)
from doodle_ml.preprocess import PreprocessConfig, encode, normalize

DRAWING = [
    np.array([[0.0, 0.0], [40.0, 10.0], [80.0, 60.0]]),
    np.array([[10.0, 70.0], [70.0, 70.0]]),
    np.array([[5.0, 5.0]]),
]


def as_tuples(drawing: list[np.ndarray]) -> list[list[tuple[float, float]]]:
    return [[(float(x), float(y)) for x, y in s] for s in drawing]


def test_encode_rows_matches_reference_encoder() -> None:
    config = PreprocessConfig(offset_scale=7.5)
    reference, mask = encode(as_tuples(DRAWING), config)
    rows = encode_rows(DRAWING, config.offset_scale, config.max_len)
    assert rows.dtype == np.float32
    np.testing.assert_allclose(rows, reference[: int(mask.sum())], atol=1e-6)


def test_encode_rows_truncates_and_handles_empty() -> None:
    assert encode_rows(DRAWING, 1.0, 4).shape == (4, 3)
    assert encode_rows([], 1.0, 200).shape == (0, 3)


def test_normalize_arrays_matches_reference() -> None:
    drawing = [s * 3.0 + 17.0 for s in DRAWING]
    out = normalize_arrays(drawing)
    reference = normalize(as_tuples(drawing))
    for a, b in zip(out, reference, strict=True):
        np.testing.assert_allclose(a, np.array(b), atol=1e-9)


def test_random_prefix_keeps_first_strokes() -> None:
    rng = np.random.default_rng(0)
    lengths = {len(random_prefix(DRAWING, rng, prob=1.0)) for _ in range(200)}
    assert lengths == {1, 2, 3}
    assert random_prefix(DRAWING, rng, prob=0.0) is DRAWING
    assert random_prefix(DRAWING[:1], rng, prob=1.0) == DRAWING[:1]


def test_affine_identity_when_ranges_are_trivial() -> None:
    config = AugmentConfig(scale_min=1.0, scale_max=1.0, rotate_deg=0.0)
    out = affine(DRAWING, np.random.default_rng(0), config)
    for a, b in zip(out, DRAWING, strict=True):
        np.testing.assert_allclose(a, b)


@pytest.mark.parametrize("seed", range(5))
def test_augment_output_is_normalized(seed: int) -> None:
    out = augment(DRAWING, np.random.default_rng(seed), AugmentConfig())
    points = np.concatenate(out)
    assert points.min(axis=0) == pytest.approx([0.0, 0.0], abs=1e-9)
    assert (points.max(axis=0) - points.min(axis=0)).max() == pytest.approx(255.0)


def test_augment_is_deterministic_given_rng_seed() -> None:
    a = augment(DRAWING, np.random.default_rng(3), AugmentConfig())
    b = augment(DRAWING, np.random.default_rng(3), AugmentConfig())
    for x, y in zip(a, b, strict=True):
        np.testing.assert_array_equal(x, y)
