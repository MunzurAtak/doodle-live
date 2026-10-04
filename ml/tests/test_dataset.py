import json
from pathlib import Path

import numpy as np
import pytest

from conftest import make_ndjson_lines
from doodle_ml.dataset import (
    DrawingSet,
    SplitSizes,
    build_splits,
    load_splits,
    main,
    parse_drawing,
    read_raw_category,
    save_splits,
    split_indices,
    summarize,
)

CATS = ["cat", "ice cream", "sun"]
SIZES = SplitSizes(train=10, val=2, test=2)


def write_raw(raw_dir: Path, n: int = 20) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    for seed, cat in enumerate(CATS):
        lines = make_ndjson_lines(cat, n, seed=seed, unrecognized_every=10**9)
        path = raw_dir / f"{cat.replace(' ', '_')}.ndjson"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# parse_drawing ------------------------------------------------------------------------


def test_parse_drawing_converts_to_point_arrays() -> None:
    strokes = parse_drawing([[[0, 10, 20], [5, 5, 5]], [[255], [0]]])
    assert len(strokes) == 2
    assert strokes[0].dtype == np.uint8
    np.testing.assert_array_equal(strokes[0], [[0, 5], [10, 5], [20, 5]])
    np.testing.assert_array_equal(strokes[1], [[255, 0]])


def test_parse_drawing_drops_empty_strokes() -> None:
    assert len(parse_drawing([[[], []], [[1], [2]]])) == 1


@pytest.mark.parametrize(
    "bad",
    [
        [[[1, 2], [3]]],  # x/y length mismatch
        [[[1, 2]]],  # missing y
        [[[0, 256], [0, 0]]],  # out of range
        [[[-1], [0]]],  # negative
    ],
)
def test_parse_drawing_rejects_malformed(bad: list) -> None:
    with pytest.raises(ValueError):
        parse_drawing(bad)


def test_read_raw_category_skips_bad_lines(tmp_path: Path) -> None:
    path = tmp_path / "x.ndjson"
    good = json.dumps({"word": "x", "drawing": [[[1, 2], [3, 4]]]})
    out_of_range = json.dumps({"word": "x", "drawing": [[[999], [0]]]})
    empty = json.dumps({"word": "x", "drawing": []})
    path.write_text("\n".join([good, out_of_range, empty, "", good]) + "\n")
    assert len(read_raw_category(path)) == 2


# DrawingSet ---------------------------------------------------------------------------


def sample_set() -> DrawingSet:
    a = [np.array([[0, 0], [1, 1]]), np.array([[5, 5]])]
    b = [np.array([[9, 9], [8, 8], [7, 7]])]
    return DrawingSet.from_drawings([a, b], [1, 0], ["cat", "dog"])


def test_drawing_set_returns_original_strokes() -> None:
    ds = sample_set()
    assert len(ds) == 2
    strokes, label = ds[0]
    assert label == 1
    assert [s.tolist() for s in strokes] == [[[0, 0], [1, 1]], [[5, 5]]]
    assert ds[1][0][0].tolist() == [[9, 9], [8, 8], [7, 7]]
    np.testing.assert_array_equal(ds.points_per_drawing(), [3, 3])


def test_drawing_set_save_load_roundtrip(tmp_path: Path) -> None:
    ds = sample_set()
    ds.save(tmp_path / "s.npz")
    loaded = DrawingSet.load(tmp_path / "s.npz")
    assert loaded.categories == ["cat", "dog"]
    for i in range(len(ds)):
        assert [s.tolist() for s in loaded[i][0]] == [s.tolist() for s in ds[i][0]]
        assert loaded[i][1] == ds[i][1]


def test_drawing_set_validates_consistency() -> None:
    ds = sample_set()
    with pytest.raises(ValueError):
        DrawingSet(ds.points[:-1], ds.stroke_lengths, ds.strokes_per_drawing, ds.labels, [])


# splitting ----------------------------------------------------------------------------


def test_split_sizes_from_total() -> None:
    assert SplitSizes.from_total(6000) == SplitSizes(5000, 500, 500)
    assert SplitSizes.from_total(1200) == SplitSizes(1000, 100, 100)
    with pytest.raises(ValueError):
        SplitSizes.from_total(5)


def test_split_indices_disjoint_and_deterministic() -> None:
    a = split_indices(20, SIZES, seed=3)
    b = split_indices(20, SIZES, seed=3)
    for split in a:
        np.testing.assert_array_equal(a[split], b[split])
    sets = [set(a[s].tolist()) for s in ("train", "val", "test")]
    assert [len(s) for s in sets] == [10, 2, 2]
    assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
    other = split_indices(20, SIZES, seed=4)
    assert not np.array_equal(a["train"], other["train"])


def test_split_indices_requires_enough_drawings() -> None:
    with pytest.raises(ValueError):
        split_indices(13, SIZES, seed=0)


def test_build_splits_end_to_end(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    write_raw(raw)
    sets = build_splits(CATS, raw, SIZES, seed=0)
    assert [len(sets[s]) for s in ("train", "val", "test")] == [30, 6, 6]
    for ds in sets.values():
        assert ds.categories == CATS
        counts = np.bincount(ds.labels, minlength=3)
        assert counts.min() == counts.max()

    out = tmp_path / "processed"
    save_splits(sets, out, SIZES, seed=0)
    reloaded = load_splits(out)
    assert len(reloaded["train"]) == 30
    meta = json.loads((out / "meta.json").read_text())
    assert meta["categories"] == CATS and meta["sizes"]["val"] == 2

    rebuilt = build_splits(CATS, raw, SIZES, seed=0)
    np.testing.assert_array_equal(rebuilt["test"].points, sets["test"].points)


def test_build_splits_errors_on_missing_category(tmp_path: Path) -> None:
    write_raw(tmp_path / "raw")
    with pytest.raises(FileNotFoundError):
        build_splits([*CATS, "dog"], tmp_path / "raw", SIZES, seed=0)


def test_build_splits_errors_when_category_too_small(tmp_path: Path) -> None:
    write_raw(tmp_path / "raw", n=12)
    with pytest.raises(ValueError, match="cat"):
        build_splits(CATS, tmp_path / "raw", SIZES, seed=0)


def test_summary_and_cli(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    raw, out = tmp_path / "raw", tmp_path / "processed"
    write_raw(raw, n=24)
    args = ["--raw-dir", str(raw), "--out-dir", str(out), "--per-class", "24"]
    assert main(args, categories=CATS) == 0
    printed = capsys.readouterr().out
    assert "train" in printed and "avg strokes" in printed

    assert main(["--out-dir", str(out), "--summary-only"]) == 0
    assert summarize(load_splits(out)).splitlines()[2].split()[1] == "60"
