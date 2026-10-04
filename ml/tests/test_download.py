import json
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest
import requests

from conftest import make_ndjson_lines
from doodle_ml.download import (
    DownloadConfig,
    InsufficientDataError,
    category_url,
    download_all,
    download_category,
    iter_recognized,
    raw_path,
    sample_lines,
)

FAST = DownloadConfig(per_class=20, pool_size=40, retry_backoff_s=0.0)


def test_category_url_encodes_spaces() -> None:
    assert category_url("ice cream").endswith("/simplified/ice%20cream.ndjson")


def test_iter_recognized_filters_and_compacts() -> None:
    lines = make_ndjson_lines("cat", 8)  # key_ids 3 and 7 are unrecognised
    kept = [json.loads(line) for line in iter_recognized(lines)]
    assert len(kept) == 6
    assert all(set(r) == {"key_id", "word", "drawing"} for r in kept)
    assert {r["key_id"] for r in kept}.isdisjoint({"3", "7"})


def test_iter_recognized_accepts_bytes_and_skips_blank_lines() -> None:
    lines = [b"", *(line.encode() for line in make_ndjson_lines("cat", 3, unrecognized_every=99))]
    assert len(list(iter_recognized(lines))) == 3


def test_sample_lines_is_deterministic_and_ordered() -> None:
    pool = [f"line{i:03d}" for i in range(100)]
    a = sample_lines(pool, 10, seed=1)
    assert a == sample_lines(pool, 10, seed=1)
    assert a != sample_lines(pool, 10, seed=2)
    assert a == sorted(a) and len(set(a)) == 10


def test_sample_lines_raises_when_pool_too_small() -> None:
    with pytest.raises(InsufficientDataError):
        sample_lines(["a", "b"], 3, seed=0)


def test_download_category_writes_sample(fake_source_factory, raw_dir: Path) -> None:
    source = fake_source_factory({"ice cream": make_ndjson_lines("ice cream", 100)})
    path, downloaded = download_category("ice cream", raw_dir, FAST, source)
    assert downloaded
    assert path == raw_path("ice cream", raw_dir)
    assert path.name == "ice_cream.ndjson"
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(records) == FAST.per_class
    assert {r["word"] for r in records} == {"ice cream"}
    assert not list(raw_dir.glob("*.part"))


def test_download_category_is_deterministic(fake_source_factory, tmp_path: Path) -> None:
    lines = {"cat": make_ndjson_lines("cat", 100)}
    a, _ = download_category("cat", tmp_path / "a", FAST, fake_source_factory(lines))
    b, _ = download_category("cat", tmp_path / "b", FAST, fake_source_factory(lines))
    assert a.read_text() == b.read_text()


def test_download_category_skips_existing_file(fake_source_factory, raw_dir: Path) -> None:
    raw_dir.mkdir(parents=True)
    raw_path("cat", raw_dir).write_text("x\n" * FAST.per_class)
    source = fake_source_factory({"cat": make_ndjson_lines("cat", 100)})
    _, downloaded = download_category("cat", raw_dir, FAST, source)
    assert not downloaded
    assert source.calls == []


def test_download_category_redownloads_too_small_file(fake_source_factory, raw_dir: Path) -> None:
    raw_dir.mkdir(parents=True)
    path = raw_path("cat", raw_dir)
    path.write_text("x\n" * (FAST.per_class - 1))  # e.g. left over from a quick test run
    source = fake_source_factory({"cat": make_ndjson_lines("cat", 100)})
    _, downloaded = download_category("cat", raw_dir, FAST, source)
    assert downloaded
    assert len(path.read_text().splitlines()) == FAST.per_class


def test_download_category_stops_reading_after_pool_is_full(raw_dir: Path) -> None:
    consumed = 0

    def counting_lines():
        nonlocal consumed
        for line in make_ndjson_lines("cat", 1000, unrecognized_every=10**9):
            consumed += 1
            yield line.encode()

    @contextmanager
    def source(url: str) -> Iterator[Iterator[bytes]]:
        yield counting_lines()

    download_category("cat", raw_dir, DownloadConfig(per_class=5, pool_size=10), source)
    assert consumed == 10


def test_download_category_retries_on_network_error(fake_source_factory, raw_dir: Path) -> None:
    source = fake_source_factory({"cat": make_ndjson_lines("cat", 100)}, fail_times=2)
    _, downloaded = download_category("cat", raw_dir, FAST, source)
    assert downloaded
    assert len(source.calls) == 3


def test_download_category_gives_up_after_max_retries(fake_source_factory, raw_dir: Path) -> None:
    source = fake_source_factory({"cat": make_ndjson_lines("cat", 100)}, fail_times=5)
    with pytest.raises(requests.ConnectionError):
        download_category("cat", raw_dir, FAST, source)
    assert not raw_dir.exists() or not list(raw_dir.iterdir())


def test_download_all_reports_failures_and_continues(fake_source_factory, raw_dir: Path) -> None:
    source = fake_source_factory(
        {"cat": make_ndjson_lines("cat", 100), "dog": make_ndjson_lines("dog", 10)}
    )
    failures = download_all(["dog", "cat"], raw_dir, FAST, source)
    assert set(failures) == {"dog"}
    assert "InsufficientDataError" in failures["dog"]
    assert raw_path("cat", raw_dir).exists()
