"""Synthetic Quick, Draw!-style data so tests never touch the network."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest


def make_record(word: str, key_id: int, recognized: bool, rng: np.random.Generator) -> dict:
    n_strokes = int(rng.integers(1, 6))
    drawing = []
    for _ in range(n_strokes):
        n_points = int(rng.integers(1, 15))
        xs = rng.integers(0, 256, n_points).tolist()
        ys = rng.integers(0, 256, n_points).tolist()
        drawing.append([xs, ys])
    return {
        "word": word,
        "countrycode": "NL",
        "timestamp": "2017-03-01 00:00:00.00000 UTC",
        "recognized": recognized,
        "key_id": str(key_id),
        "drawing": drawing,
    }


def make_ndjson_lines(word: str, n: int, seed: int = 0, unrecognized_every: int = 4) -> list[str]:
    """``n`` records; every ``unrecognized_every``-th one (key_id 3, 7, ... for 4) is
    unrecognised."""
    rng = np.random.default_rng(seed)
    return [
        json.dumps(make_record(word, i, recognized=((i + 1) % unrecognized_every != 0), rng=rng))
        for i in range(n)
    ]


class FakeSource:
    """Stands in for the HTTP line stream; records which URLs were requested."""

    def __init__(self, lines_by_word: dict[str, list[str]], fail_times: int = 0) -> None:
        self.lines_by_word = lines_by_word
        self.fail_times = fail_times
        self.calls: list[str] = []

    @contextmanager
    def __call__(self, url: str) -> Iterator[Iterable[bytes]]:
        import requests

        self.calls.append(url)
        if self.fail_times > 0:
            self.fail_times -= 1
            raise requests.ConnectionError("simulated network failure")
        word = next(w for w in self.lines_by_word if url.endswith(_encoded(w) + ".ndjson"))
        yield (line.encode("utf-8") for line in self.lines_by_word[word])


def _encoded(word: str) -> str:
    from urllib.parse import quote

    return quote(word)


@pytest.fixture
def fake_source_factory() -> Callable[..., FakeSource]:
    return FakeSource


@pytest.fixture
def raw_dir(tmp_path: Path) -> Path:
    return tmp_path / "raw"
