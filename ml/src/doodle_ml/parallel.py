"""Small helper to map a function over chunks of work with a process pool."""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from typing import TypeVar

from tqdm import tqdm

T = TypeVar("T")
R = TypeVar("R")


def default_workers() -> int:
    return max(1, (os.cpu_count() or 2) - 1)


def chunked_map(
    fn: Callable[[Sequence[T]], list[R]],
    items: Sequence[T],
    workers: int | None = None,
    chunk_size: int = 2000,
    desc: str | None = None,
) -> list[R]:
    """Apply ``fn`` (which takes a chunk and returns a list) to chunks of ``items`` and
    concatenate the results in order. ``fn`` must be a top-level function so it can be
    pickled (Windows uses spawn). ``workers=1`` runs in-process."""
    chunks = [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]
    workers = workers or default_workers()
    results: list[R] = []
    if workers == 1 or len(chunks) <= 1:
        for chunk in tqdm(chunks, desc=desc, unit="chunk", disable=desc is None):
            results.extend(fn(chunk))
        return results
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for out in tqdm(
            pool.map(fn, chunks), total=len(chunks), desc=desc, unit="chunk", disable=desc is None
        ):
            results.extend(out)
    return results
