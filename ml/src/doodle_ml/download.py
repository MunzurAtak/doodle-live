"""Download a fixed-size sample of recognised Quick, Draw! drawings per category.

For each category we stream the "simplified" ndjson file from Google's public bucket,
keep only drawings with ``recognized == true`` and stop once a pool of ``pool_size``
recognised drawings is collected. From that pool a seeded random sample of
``per_class`` drawings is written to ``ml/data/raw/<category>.ndjson``.

Stopping early keeps the download to a few minutes instead of several GB, at the cost
of sampling from the start of each file rather than the whole file.

The download is resumable per category: complete files are skipped, and a file is only
moved into place once complete (written to ``.part`` first). After downloading, the
train/val/test splits are built (see :mod:`doodle_ml.dataset`).

Usage (from ``ml/``)::

    uv run python -m doodle_ml.download
    uv run python -m doodle_ml.download --limit-categories 3 --per-class 1200 --no-build
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import zlib
from collections.abc import Callable, Iterable, Iterator
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from urllib.parse import quote

import numpy as np
import requests
from tqdm import tqdm

from doodle_ml.config import RAW_DIR, SEED, category_slug, load_categories

BASE_URL = "https://storage.googleapis.com/quickdraw_dataset/full/simplified/"

LineSource = Callable[[str], AbstractContextManager[Iterable[bytes | str]]]


class InsufficientDataError(RuntimeError):
    """Raised when a category has fewer recognised drawings than requested."""


@dataclass(frozen=True)
class DownloadConfig:
    per_class: int = 6000
    pool_size: int = 20000
    seed: int = SEED
    timeout_s: float = 30.0
    max_retries: int = 3
    retry_backoff_s: float = 2.0

    def __post_init__(self) -> None:
        if self.per_class <= 0:
            raise ValueError("per_class must be positive")
        if self.pool_size < self.per_class:
            raise ValueError("pool_size must be >= per_class")


def category_url(category: str) -> str:
    return f"{BASE_URL}{quote(category)}.ndjson"


def raw_path(category: str, raw_dir: Path = RAW_DIR) -> Path:
    return raw_dir / f"{category_slug(category)}.ndjson"


def category_seed(seed: int, category: str) -> int:
    """Stable per-category seed so categories don't share a sampling pattern."""
    return seed + zlib.crc32(category.encode("utf-8"))


@contextmanager
def http_lines(url: str, timeout_s: float = 30.0) -> Iterator[Iterable[bytes]]:
    """Stream the lines of a remote text file; the connection closes on exit."""
    with requests.get(url, stream=True, timeout=timeout_s) as response:
        response.raise_for_status()
        yield response.iter_lines(chunk_size=1 << 16)


def iter_recognized(lines: Iterable[bytes | str]) -> Iterator[str]:
    """Yield compact JSON lines (key_id, word, drawing) for recognised drawings only."""
    for line in lines:
        if not line:
            continue
        text = line.decode("utf-8") if isinstance(line, bytes) else line
        record = json.loads(text)
        if record.get("recognized") is not True:
            continue
        compact = {
            "key_id": record.get("key_id"),
            "word": record["word"],
            "drawing": record["drawing"],
        }
        yield json.dumps(compact, separators=(",", ":"))


def sample_lines(pool: list[str], n: int, seed: int) -> list[str]:
    """Seeded sample of ``n`` lines, kept in their original file order."""
    if len(pool) < n:
        raise InsufficientDataError(f"only {len(pool)} recognised drawings, need {n}")
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(len(pool), size=n, replace=False))
    return [pool[i] for i in indices]


def count_lines(path: Path) -> int:
    with path.open(encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def download_category(
    category: str,
    raw_dir: Path = RAW_DIR,
    config: DownloadConfig = DownloadConfig(),  # noqa: B008 (frozen dataclass, safe default)
    source: LineSource | None = None,
) -> tuple[Path, bool]:
    """Download one category. Returns ``(path, downloaded)``; ``downloaded`` is False
    when a file with at least ``per_class`` drawings already existed and was skipped
    (a smaller file, e.g. from a quick test run, is downloaded again)."""
    out = raw_path(category, raw_dir)
    if out.exists() and count_lines(out) >= config.per_class:
        return out, False

    def default_source(url: str) -> AbstractContextManager[Iterable[bytes]]:
        return http_lines(url, config.timeout_s)

    open_lines = source or default_source
    url = category_url(category)
    pool: list[str] = []
    for attempt in range(1, config.max_retries + 1):
        try:
            with open_lines(url) as lines:
                pool = list(islice(iter_recognized(lines), config.pool_size))
            break
        except requests.RequestException:
            if attempt == config.max_retries:
                raise
            time.sleep(config.retry_backoff_s * 2 ** (attempt - 1))

    sample = sample_lines(pool, config.per_class, category_seed(config.seed, category))

    raw_dir.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".part")
    tmp.write_text("\n".join(sample) + "\n", encoding="utf-8")
    tmp.replace(out)
    return out, True


def download_all(
    categories: list[str],
    raw_dir: Path = RAW_DIR,
    config: DownloadConfig = DownloadConfig(),  # noqa: B008
    source: LineSource | None = None,
) -> dict[str, str]:
    """Download every category, continuing past failures. Returns ``{category: error}``."""
    failures: dict[str, str] = {}
    progress = tqdm(categories, desc="Downloading", unit="cat")
    for category in progress:
        progress.set_postfix_str(category)
        try:
            download_category(category, raw_dir, config, source)
        except (requests.RequestException, InsufficientDataError, ValueError) as exc:
            failures[category] = f"{type(exc).__name__}: {exc}"
    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--per-class", type=int, default=DownloadConfig.per_class)
    parser.add_argument("--pool-size", type=int, default=DownloadConfig.pool_size)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument(
        "--limit-categories", type=int, default=None, help="only the first N (quick test)"
    )
    parser.add_argument(
        "--no-build", action="store_true", help="skip building train/val/test splits"
    )
    args = parser.parse_args(argv)

    categories = load_categories()
    if args.limit_categories is not None:
        categories = categories[: args.limit_categories]
    config = DownloadConfig(per_class=args.per_class, pool_size=args.pool_size, seed=args.seed)

    failures = download_all(categories, args.raw_dir, config)
    if failures:
        print(f"\n{len(failures)} categories failed (re-run to retry):", file=sys.stderr)
        for category, error in failures.items():
            print(f"  {category}: {error}", file=sys.stderr)
        return 1
    print(f"All {len(categories)} categories present in {args.raw_dir}")

    if not args.no_build:
        from doodle_ml import dataset

        build_args = ["--raw-dir", str(args.raw_dir), "--per-class", str(args.per_class)]
        return dataset.main(build_args, categories=categories)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
