"""Project paths, constants and the category list."""

from __future__ import annotations

from pathlib import Path

ML_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ML_ROOT / "configs"
DATA_DIR = ML_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
CATEGORIES_FILE = CONFIG_DIR / "categories.txt"

SEED = 42


def load_categories(path: Path = CATEGORIES_FILE) -> list[str]:
    """Read one category per line, ignoring blank lines and ``#`` comments."""
    lines = (line.strip() for line in path.read_text(encoding="utf-8").splitlines())
    categories = [line for line in lines if line and not line.startswith("#")]
    duplicates = sorted({c for c in categories if categories.count(c) > 1})
    if duplicates:
        raise ValueError(f"Duplicate categories in {path}: {duplicates}")
    if not categories:
        raise ValueError(f"No categories found in {path}")
    return categories


def category_slug(category: str) -> str:
    """File-system friendly name: ``"ice cream"`` -> ``"ice_cream"``."""
    return category.replace(" ", "_")
