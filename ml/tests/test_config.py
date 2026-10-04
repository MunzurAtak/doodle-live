from pathlib import Path

import pytest

from doodle_ml.config import category_slug, load_categories


def test_default_categories_are_the_50_from_the_spec() -> None:
    categories = load_categories()
    assert len(categories) == 50
    assert categories[0] == "airplane"
    assert "ice cream" in categories
    assert categories[-1] == "wine glass"


def test_load_categories_ignores_blanks_and_comments(tmp_path: Path) -> None:
    path = tmp_path / "cats.txt"
    path.write_text("# header\ncat\n\n  dog  \n", encoding="utf-8")
    assert load_categories(path) == ["cat", "dog"]


def test_load_categories_rejects_duplicates(tmp_path: Path) -> None:
    path = tmp_path / "cats.txt"
    path.write_text("cat\ndog\ncat\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate"):
        load_categories(path)


def test_category_slug() -> None:
    assert category_slug("ice cream") == "ice_cream"
    assert category_slug("cat") == "cat"
