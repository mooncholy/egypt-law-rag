"""The committed PDF excerpts that the smoke tests run on (made by make_fixtures.py)."""

import re

import pymupdf
import pytest

pytestmark = pytest.mark.unit

MAX_BYTES = 1024 * 1024  # the pre-commit large-file limit


def test_every_excerpt_is_committed(fixture_pdfs):
    missing = [name for name, path in fixture_pdfs.items() if not path.exists()]

    assert missing == [], "run `uv run python scripts/make_fixtures.py`"


def test_excerpts_hold_the_pages_their_names_promise(fixture_pdfs):
    for name, path in fixture_pdfs.items():
        pages = [int(n) for n in re.findall(r"\d+", name)]  # page_081, pages_007_009
        expected = pages[-1] - pages[0] + 1

        assert pymupdf.open(path).page_count == expected, name


def test_excerpts_fit_the_commit_size_limit(fixture_pdfs):
    sizes = {name: path.stat().st_size for name, path in fixture_pdfs.items()}

    assert all(size <= MAX_BYTES for size in sizes.values()), sizes
