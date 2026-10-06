from types import SimpleNamespace

import pymupdf
import pytest

from raglaw.ingest.table import iter_rows, split_sides

TABLE = SimpleNamespace(bbox=(0, 0, 200, 100))


def row(*cells):
    return SimpleNamespace(cells=list(cells))


@pytest.mark.unit
def test_cells_split_at_the_table_midpoint():
    en, ar = split_sides(TABLE, row((0, 0, 100, 20), (100, 0, 200, 20)))

    assert en == pymupdf.Rect(0, 0, 100, 20)
    assert ar == pymupdf.Rect(100, 0, 200, 20)


@pytest.mark.unit
def test_spurious_ruling_lines_are_unioned_into_one_side():
    en, ar = split_sides(
        TABLE, row((0, 0, 40, 20), (40, 0, 100, 20), (100, 0, 200, 20))
    )

    assert en == pymupdf.Rect(0, 0, 100, 20)
    assert ar == pymupdf.Rect(100, 0, 200, 20)


@pytest.mark.unit
def test_a_missing_side_is_none_not_dropped():
    en, ar = split_sides(TABLE, row((0, 0, 100, 20), None))

    assert en == pymupdf.Rect(0, 0, 100, 20)
    assert ar is None


@pytest.mark.smoke
def test_rows_carry_source_page_numbers(fixture_pdfs):
    with pymupdf.open(fixture_pdfs["pages_046_047"]) as doc:
        rows = list(iter_rows(doc[0], 46))

    assert {r.page for r in rows} == {46}
    assert [r.index for r in rows] == list(range(len(rows)))
    assert [r.is_last for r in rows] == [False] * (len(rows) - 1) + [True]
    assert all(r.has_both_sides for r in rows)
