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


@pytest.mark.unit
def test_a_ruled_page_yields_every_row_in_order(synthetic_pdf):
    with pymupdf.open(synthetic_pdf) as doc:
        rows = list(iter_rows(doc[0], 1))

    assert len(rows) == 4
    assert {r.page for r in rows} == {1}
    assert [r.index for r in rows] == [0, 1, 2, 3]
    assert [r.is_last for r in rows] == [False, False, False, True]
    assert all(r.has_both_sides for r in rows)
