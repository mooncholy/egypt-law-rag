"""Find each page's table and split its rows into an English and an Arabic side.

Every page of the source is one ruled two-column table, English left and Arabic
right (P3). This module is the only place that turns a page into rows, so the
measurements in ``measure`` and the text in ``extract`` see exactly the same
rows (R1, R2).
"""

from collections.abc import Iterator
from dataclasses import dataclass

import pymupdf

# The default "lines" strategy reads the yellow highlight fills on 9 pages as
# ruling lines and finds false rows and columns (P6); "lines_strict" ignores
# filled rectangles.
TABLE_STRATEGY = "lines_strict"


@dataclass(frozen=True)
class TableRow:
    """One detected row: where it is, and the rectangle of each language's side.

    A side is ``None`` when the row has no cell on that half of the table.
    """

    page: int
    index: int
    is_last: bool
    table_bbox: tuple[float, float, float, float]
    en_rect: pymupdf.Rect | None
    ar_rect: pymupdf.Rect | None

    @property
    def has_both_sides(self) -> bool:
        return self.en_rect is not None and self.ar_rect is not None


def find_tables(page: pymupdf.Page) -> list[pymupdf.table.Table]:
    """
    Detect the page's tables with the strategy the whole pipeline uses.

    returns:
    - tables (list[Table]): every table ``lines_strict`` finds; one per page
      in the source
    """
    return page.find_tables(strategy=TABLE_STRATEGY).tables


def _union(rects: list[pymupdf.Rect]) -> pymupdf.Rect | None:
    if not rects:
        return None
    union = pymupdf.Rect(rects[0])
    for rect in rects[1:]:
        union |= rect
    return union


def split_sides(
    table: pymupdf.table.Table, row: pymupdf.table.TableRow
) -> tuple[pymupdf.Rect | None, pymupdf.Rect | None]:
    """
    Split a row's cells at the table's horizontal midpoint.

    Some pages have spurious ruling lines that split one side into several
    cells, so each language is the union of the cells on its half.

    returns:
    - sides (tuple[Rect | None, Rect | None]): the English and Arabic
      rectangles, ``None`` for a side with no cell
    """
    cells = [pymupdf.Rect(c) for c in row.cells if c]
    mid = (table.bbox[0] + table.bbox[2]) / 2
    left = [c for c in cells if (c.x0 + c.x1) / 2 < mid]
    right = [c for c in cells if (c.x0 + c.x1) / 2 >= mid]
    return _union(left), _union(right)


def iter_rows(page: pymupdf.Page, page_number: int) -> Iterator[TableRow]:
    """
    Yield every detected row of the page's tables, in reading order.

    Rows missing a side are yielded too: dropping them here would be a silent
    path, and the caller decides how to report them.

    returns:
    - rows (Iterator[TableRow]): one per detected row, with ``page`` set to
      ``page_number`` (the source page number, 1-based)
    """
    for table in find_tables(page):
        last = len(table.rows) - 1
        for index, row in enumerate(table.rows):
            en_rect, ar_rect = split_sides(table, row)
            yield TableRow(
                page=page_number,
                index=index,
                is_last=index == last,
                table_bbox=tuple(table.bbox),
                en_rect=en_rect,
                ar_rect=ar_rect,
            )
