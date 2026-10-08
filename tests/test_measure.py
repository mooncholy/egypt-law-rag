import collections
from pathlib import Path

import pymupdf
import pytest

from raglaw.ingest.measure import (
    PageFacts,
    RowFacts,
    SourceFacts,
    defect_counts,
    parse_ar_header,
)

pytestmark = pytest.mark.unit


def row_facts(en: str, ar: str = "", **overrides) -> RowFacts:
    fields = {
        "page": 1,
        "index": 0,
        "is_last": False,
        "table_bbox": (0, 0, 1, 1),
        "ar_rect": pymupdf.Rect(),
        "en_lines": en.splitlines(),
        "ar_lines": ar.splitlines(),
        "en_all_bold": False,
        "lam_alef_hits": [],
        "digit_order": collections.Counter(),
    }
    return RowFacts(**(fields | overrides))


def source(*rows: RowFacts) -> SourceFacts:
    page = PageFacts(
        number=1,
        fonts=set(),
        has_text=True,
        n_tables=1,
        col_counts=[2],
        n_rows=len(rows),
        highlights=0,
        default_strategy_rows=len(rows),
        text_outside_table="",
        rows=list(rows),
    )
    return SourceFacts(
        path=Path("x.pdf"),
        metadata={},
        page_count=1,
        pages=[page],
        struct={"marked": False, "tags": {}, "tr_per_page": {}, "multi_page_trs": 0},
    )


def test_row_kind_follows_the_first_english_line():
    assert row_facts("Article 12\nThe body").kind == "article"
    assert row_facts("The body\nas in Article 12").kind == "body"
    assert row_facts("SECTION I", en_all_bold=True).kind == "heading"


def test_defect_counts_count_each_defect_class():
    facts = source(
        row_facts("rticle 452", "م ادة\n٤٥٢"),
        row_facts("Article1022", "ما دة\n١٠٢٢"),
        row_facts("Article 277 If the option", "مادة\n٢٧٧"),
        row_facts(
            "Article 5",
            "ماد ة ( ٥ )",
            lam_alef_hits=["x", "y"],
            digit_order=collections.Counter(descending_x=3, ascending_x=1),
        ),
        row_facts("Not an article", "م ادة"),
    )

    assert defect_counts(facts) == {
        "lam_alef_signatures": 2,
        "rtl_digit_runs": 3,
        "header_typos": 2,
        "same_line_headers": 1,
        "spaced_mada_headers": 3,
    }


@pytest.mark.parametrize(
    ("lines", "digits"),
    [
        (["مادة", "١٤٧"], "١٤٧"),
        (["( مادة", "(١٤٧)"], "١٤٧"),
        (["م ادة ١٤٧"], "١٤٧"),
        (["المواد من"], None),
    ],
)
def test_arabic_header_number_is_read_across_lines(lines, digits):
    assert parse_ar_header(lines) == digits
