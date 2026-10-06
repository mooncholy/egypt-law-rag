import pytest
from pydantic import ValidationError

from raglaw.ingest.errata import (
    Erratum,
    ErratumNotFoundError,
    apply_errata,
    apply_erratum,
    load_errata,
)
from raglaw.schema import Row

pytestmark = pytest.mark.unit


def row(en: str = "", ar: str = "", page: int = 59, index: int = 2) -> Row:
    return Row(page=page, row_index=index, en_text=en, ar_text=ar, en_all_bold=False)


def erratum(**overrides) -> Erratum:
    fields = {
        "page": 59,
        "row": 2,
        "side": "en",
        "expect": "rticle 452",
        "replace": "Article 452",
        "reason": "Header typo: missing 'A' (P9)",
    }
    return Erratum(**(fields | overrides))


# --- U7 ---------------------------------------------------------------------


def test_a_matching_whole_line_is_replaced():
    fixed = apply_erratum(row("rticle 452\nAn action on a warranty"), erratum())

    assert fixed.en_text == "Article 452\nAn action on a warranty"


def test_an_absent_line_raises():
    with pytest.raises(ErratumNotFoundError, match="found 0"):
        apply_erratum(row("Article 453\nBody"), erratum())


def test_a_fixed_line_is_not_fixed_again():
    """`rticle 452` is a substring of its fix; whole-line matching stops a re-apply."""
    with pytest.raises(ErratumNotFoundError):
        apply_erratum(row("Article 452\nBody"), erratum())


def test_an_ambiguous_line_raises():
    with pytest.raises(ErratumNotFoundError, match="found 2"):
        apply_erratum(row("rticle 452\nrticle 452"), erratum())


def test_the_arabic_side_is_addressed_separately():
    fixed = apply_erratum(
        row("Article 601", "مادة\n٦٠ ١", page=81, index=6),
        erratum(page=81, row=6, side="ar", expect="٦٠ ١", replace="٦٠١"),
    )

    assert (fixed.en_text, fixed.ar_text) == ("Article 601", "مادة\n٦٠١")


def test_consecutive_lines_are_replaced_together():
    """A word the PDF split across a line break is rejoined (P27)."""
    fixed = apply_erratum(
        row(ar="الدائن المر\n ن الحق\nفى حبس", page=163, index=0),
        erratum(
            page=163,
            row=0,
            side="ar",
            expect="الدائن المر\n ن الحق",
            replace="الدائن المرتهن الحق",
        ),
    )

    assert fixed.ar_text == "الدائن المرتهن الحق\nفى حبس"


def test_consecutive_lines_must_all_be_whole_lines():
    with pytest.raises(ErratumNotFoundError, match="found 0"):
        apply_erratum(
            row(ar="الدائن المر\n ن الحق", page=163, index=0),
            erratum(page=163, row=0, side="ar", expect="المر\n ن الحق"),
        )


def test_errata_apply_to_their_own_rows_only():
    rows = [row("rticle 452", index=1), row("rticle 452", index=2)]

    fixed = apply_errata(rows, [erratum()])

    assert [r.en_text for r in fixed] == ["rticle 452", "Article 452"]


def test_an_erratum_for_a_missing_row_raises():
    with pytest.raises(ErratumNotFoundError, match="no such row"):
        apply_errata([row(index=1)], [erratum()])


# --- The file ---------------------------------------------------------------


def test_errata_file_entries_are_validated(tmp_path):
    path = tmp_path / "errata.yaml"
    path.write_text("- page: 0\n  row: 1\n  side: fr\n  expect: x\n  replace: y\n")

    with pytest.raises(ValidationError):
        load_errata(path)


def test_an_empty_errata_file_has_no_entries(tmp_path):
    path = tmp_path / "errata.yaml"
    path.write_text("# no entries yet\n")

    assert load_errata(path) == []


def test_the_project_errata_load(repo_root):
    """Every active entry is complete, and E1 to E3 come first (report section 5)."""
    errata = load_errata(repo_root / "data" / "errata.yaml")

    assert [(e.page, e.row, e.side) for e in errata[:3]] == [
        (59, 2, "en"),
        (147, 6, "en"),
        (81, 6, "ar"),
    ]
