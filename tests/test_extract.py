import json

import pytest

from raglaw.ingest import profile
from raglaw.ingest.extract import (
    ExtractCountError,
    check_counts,
    extract_document,
    run_extract,
)
from raglaw.records import read_records
from raglaw.schema import Row

# --- Unit: the synthetic PDF ----------------------------------------------------


@pytest.mark.unit
def test_every_table_row_becomes_one_row_in_order(synthetic_pdf):
    rows, counts = extract_document(synthetic_pdf)

    assert [(r.page, r.row_index) for r in rows] == [
        (1, 0),
        (1, 1),
        (1, 2),
        (1, 3),
        (2, 0),
        (2, 1),
        (2, 2),
    ]
    assert (counts["pages"], counts["rows"], counts["rows_missing_a_side"]) == (2, 7, 0)


@pytest.mark.unit
def test_rows_hold_each_side_and_the_bold_signal(synthetic_pdf):
    rows, _ = extract_document(synthetic_pdf)

    heading, article = rows[0], rows[1]
    assert heading.en_text == "SECTION I\nGeneral Provisions"
    assert heading.en_all_bold
    assert article.en_text == "Article 1\nLegislative provisions govern."
    assert article.ar_text == "MADA 1\nBody one."
    assert not article.en_all_bold


@pytest.mark.unit
def test_text_outside_the_table_is_never_read(synthetic_pdf, outside_table_text):
    """U14: page 1's line above the table, like the promulgation law (P7)."""
    rows, _ = extract_document(synthetic_pdf)

    assert not any(outside_table_text in r.en_text + r.ar_text for r in rows)


@pytest.mark.unit
def test_counts_off_their_baselines_fail_the_stage():
    with pytest.raises(ExtractCountError, match="lam_alef_swaps"):
        check_counts({"lam_alef_swaps": 3_499, "rtl_digit_runs_reordered": 1_167})


@pytest.mark.unit
def test_stage_writes_rows_and_metrics_even_when_counts_fail(synthetic_pdf, tmp_path):
    out, metrics = tmp_path / "rows.jsonl", tmp_path / "extract.json"

    with pytest.raises(ExtractCountError):  # no lam-alefs in a Latin PDF
        run_extract(synthetic_pdf, out, metrics)

    assert len(read_records(out, Row)) == 7
    assert json.loads(metrics.read_text())["rows"] == 7


# --- Corpus: the full PDF -------------------------------------------------------


@pytest.mark.corpus
def test_extract_counts_equal_their_baselines(stage_metrics):
    counts = stage_metrics("extract")

    assert counts["pages"] == profile.PAGES
    assert counts["rows"] == profile.ROWS_DETECTED
    assert counts["rows_missing_a_side"] == 0
    assert counts["lam_alef_swaps"] == profile.RAW_DEFECTS["lam_alef_signatures"]
    assert counts["rtl_digit_runs_reordered"] == profile.RAW_DEFECTS["rtl_digit_runs"]
