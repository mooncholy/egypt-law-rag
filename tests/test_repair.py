import json

import pytest

from raglaw.ingest import profile
from raglaw.ingest.errata import Erratum, load_errata
from raglaw.ingest.repair import (
    RepairCountError,
    check_counts,
    normalize_ar_header,
    parse_ar_header,
    repair_rows,
    run_repair,
    split_same_line_header,
)
from raglaw.records import read_records, write_records
from raglaw.schema import LogEvent, Row


def row(en: str, ar: str = "", index: int = 0, bold: bool = False) -> Row:
    return Row(
        page=10,
        row_index=index,
        en_text=en,
        ar_text=ar,
        en_all_bold=bold,
        ar_all_bold=bold,
    )


# --- Same-line headers (U6) -----------------------------------------------------


@pytest.mark.unit
def test_header_and_body_on_one_line_are_split():
    split = split_same_line_header(row("Article 277 If the option belongs\nto the"))

    assert split.en_text == "Article 277\nIf the option belongs\nto the"


@pytest.mark.unit
@pytest.mark.parametrize(
    "en",
    ["Article 277\nIf the option", "paragraph 2 of Article 717.", "SECTION I"],
)
def test_nothing_to_split(en):
    assert split_same_line_header(row(en)) is None


# --- Arabic headers (U5) --------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize(
    ("lines", "expected"),
    [
        (["مادة", "١٤٧", "نص"], ("١٤٧", 2, False)),
        (["( مادة", "١٥(", "نص"], ("١٥", 2, False)),
        (["م ادة", "٤٥٢"], ("٤٥٢", 2, True)),
        (["ما دة", "٤٥٢"], ("٤٥٢", 2, True)),
        (["ماد ة", "٤٥٢"], ("٤٥٢", 2, True)),
        (["مادة ٨٨"], ("٨٨", 1, False)),
        (["مادة ( ٣)"], ("٣", 1, False)),
    ],
)
def test_arabic_header_forms_are_read(lines, expected):
    assert parse_ar_header(lines) == expected


@pytest.mark.unit
@pytest.mark.parametrize(
    "lines",
    [["المواد من ٥٤ إلى ٨٠"], [""], ["مادة", "٦٠ ١"], ["مادة"], []],
)
def test_lines_that_are_not_a_header_are_rejected(lines):
    assert parse_ar_header(lines) is None


@pytest.mark.unit
def test_header_is_rewritten_as_one_line():
    normalized, spaced = normalize_ar_header(
        row("Article 452", "( م ادة\n٤٥٢(\nنص المادة")
    )

    assert normalized.ar_text == "مادة ٤٥٢\nنص المادة"
    assert spaced


# --- The repairs together -------------------------------------------------------


@pytest.mark.unit
def test_repairs_apply_in_order_to_article_rows_only():
    rows = [
        row("SECTION I", "( مادة\n١", index=0, bold=True),
        row("rticle 2 Body on the header line", "مادة\n٢", index=1),
        row("Article 3", "المواد من", index=2),
    ]
    typo = Erratum(
        page=10,
        row=1,
        side="en",
        expect="rticle 2 Body on the header line",
        replace="Article 2 Body on the header line",
        reason="test",
    )

    repaired, counts = repair_rows(rows, [typo])

    assert repaired[0] == rows[0]  # not an article row: untouched
    assert repaired[1].en_text == "Article 2\nBody on the header line"
    assert repaired[1].ar_text == "مادة ٢"
    assert repaired[2] == rows[2]  # unreadable header: left as printed
    assert counts == {
        "errata_applied": 1,
        "same_line_headers_split": 1,
        "ar_headers_normalized": 1,
        "spaced_mada_normalized": 0,
        "ar_headers_unparsed": 1,
        "header_typos_remaining": 0,
        "same_line_headers_remaining": 0,
        "spaced_mada_remaining": 0,
    }


@pytest.mark.unit
def test_unreadable_arabic_header_is_an_anomaly(log_records):
    repair_rows([row("Article 54", "المواد من ٥٤", index=3)], [])

    [anomaly] = log_records(event_type=LogEvent.ANOMALY)
    assert anomaly["anomaly_type"] == "unparsed_ar_header"
    assert (anomaly["page"], anomaly["row_index"]) == (10, 3)
    assert (anomaly["row_class"], anomaly["article_number"]) == ("article", 54)


@pytest.mark.unit
def test_counts_off_their_baselines_fail_the_stage():
    counts = {
        "errata_applied": 3,
        "same_line_headers_split": 5,
        "spaced_mada_normalized": 0,
        "header_typos_remaining": 0,
        "same_line_headers_remaining": 0,
        "spaced_mada_remaining": 1,
    }

    with pytest.raises(RepairCountError, match="same_line_headers_split") as err:
        check_counts(counts, n_errata=3)
    assert "spaced_mada_remaining" in str(err.value)


@pytest.mark.unit
def test_stage_reads_rows_and_writes_repaired_rows_and_metrics(tmp_path):
    rows_in, out, metrics = (
        tmp_path / "rows.jsonl",
        tmp_path / "rows_repaired.jsonl",
        tmp_path / "repair.json",
    )
    write_records(rows_in, Row, [row("Article 1 Body", "مادة\n١")])
    errata = tmp_path / "errata.yaml"
    errata.write_text("")

    with pytest.raises(RepairCountError):  # one split, not the source's six
        run_repair(rows_in, errata, out, metrics)

    [repaired] = read_records(out, Row)
    assert repaired.en_text == "Article 1\nBody"
    assert json.loads(metrics.read_text())["same_line_headers_split"] == 1


# --- Corpus: the full PDF -------------------------------------------------------


@pytest.mark.corpus
def test_repair_counts_equal_their_baselines(stage_metrics, repo_root):
    counts = stage_metrics("repair")

    assert counts["errata_applied"] == len(load_errata(repo_root / "data/errata.yaml"))
    assert counts["same_line_headers_split"] == profile.RAW_DEFECTS["same_line_headers"]
    assert counts["spaced_mada_normalized"] == 0  # extract joined all 13 (P30)
    assert counts["ar_headers_normalized"] + counts["ar_headers_unparsed"] == 1_094
    assert (
        counts["ar_headers_unparsed"] == 2
    )  # Article 54's repeal note, 1022's empty cell
    assert all(v == 0 for k, v in counts.items() if k.endswith("_remaining"))
