"""The ``extract`` stage: one ``Row`` record per detected table row.

Rows come only from each page's ``lines_strict`` table (R1 to R3), so text
outside the table, such as the page 1 promulgation law, is never read. Each
cell's text is built from clipped ``rawdict`` characters with the glyph repairs
applied before joining (R4 to R6; see ``glyphs``).

The repair counts must equal the raw baselines that ``profile`` checked (G6):
a different count means a repair touched something it shouldn't have, or
missed something, and the stage fails.

Run as ``python -m raglaw.ingest.extract``.
"""

import argparse
import json
import logging
from collections import Counter
from pathlib import Path

import pymupdf

from raglaw.config import Settings
from raglaw.ingest import profile
from raglaw.ingest.glyphs import read_cell
from raglaw.ingest.measure import is_bold, spans_in
from raglaw.ingest.table import TableRow, iter_rows
from raglaw.logging_setup import log_anomaly
from raglaw.records import write_records
from raglaw.schema import LogEvent, Row
from raglaw.tracking import sha256_file, stage_run

logger = logging.getLogger(__name__)


class ExtractCountError(RuntimeError):
    """A glyph repair count differs from its raw baseline."""


def extract_row(page: pymupdf.Page, row: TableRow, counts: Counter) -> Row:
    """
    Read one table row into a ``Row``, adding its repair counts to ``counts``.

    A side with no cell gives an empty text and an anomaly, never a dropped row.

    returns:
    - row (Row): the row's repaired English and Arabic text
    """
    en = read_cell(page, row.en_rect, rtl=False)
    ar = read_cell(page, row.ar_rect, rtl=True)
    for side, cell in (("en", en), ("ar", ar)):
        counts[f"line_pieces_merged_{side}"] += cell.pieces_merged
        counts["lam_alef_swaps"] += cell.lam_alef_swaps
        counts["rtl_digit_runs_reordered"] += cell.digit_runs_reordered
        counts["stray_zero_width_alefs"] += cell.stray_zero_width_alefs
        counts["private_use_glyphs"] += len(cell.private_use_glyphs)
    if not row.has_both_sides:
        counts["rows_missing_a_side"] += 1
        log_anomaly(
            logger,
            "Row has no cell on one side of the table",
            page=row.page,
            row_index=row.index,
            row_class="unclassified",
            article_number=None,
            anomaly_type="missing_side",
        )
    strays = en.stray_zero_width_alefs + ar.stray_zero_width_alefs
    if strays:
        log_anomaly(
            logger,
            "Cell holds zero-width alefs with no lam, kept as printed",
            page=row.page,
            row_index=row.index,
            row_class="unclassified",
            article_number=None,
            anomaly_type="stray_zero_width_alef",
            count=strays,
        )
    glyphs = en.private_use_glyphs + ar.private_use_glyphs
    if glyphs:
        log_anomaly(
            logger,
            "Cell holds private-use glyphs that no rule maps yet",
            page=row.page,
            row_index=row.index,
            row_class="unclassified",
            article_number=None,
            anomaly_type="private_use_glyph",
            glyphs=[f"U+{ord(g):04X}" for g in glyphs],
        )
    en_spans = spans_in(page, row.en_rect) if row.en_rect is not None else []
    return Row(
        page=row.page,
        row_index=row.index,
        en_text=en.text,
        ar_text=ar.text,
        en_all_bold=bool(en_spans) and all(is_bold(s) for s in en_spans),
    )


def extract_document(pdf: Path) -> tuple[list[Row], dict[str, int]]:
    """
    Extract every table row of the PDF, in page and row order.

    returns:
    - rows (list[Row]): one per detected row
    - counts (dict[str, int]): pages, rows and the count of each glyph repair
    """
    counts: Counter = Counter()
    rows: list[Row] = []
    with pymupdf.open(pdf) as doc:
        for number, page in enumerate(doc, start=1):
            rows += [extract_row(page, r, counts) for r in iter_rows(page, number)]
        counts["pages"] = doc.page_count
    counts["rows"] = len(rows)
    counts["rows_empty_en"] = sum(not r.en_text for r in rows)
    counts["rows_empty_ar"] = sum(not r.ar_text for r in rows)
    keys = (
        "pages",
        "rows",
        "rows_missing_a_side",
        "rows_empty_en",
        "rows_empty_ar",
        "line_pieces_merged_en",
        "line_pieces_merged_ar",
        "lam_alef_swaps",
        "stray_zero_width_alefs",
        "rtl_digit_runs_reordered",
        "private_use_glyphs",
    )
    return rows, {k: counts[k] for k in keys}


def check_counts(counts: dict[str, int]) -> None:
    """
    Compare the repair counts with the raw baselines (G6, C11).

    exceptions:
    - ExtractCountError: a count differs from its baseline
    """
    expected = {
        "lam_alef_swaps": profile.RAW_DEFECTS["lam_alef_signatures"],
        "rtl_digit_runs_reordered": profile.RAW_DEFECTS["rtl_digit_runs"],
    }
    wrong = {k: (v, counts[k]) for k, v in expected.items() if counts[k] != v}
    if wrong:
        raise ExtractCountError(
            "glyph repair counts differ from their baselines "
            "(expected, actual): " + ", ".join(f"{k} {v}" for k, v in wrong.items())
        )


def run_extract(pdf: Path, out: Path, metrics_out: Path) -> dict[str, int]:
    """
    Extract ``pdf`` into ``out`` (``Row`` records) and its counts into ``metrics_out``.

    returns:
    - counts (dict[str, int]): what was written to ``metrics_out``

    exceptions:
    - ExtractCountError: a repair count differs from its baseline; both files
      are still written, for diagnosis
    """
    rows, counts = extract_document(pdf)
    write_records(out, Row, rows)
    metrics_out.parent.mkdir(parents=True, exist_ok=True)
    metrics_out.write_text(json.dumps(counts, indent=2) + "\n", encoding="utf-8")
    logger.info(
        "Extracted %d rows from %d pages",
        counts["rows"],
        counts["pages"],
        extra={"event_type": LogEvent.STAGE_COMPLETED, **counts},
    )
    check_counts(counts)
    return counts


def main(argv: list[str] | None = None) -> None:
    settings = Settings()
    paths = settings.paths
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pdf", type=Path, default=paths.raw_pdf)
    ap.add_argument("--out", type=Path, default=paths.interim_dir / "rows.jsonl")
    ap.add_argument("--metrics", type=Path, default=paths.metrics_dir / "extract.json")
    args = ap.parse_args(argv)

    try:
        with stage_run(
            "extract",
            input_hash=sha256_file(args.pdf),
            output_model=Row,
            settings=settings,
        ) as run:
            try:
                metrics = run_extract(args.pdf, args.out, args.metrics)
            except ExtractCountError:
                run.log_metrics(json.loads(args.metrics.read_text("utf-8")))
                raise
            run.log_metrics(metrics)
    except ExtractCountError as exc:
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
