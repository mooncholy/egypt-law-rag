"""The ``repair`` stage: the text-level repairs, in a fixed order.

1. **Errata** (R7): owner-approved fixes for one-off errors (``errata``).
2. **Same-line headers** (R8): ``Article 277 If the option…`` becomes a header
   line and a body line.
3. **Arabic headers** (R9): ``مادة`` with inner spaces or stray parentheses,
   its number on the next line, becomes one line ``مادة ١٤٧``.

Header repairs touch only article rows: rows whose English first line is a
header (R11, R12). No other text is edited (R10). Each count must equal its raw
baseline (G6), and every defect a repair targets must be gone afterwards.

Run as ``python -m raglaw.ingest.repair``.
"""

import argparse
import json
import logging
import re
from collections import Counter
from pathlib import Path

from raglaw.config import Settings
from raglaw.ingest import profile
from raglaw.ingest.errata import ErratumNotFoundError, apply_errata, load_errata
from raglaw.ingest.measure import AR_SPACED_MADA, EN_ARTICLE
from raglaw.logging_setup import log_anomaly
from raglaw.records import read_records, write_records
from raglaw.schema import LogEvent, Row
from raglaw.tracking import sha256_file, stage_run

logger = logging.getLogger(__name__)

# The header word, alone or with its number, among stray parentheses (P10).
AR_HEADER_LINE = re.compile(r"^[\s()]*(م\s*ا\s*د\s*ة)[\s()]*(?:([٠-٩]+)[\s()]*)?$")
AR_NUMBER_LINE = re.compile(r"^[\s()]*([٠-٩]+)[\s()]*$")
AR_HEADER_WORD = "مادة"


class RepairCountError(RuntimeError):
    """A repair count differs from its baseline, or a targeted defect remains."""


def article_header(row: Row) -> re.Match | None:
    """The English header match when the row is an article row (R11, R12)."""
    first = row.en_text.split("\n", 1)[0]
    return EN_ARTICLE.match(first)


def split_same_line_header(row: Row) -> Row | None:
    """
    Split ``Article N <body>`` on the first English line into two lines (R8).

    returns:
    - row (Row | None): the split row, or None when there is nothing to split
    """
    m = article_header(row)
    if not m or not m.group(4):
        return None
    first, *rest = row.en_text.split("\n")
    header = first[: m.start(4)].rstrip()
    return row.model_copy(update={"en_text": "\n".join([header, m.group(4), *rest])})


def parse_ar_header(lines: list[str]) -> tuple[str, int, bool] | None:
    """
    Read the Arabic header: ``مادة``, then its number on the same or next line.

    Spaces inside the word (``م ادة``, ``ما دة``, ``ماد ة``) and stray
    parentheses are accepted (P10).

    returns:
    - header (tuple | None): the digits, how many lines the header spans, and
      whether the word had inner spaces; None when the lines aren't a header
    """
    if not lines:
        return None
    m = AR_HEADER_LINE.match(lines[0])
    if not m:
        return None
    spaced = m.group(1) != AR_HEADER_WORD
    if m.group(2):
        return m.group(2), 1, spaced
    number = AR_NUMBER_LINE.match(lines[1]) if len(lines) > 1 else None
    if not number:
        return None
    return number.group(1), 2, spaced


def normalize_ar_header(row: Row) -> tuple[Row, bool] | None:
    """
    Rewrite an article row's Arabic header as one line, ``مادة <number>`` (R9).

    returns:
    - result (tuple | None): the rewritten row and whether its word was
      spaced; None when the header can't be read
    """
    lines = row.ar_text.split("\n")
    header = parse_ar_header(lines)
    if header is None:
        return None
    digits, span, spaced = header
    lines = [f"{AR_HEADER_WORD} {digits}", *lines[span:]]
    return row.model_copy(update={"ar_text": "\n".join(lines)}), spaced


def remaining_defects(rows: list[Row]) -> dict[str, int]:
    """
    Count the header defects the repairs target, after repair (C10).

    returns:
    - counts (dict[str, int]): header typos, same-line headers and spaced
      ``مادة`` left on article rows; each should be 0
    """
    counts = Counter()
    for row in rows:
        m = article_header(row)
        if not m:
            continue
        counts["header_typos_remaining"] += not m.group(1) or not m.group(2)
        counts["same_line_headers_remaining"] += bool(m.group(4))
        first_ar = row.ar_text.split("\n", 1)[0]
        counts["spaced_mada_remaining"] += bool(AR_SPACED_MADA.search(first_ar))
    keys = (
        "header_typos_remaining",
        "same_line_headers_remaining",
        "spaced_mada_remaining",
    )
    return {k: counts[k] for k in keys}


def repair_rows(rows: list[Row], errata: list) -> tuple[list[Row], dict[str, int]]:
    """
    Apply the three repairs, in order, to every row.

    returns:
    - rows (list[Row]): the repaired rows, in order
    - counts (dict[str, int]): how often each repair applied, how many Arabic
      headers couldn't be read, and the defects left (``remaining_defects``)

    exceptions:
    - ErratumNotFoundError: an erratum's line isn't where it says
    """
    rows = apply_errata(rows, errata)
    counts = Counter(errata_applied=len(errata))
    repaired = []
    for row in rows:
        if article_header(row):
            split = split_same_line_header(row)
            if split is not None:
                row = split
                counts["same_line_headers_split"] += 1
            normalized = normalize_ar_header(row)
            if normalized is None:
                counts["ar_headers_unparsed"] += 1
                log_anomaly(
                    logger,
                    "Arabic header can't be read; left as printed",
                    page=row.page,
                    row_index=row.row_index,
                    row_class="article",
                    article_number=int(article_header(row).group(3)),
                    anomaly_type="unparsed_ar_header",
                    ar_first_line=row.ar_text.split("\n", 1)[0],
                )
            else:
                row, spaced = normalized
                counts["ar_headers_normalized"] += 1
                counts["spaced_mada_normalized"] += spaced
        repaired.append(row)
    keys = (
        "errata_applied",
        "same_line_headers_split",
        "ar_headers_normalized",
        "spaced_mada_normalized",
        "ar_headers_unparsed",
    )
    return repaired, {k: counts[k] for k in keys} | remaining_defects(repaired)


def check_counts(counts: dict[str, int], n_errata: int) -> None:
    """
    Compare the repair counts with their baselines (G6, C10, C11).

    exceptions:
    - RepairCountError: a count differs, or a targeted defect remains
    """
    expected = {
        "errata_applied": n_errata,
        "same_line_headers_split": profile.RAW_DEFECTS["same_line_headers"],
        "spaced_mada_normalized": profile.RAW_DEFECTS["spaced_mada_headers"],
        "header_typos_remaining": 0,
        "same_line_headers_remaining": 0,
        "spaced_mada_remaining": 0,
    }
    wrong = {k: (v, counts[k]) for k, v in expected.items() if counts[k] != v}
    if wrong:
        raise RepairCountError(
            "repair counts differ from their baselines (expected, actual): "
            + ", ".join(f"{k} {v}" for k, v in wrong.items())
        )


def run_repair(
    rows_in: Path, errata_path: Path, out: Path, metrics_out: Path
) -> dict[str, int]:
    """
    Repair the rows in ``rows_in`` into ``out``, with counts in ``metrics_out``.

    returns:
    - counts (dict[str, int]): what was written to ``metrics_out``

    exceptions:
    - ErratumNotFoundError: an erratum's line isn't where it says; nothing is
      written
    - RepairCountError: a count differs from its baseline; both files are still
      written, for diagnosis
    """
    errata = load_errata(errata_path)
    rows, counts = repair_rows(read_records(rows_in, Row), errata)
    write_records(out, Row, rows)
    metrics_out.parent.mkdir(parents=True, exist_ok=True)
    metrics_out.write_text(json.dumps(counts, indent=2) + "\n", encoding="utf-8")
    logger.info(
        "Repaired %d rows",
        len(rows),
        extra={"event_type": LogEvent.STAGE_COMPLETED, **counts},
    )
    check_counts(counts, len(errata))
    return counts


def main(argv: list[str] | None = None) -> None:
    settings = Settings()
    paths = settings.paths
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rows", type=Path, default=paths.interim_dir / "rows.jsonl")
    ap.add_argument("--errata", type=Path, default=paths.errata)
    ap.add_argument(
        "--out", type=Path, default=paths.interim_dir / "rows_repaired.jsonl"
    )
    ap.add_argument("--metrics", type=Path, default=paths.metrics_dir / "repair.json")
    args = ap.parse_args(argv)

    try:
        with stage_run(
            "repair",
            input_hash=sha256_file(args.rows),
            output_model=Row,
            settings=settings,
        ) as run:
            try:
                metrics = run_repair(args.rows, args.errata, args.out, args.metrics)
            except RepairCountError:
                run.log_metrics(json.loads(args.metrics.read_text("utf-8")))
                raise
            run.log_metrics(metrics)
    except (RepairCountError, ErratumNotFoundError) as exc:
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
