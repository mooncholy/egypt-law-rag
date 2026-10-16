"""The ``validate`` stage (Phase 4): prove the repairs worked on the built corpus.

It re-measures ``articles.json`` and the repaired rows, where every targeted
defect must now be gone, and compares the corpus with the hand-checked gold
sample (``data/gold/articles_gold.json``). Each check is one of C1-C16 in
``development/docs/guidelines.md``; owner-approved exceptions are named below,
never absorbed. It writes ``docs/metrics/corpus.json`` and
``docs/reports/corpus_validation.md``, and fails when any check fails.

Run as ``python -m raglaw.ingest.validate``.
"""

import argparse
import json
import logging
import re
import statistics
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from raglaw.config import Settings
from raglaw.ingest import extract, profile, repair
from raglaw.ingest.assemble import assemble_rows
from raglaw.ingest.errata import load_errata
from raglaw.ingest.measure import EN_ARTICLE
from raglaw.records import read_records
from raglaw.schema import Article, Row
from raglaw.text.normalize import normalize_arabic
from raglaw.tracking import sha256_file, stage_run

logger = logging.getLogger(__name__)

ARTICLE_COUNT = 1_149  # D1, P11
REPEALED_RANGES = ((54, 80), (389, 417))  # P12
HEADING_ROWS = 222  # P17
CER_LIMIT = 0.005  # D2
LATIN_LIMIT = 0.01  # C13: Latin letters in any text_ar
# Owner-approved exceptions, each with its ruling.
EMPTY_AR_ALLOWED = {1022: "its Arabic cell is empty in the print (C8, 2026-10-14)"}
# C14: each accepted difference from the gold sample as (stored, gold) spans of
# text_ar, put back in their gold form before the comparison, so everything
# else in the article is checked like any other.
CER_ACCEPTED = {
    54: (
        "its repeal note's dates and number line are kept as stored (2026-10-14)",
        (
            ("المواد من ٥٤ إلى ٨٠\n", ""),
            ("في ٣/١١/١٩٥٦", "في ١٩٥٦/١١/٣"),
            ("في .١٩٦٤/٣/١٢", "في ١٢/٣/١٩٦٤."),
        ),
    )
}
# Rows whose Arabic header can't be read (repair.EXPECTED_UNPARSED_AR_HEADERS).
UNPARSED_AR_HEADERS = set(repair.EXPECTED_UNPARSED_AR_HEADERS)
# A phrase of the promulgation law above page 1's table (P7, D9), and the words
# Article 1 opens with, so nothing can sit before it.
PROMULGATION = "يلغي القانون المدني المعمول به"
ARTICLE_1_OPENS = "(١) تسرى النصوص التشريعية"
AR_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
HEADER_LINE_EN = re.compile(r"A?rticle\s*\d+")
HEADER_LINE_AR = re.compile(r"م\s*ا\s*د\s*ة\s*[٠-٩]+")
PRIVATE_USE = re.compile(r"[-]")


@dataclass(frozen=True)
class Check:
    """One validation check: whether it passed, what it measured, and why."""

    id: str
    passed: bool
    value: Any
    detail: str


def _ranges() -> set[int]:
    return {n for first, last in REPEALED_RANGES for n in range(first, last + 1)}


def check_numbers(articles: Sequence[Article]) -> Check:
    """C1: article numbers are exactly 1 to 1,149, with no duplicates."""
    numbers = [a.article_number for a in articles]
    dupes = sorted({n for n in numbers if numbers.count(n) > 1})
    missing = sorted(set(range(1, ARTICLE_COUNT + 1)) - set(numbers))
    extra = sorted(set(numbers) - set(range(1, ARTICLE_COUNT + 1)))
    return Check(
        "C1",
        not (dupes or missing or extra),
        len(numbers),
        f"duplicates {dupes}, missing {missing}, outside 1-{ARTICLE_COUNT} {extra}",
    )


def check_repealed(articles: Sequence[Article]) -> Check:
    """C2: 54-80 and 389-417 are flagged repealed, and no other record is."""
    flagged = {a.article_number for a in articles if a.is_repealed}
    expected = _ranges()
    return Check(
        "C2",
        flagged == expected,
        len(flagged),
        f"unflagged {sorted(expected - flagged)}, wrongly flagged "
        f"{sorted(flagged - expected)}",
    )


def check_numbers_agree(rows: Sequence[Row]) -> Check:
    """C3: every article row's Arabic header number equals its English one."""
    mismatched, read = [], 0
    for row in rows:
        en = EN_ARTICLE.match(row.en_text.split("\n", 1)[0])
        if not en or not en.group(1):
            continue
        number = int(en.group(3))
        ar = re.fullmatch(r"مادة ([٠-٩]+)", row.ar_text.split("\n", 1)[0])
        if ar is None:
            if number not in UNPARSED_AR_HEADERS:
                mismatched.append((number, "unreadable"))
            continue
        read += 1
        if int(ar.group(1).translate(AR_DIGITS)) != number:
            mismatched.append((number, ar.group(1)))
    return Check("C3", not mismatched, read, f"mismatched {mismatched[:10]}")


def check_rows(assemble: dict[str, int]) -> list[Check]:
    """C4, C5, C9: no anomaly rows, the row accounting closes, every heading row consumed."""
    anomalies = assemble.get("anomalies", 0)
    total = sum(assemble[k] for k in assemble if k.startswith("rows_"))
    return [
        Check("C4", anomalies == 0, anomalies, "anomaly rows left by assemble"),
        Check(
            "C5",
            total == profile.ROWS_DETECTED,
            total,
            f"article + heading + continuation + repeal rows against "
            f"{profile.ROWS_DETECTED} detected",
        ),
        Check(
            "C9",
            assemble.get("rows_heading") == HEADING_ROWS,
            assemble.get("rows_heading"),
            f"heading rows consumed by the stack, of {HEADING_ROWS}",
        ),
    ]


def check_bodies(articles: Sequence[Article]) -> list[Check]:
    """C6, C7, C8: no header line inside a body, paragraph markers in sequence,
    no empty Arabic text or heading path."""
    headers, markers, empty_ar, no_path = [], [], [], []
    for a in articles:
        if any(HEADER_LINE_EN.fullmatch(ln.strip()) for ln in a.text_en.split("\n")):
            headers.append(a.article_number)
        if any(HEADER_LINE_AR.fullmatch(ln.strip()) for ln in a.text_ar.split("\n")):
            headers.append(a.article_number)
        if not a.is_repealed:
            numbers = [
                int(m.group(1).translate(AR_DIGITS))
                for ln in a.text_ar.split("\n")
                if (m := repair.AR_MARKER_LINE.match(ln))
            ]
            if numbers and numbers != list(range(1, len(numbers) + 1)):
                markers.append((a.article_number, numbers))
            if not a.heading_path:
                no_path.append(a.article_number)
        if not a.text_ar.strip() and a.article_number not in EMPTY_AR_ALLOWED:
            empty_ar.append(a.article_number)
    return [
        Check("C6", not headers, len(headers), f"articles {sorted(set(headers))}"),
        Check("C7", not markers, len(markers), f"out of sequence {markers[:10]}"),
        Check(
            "C8",
            not (empty_ar or no_path),
            len(empty_ar) + len(no_path),
            f"empty text_ar {empty_ar} (allowed: {sorted(EMPTY_AR_ALLOWED)}), "
            f"no heading path {no_path}",
        ),
    ]


def check_defects(articles: Sequence[Article], rows: Sequence[Row]) -> Check:
    """C10: every defect a repair targets, visible in text, is gone."""
    counts = {
        "header_typos": sum(
            bool(m := EN_ARTICLE.match(r.en_text.split("\n", 1)[0])) and not m.group(1)
            for r in rows
        ),
        "same_line_headers": sum(
            bool(m := EN_ARTICLE.match(r.en_text.split("\n", 1)[0]))
            and bool(m.group(4))
            for r in rows
        ),
        "private_use_glyphs": sum(
            len(PRIVATE_USE.findall(a.text_ar + a.text_en)) for a in articles
        ),
        "malformed_markers": sum(
            not repair.WELL_FORMED_MARKER.match(ln)
            for a in articles
            for ln in a.text_ar.split("\n")
            if repair.AR_MARKER_LINE.match(ln)
        ),
    }
    return Check("C10", not any(counts.values()), counts, "defects left in the text")


def check_repair_counts(
    extract_metrics: dict[str, int], repair_metrics: dict[str, int], n_errata: int
) -> Check:
    """C11: each repair count equals its baseline (G6 and the measured ones)."""
    expected = {
        "lam_alef_swaps": profile.RAW_DEFECTS["lam_alef_signatures"],
        "rtl_digit_runs_reordered": profile.RAW_DEFECTS["rtl_digit_runs"],
        "stray_zero_width_alefs": extract.STRAY_ZERO_WIDTH_ALEFS,
        "punctuation_moved": extract.PUNCTUATION_MOVED,
        "same_line_headers_split": profile.RAW_DEFECTS["same_line_headers"],
        "paragraph_markers_normalized": repair.PARAGRAPH_MARKERS_NORMALIZED,
        "errata_applied": n_errata,
    }
    found = extract_metrics | repair_metrics
    wrong = {k: (v, found.get(k)) for k, v in expected.items() if found.get(k) != v}
    return Check("C11", not wrong, len(expected), f"(expected, found): {wrong}")


def check_root(articles: Sequence[Article], root_en: str) -> Check:
    """C12: no promulgation-law text, and Article 1's path starts at the root."""
    leaked = [a.article_number for a in articles if PROMULGATION in a.text_ar]
    first = next((a for a in articles if a.article_number == 1), None)
    if first is None:
        return Check("C12", False, leaked, "Article 1 is missing")
    opens = first.text_ar.startswith(ARTICLE_1_OPENS)
    return Check(
        "C12",
        not leaked
        and opens
        and bool(first.heading_path)
        and first.heading_path[0] == root_en,
        leaked,
        f"promulgation text in {leaked}; Article 1 opens with its first words: "
        f"{opens}; its path starts {first.heading_path[:1]}",
    )


def _latin_share(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    return sum(c.isascii() for c in letters) / len(letters) if letters else 0.0


def check_characters(articles: Sequence[Article]) -> Check:
    """C13: no presentation forms or control characters, at most 1% Latin in
    any text_ar, and normalization idempotent over the corpus."""
    presentation = sum(
        0xFB50 <= ord(c) <= 0xFDFF or 0xFE70 <= ord(c) <= 0xFEFF
        for a in articles
        for c in a.text_ar
    )
    control = sum(
        unicodedata.category(c) in ("Cc", "Cf") and c != "\n"
        for a in articles
        for c in a.text_ar + a.text_en
    )
    latin = [
        a.article_number for a in articles if _latin_share(a.text_ar) > LATIN_LIMIT
    ]
    unstable = [
        a.article_number
        for a in articles
        if normalize_arabic(normalize_arabic(a.text_ar)) != normalize_arabic(a.text_ar)
    ]
    value = {
        "presentation_forms": presentation,
        "control_characters": control,
        "latin_over_1pct": latin,
        "normalization_unstable": unstable,
    }
    return Check(
        "C13",
        not (presentation or control or latin or unstable),
        value,
        "characters and normalization",
    )


def edit_distance(a: str, b: str) -> int:
    """Levenshtein distance in characters."""
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(
                min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb))
            )
        previous = current
    return previous[-1]


def accepted_text(number: int, text_ar: str) -> str:
    """
    ``text_ar`` with each accepted span of the article put in its gold form; a
    span that isn't there exactly once is left alone, so it counts as errors.
    """
    for stored, gold in CER_ACCEPTED.get(number, ("", ()))[1]:
        if text_ar.count(stored) == 1:
            text_ar = text_ar.replace(stored, gold)
    return text_ar


def check_gold(articles: Sequence[Article], gold: Sequence[dict]) -> Check:
    """C14: against the gold sample, article boundaries match exactly and the
    Arabic character error rate is at most 0.5% (D2), the accepted exceptions
    excluded and reported."""
    by_number = {a.article_number: a for a in articles}
    boundaries = sum(
        g["article_number"] in by_number
        and by_number[g["article_number"]].text_en.strip() == g["text_en"].strip()
        for g in gold
    )
    found = [g for g in gold if g["article_number"] in by_number]
    raw_errors = {
        g["article_number"]: edit_distance(
            by_number[g["article_number"]].text_ar, g["text_ar"]
        )
        for g in found
    }
    errors = {
        g["article_number"]: edit_distance(
            accepted_text(g["article_number"], by_number[g["article_number"]].text_ar),
            g["text_ar"],
        )
        for g in found
    }
    length = sum(len(g["text_ar"]) for g in found)
    cer = sum(errors.values()) / length if length else 1.0
    raw = sum(raw_errors.values()) / length if length else 1.0
    value = {
        "boundary_recall": boundaries / len(gold),
        "boundary_precision": boundaries / len(errors) if errors else 0.0,
        "ar_cer": round(cer, 6),
        "ar_cer_with_accepted": round(raw, 6),
        "errors_by_article": {n: e for n, e in errors.items() if e},
    }
    return Check(
        "C14",
        boundaries == len(gold) and cer <= CER_LIMIT,
        value,
        f"ar_cer {cer:.2%} (limit {CER_LIMIT:.1%}), {raw:.2%} without the "
        f"accepted spans of {sorted(CER_ACCEPTED)}",
    )


def check_deterministic(rows: Sequence[Row], root_en: str, root_ar: str) -> Check:
    """C16: assembling the same rows twice gives identical articles."""
    first = [
        a.model_dump() for a in assemble_rows(list(rows), root_en, root_ar).articles
    ]
    second = [
        a.model_dump() for a in assemble_rows(list(rows), root_en, root_ar).articles
    ]
    return Check("C16", first == second, len(first), "two assemblies compared")


def outliers(articles: Sequence[Article]) -> dict[str, Any]:
    """The articles worth a look, reported but never failed (Gate 4)."""
    live = [a for a in articles if not a.is_repealed and a.text_ar.strip()]
    ratios = {a.article_number: len(a.text_en) / len(a.text_ar) for a in live}
    q1, _, q3 = statistics.quantiles(ratios.values(), n=4)
    low, high = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
    lengths = sorted(live, key=lambda a: len(a.text_ar))
    cut = max(1, len(lengths) // 100)
    return {
        "en_ar_ratio_bounds": [round(low, 3), round(high, 3)],
        "en_ar_ratio_outliers": sorted(
            n for n, r in ratios.items() if not low <= r <= high
        ),
        "shortest_ar": [a.article_number for a in lengths[:cut]],
        "longest_ar": [a.article_number for a in lengths[-cut:]],
        "latin_in_ar_over_1pct": [
            a.article_number for a in live if _latin_share(a.text_ar) > LATIN_LIMIT
        ],
    }


def validate(
    articles: Sequence[Article],
    rows: Sequence[Row],
    gold: Sequence[dict],
    metrics: dict[str, dict],
    n_errata: int,
    root: tuple[str, str],
) -> list[Check]:
    """Every check, in ID order."""
    checks = [
        check_numbers(articles),
        check_repealed(articles),
        check_numbers_agree(rows),
        *check_rows(metrics["assemble"]),
        *check_bodies(articles),
        check_defects(articles, rows),
        check_repair_counts(metrics["extract"], metrics["repair"], n_errata),
        check_root(articles, root[0]),
        check_characters(articles),
        check_gold(articles, gold),
        check_deterministic(rows, *root),
    ]
    return sorted(checks, key=lambda c: int(c.id[1:]))


def report(checks: Sequence[Check], found: dict[str, Any], sha: str) -> str:
    """The validation report (Markdown)."""
    lines = [
        "# Corpus validation",
        "",
        (
            "Generated by `python -m raglaw.ingest.validate` (Phase 4). Each check is "
            "one of C1-C16 in `development/docs/guidelines.md`."
        ),
        "",
        f"`articles.json` sha256: `{sha}`",
        "",
        "| Check | Result | Measured | Detail |",
        "| --- | --- | --- | --- |",
    ]
    for c in checks:
        value = json.dumps(c.value, ensure_ascii=False)
        lines.append(
            f"| {c.id} | {'pass' if c.passed else '**FAIL**'} | `{value}` | {c.detail} |"
        )
    lines += ["", "## Accepted exceptions", ""]
    accepted = {n: why for n, (why, _) in CER_ACCEPTED.items()}
    for n, why in {**EMPTY_AR_ALLOWED, **accepted}.items():
        lines.append(f"- Article {n}: {why}.")
    lines += ["", "## Outliers (reported, not failed)", ""]
    lines += [f"- `{k}`: {json.dumps(v, ensure_ascii=False)}" for k, v in found.items()]
    return "\n".join(lines) + "\n"


def run_validate(
    articles_path: Path,
    rows_path: Path,
    gold_path: Path,
    metrics_dir: Path,
    errata_path: Path,
    root: tuple[str, str],
    metrics_out: Path,
    report_out: Path,
) -> list[Check]:
    """Run every check and write the metrics and the report."""
    articles = read_records(articles_path, Article)
    rows = read_records(rows_path, Row)
    gold = json.loads(gold_path.read_text("utf-8"))["articles"]

    def read(name: str) -> dict:
        return json.loads((metrics_dir / f"{name}.json").read_text("utf-8"))

    metrics = {name: read(name) for name in ("extract", "repair", "assemble")}
    checks = validate(
        articles, rows, gold, metrics, len(load_errata(errata_path)), root
    )
    found = outliers(articles)
    sha = sha256_file(articles_path)
    payload = {
        "corpus_sha256": sha,
        "checks": {c.id: "pass" if c.passed else "fail" for c in checks},
        "values": {c.id: c.value for c in checks},
        "outliers": found,
    }
    metrics_out.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", "utf-8"
    )
    report_out.write_text(report(checks, found, sha), "utf-8")
    return checks


def main(argv: list[str] | None = None) -> None:
    settings = Settings()
    paths = settings.paths
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--metrics", type=Path, default=Path(paths.metrics_dir) / "corpus.json"
    )
    ap.add_argument(
        "--report", type=Path, default=Path(paths.reports_dir) / "corpus_validation.md"
    )
    args = ap.parse_args(argv)
    articles_path = Path(paths.corpus_dir) / "articles.json"
    with stage_run(
        "validate", input_hash=sha256_file(articles_path), settings=settings
    ) as run:
        checks = run_validate(
            articles_path,
            Path(paths.interim_dir) / "rows_repaired.jsonl",
            Path(paths.gold_dir) / "articles_gold.json",
            Path(paths.metrics_dir),
            Path(paths.errata),
            (settings.root_heading.en, settings.root_heading.ar),
            args.metrics,
            args.report,
        )
        run.log_metrics({f"check_{c.id}": float(c.passed) for c in checks})
        run.log_artifact(args.report, artifact_path="reports")
    failed = [c.id for c in checks if not c.passed]
    for c in checks:
        (logger.info if c.passed else logger.error)(
            "%s %s: %s", c.id, c.passed, c.detail
        )
    if failed:
        raise SystemExit(f"validation failed: {', '.join(failed)}")


if __name__ == "__main__":
    main()
