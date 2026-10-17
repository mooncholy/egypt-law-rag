"""The ``assemble`` stage: repaired rows become one ``Article`` record per article.

1. **Classify** each row (R11, R12): an article starts at a header-form first
   line; a row declaring a repealed range is a repeal (R19); a bold row is a
   heading (the Arabic side decides when the English cell is empty in the
   PDF); a non-bold first row on a page is a continuation; anything else is an
   anomaly, logged and never merged.
2. **Merge** each continuation into the previous page's last unit (R13).
3. **Headings** (R14 to R18): a heading cell holds one or more headings. Lines
   are grouped per language, a keyword-only line taking the next line as its
   title. A plain line is a wrapped title when the other language has fewer
   headings in the cell. Each heading's rank comes from its keyword, else its
   numbering (in either language), else it is plain; a heading listed after
   another in the same cell nests under it. A heading of rank *r* pops every
   open heading of rank >= *r*. A keyword-only heading that ends a page takes
   its title from the next page.
4. **Repeals** expand to one record per article (R19).
   Highlighted text (P32) is carried onto each article as ``only_in_en`` and
   ``only_in_ar``: passages with no counterpart in the other language. A
   heading in one language only is marked in the heading tree (R28).
5. **Records** are keyed on the English number, which the Arabic header must
   agree with (R20); Arabic text is normalized (``docs/normalization.md``).

Run as ``python -m raglaw.ingest.assemble``.
"""

import argparse
import json
import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from raglaw.config import Settings
from raglaw.ingest.measure import (
    AR_DIGITS,
    AR_NUMBERED,
    EN_ARTICLE,
    EN_NUMBERED,
    EN_RANGE,
    KEYWORD,
    KEYWORD_ONLY,
)
from raglaw.logging_setup import log_anomaly
from raglaw.records import read_records, write_records
from raglaw.schema import Article, LogEvent, Row
from raglaw.text.normalize import normalize_arabic
from raglaw.tracking import sha256_file, stage_run

logger = logging.getLogger(__name__)

ROOT_RANK, NUMBERED_RANK, PLAIN_RANK = 1, 5, 6
RANK_BY_KEYWORD = {"PART": 1, "BOOK": 2, "CHAPTER": 3, "SECTION": 4}
# The English keyword for each Arabic one (P17), so both rank the same way.
EN_KEYWORD_OF = {
    "القسم": "PART",
    "الكتاب": "BOOK",
    "الباب": "CHAPTER",
    "الفصل": "SECTION",
}
# An Arabic keyword line: the keyword and its ordinal word, alone (الفصل الأول).
AR_KEYWORD_ONLY = re.compile(r"^(القسم|الكتاب|الباب|الفصل)\s+\S+$")
AR_KEYWORD = re.compile(r"^(القسم|الكتاب|الباب|الفصل)\b")
# Arabic ordinal-word numbering, as on page 8 (P21).
AR_ORDINAL = re.compile(r"^(أولاً|ثانياً|ثالثاً|رابعاً|خامساً)")
AR_HEADER_DIGITS = re.compile(r"^مادة\s*([٠-٩]+)$")
# A highlighted header line (``Article1022``, ``مادة``) is not body text.
HEADER_FRAGMENT = re.compile(r"^\(?\s*(A?rticle\s*\d+|م\s*ا\s*د\s*ة)")


class RowClass(StrEnum):
    ARTICLE = "article"
    REPEAL = "repeal"
    HEADING = "heading"
    CONTINUATION = "continuation"
    ANOMALY = "anomaly"


def classify_row(row: Row) -> RowClass:
    """
    Classify one repaired row (R11, R12, R19).

    Only a header-form *first* line starts an article, so an inline citation
    (``paragraph 2 of Article 717.``) never does.

    returns:
    - kind (RowClass): article, repeal, heading, continuation or anomaly
    """
    if EN_ARTICLE.match(row.en_text.split("\n", 1)[0]):
        return RowClass.ARTICLE
    if EN_RANGE.search(row.en_text):
        return RowClass.REPEAL
    if row.en_all_bold or (row.en_all_bold is None and row.ar_all_bold):
        return RowClass.HEADING
    if row.row_index == 0:
        return RowClass.CONTINUATION
    return RowClass.ANOMALY


# --- Headings ---------------------------------------------------------------


@dataclass
class Unit:
    """One heading as written in one language: its lines, and how it's marked."""

    lines: list[str]
    keyword: str | None = None  # PART, BOOK, CHAPTER, SECTION (English names)
    numbered: bool = False
    keyword_only: bool = False  # a keyword line still waiting for its title

    @property
    def text(self) -> str:
        return " ".join(self.lines)

    @property
    def plain(self) -> bool:
        return self.keyword is None and not self.numbered


def _en_unit(line: str) -> Unit:
    kw = KEYWORD.search(line)
    return Unit(
        [line],
        keyword=kw.group(1).upper() if kw else None,
        numbered=bool(EN_NUMBERED.match(line)),
        keyword_only=bool(KEYWORD_ONLY.match(line.rstrip("."))),
    )


def _ar_unit(line: str) -> Unit:
    kw = AR_KEYWORD.match(line)
    return Unit(
        [line],
        keyword=EN_KEYWORD_OF[kw.group(1)] if kw else None,
        numbered=bool(AR_NUMBERED.match(line) or AR_ORDINAL.match(line)),
        keyword_only=bool(AR_KEYWORD_ONLY.match(line)),
    )


def heading_units(lines: list[str], unit_of) -> list[Unit]:
    """
    Group one language's heading lines into headings (R16).

    A keyword-only line takes the next line as its title; every other line
    starts a heading of its own. Wrapped titles are joined later, by
    ``align_units``, when the other language shows fewer headings.

    returns:
    - units (list[Unit]): the cell's headings, in order
    """
    units: list[Unit] = []
    i = 0
    while i < len(lines):
        unit = unit_of(lines[i])
        if unit.keyword_only and i + 1 < len(lines):
            unit.lines.append(lines[i + 1])
            unit.keyword_only = False
            i += 1
        units.append(unit)
        i += 1
    return units


def _join_wraps(units: list[Unit], target: int) -> list[Unit]:
    """Join plain units onto the one before, last first, until ``target`` remain."""
    units = list(units)
    i = len(units) - 1
    while len(units) > target and i > 0:
        if units[i].plain:
            units[i - 1].lines += units.pop(i).lines
        i -= 1
    return units


def align_units(
    en: list[Unit], ar: list[Unit]
) -> tuple[list[tuple[Unit | None, Unit | None]], bool]:
    """
    Pair the two languages' headings in a cell, joining wrapped titles.

    A side with more headings than the other has its plain units joined onto
    the heading before them (a title wrapped onto a second line). An empty side
    pairs every heading with None.

    returns:
    - pairs (list[tuple]): (English, Arabic) per heading
    - aligned (bool): False when the counts still differ after joining
    """
    if not en or not ar:
        return [(u, None) for u in en] + [(None, u) for u in ar], True
    if len(en) > len(ar):
        en = _join_wraps(en, len(ar))
    elif len(ar) > len(en):
        ar = _join_wraps(ar, len(en))
    aligned = len(en) == len(ar)
    n = max(len(en), len(ar))
    en_padded = en + [None] * (n - len(en))
    ar_padded = ar + [None] * (n - len(ar))
    return list(zip(en_padded, ar_padded, strict=True)), aligned


@dataclass
class Heading:
    """One heading of the code, with its rank and both labels."""

    rank: int
    en: str
    ar: str
    page: int
    row_index: int
    articles: list[int] = field(default_factory=list)
    only_in: str | None = None  # "en" or "ar": no counterpart in the other language


def heading_rank(en: Unit | None, ar: Unit | None) -> tuple[int, bool]:
    """
    Rank one heading (R14, R15): keyword, else numbered in either language, else plain.

    returns:
    - rank (int): 1 to 6
    - mismatch (bool): True when only one language numbers it (P21)
    """
    keyword = (en.keyword if en else None) or (ar.keyword if ar else None)
    if keyword:
        return RANK_BY_KEYWORD[keyword], False
    en_num, ar_num = bool(en and en.numbered), bool(ar and ar.numbered)
    if en_num or ar_num:
        return NUMBERED_RANK, bool(en and ar) and en_num != ar_num
    return PLAIN_RANK, False


class HeadingStack:
    """The open headings, root first (R14, R18)."""

    def __init__(self, root: Heading) -> None:
        self.open = [root]

    def push(self, heading: Heading) -> None:
        """Pop every open heading of rank >= the new one's, then push it."""
        self.open = [h for h in self.open if h.rank < heading.rank] + [heading]

    def paths(self) -> tuple[list[str], list[str]]:
        return [h.en for h in self.open], [h.ar for h in self.open]


# --- Assembly ---------------------------------------------------------------


@dataclass
class Draft:
    """An article being assembled: its header row plus any continuations."""

    number: int
    en_lines: list[str]
    ar_lines: list[str]
    heading_path: list[str]
    heading_path_ar: list[str]
    rows: list[tuple[int, int]]
    ar_number: int | None = None
    repeal: tuple[int, int] | None = None
    en_highlighted: list[str] = field(default_factory=list)
    ar_highlighted: list[str] = field(default_factory=list)


def _ar_header_number(ar_lines: list[str]) -> tuple[int | None, list[str]]:
    """The number on a ``مادة N`` first line (after R9), and the lines after it."""
    if ar_lines:
        m = AR_HEADER_DIGITS.match(ar_lines[0])
        if m:
            return int(m.group(1).translate(AR_DIGITS)), ar_lines[1:]
    return None, ar_lines


def _lines(text: str) -> list[str]:
    return [line for line in text.split("\n") if line]


def _anomaly(message: str, row: Row, row_class: str, number: int | None, **fields):
    log_anomaly(
        logger,
        message,
        page=row.page,
        row_index=row.row_index,
        row_class=row_class,
        article_number=number,
        **fields,
    )


@dataclass
class Assembly:
    """Everything ``assemble_rows`` produced."""

    articles: list[Article]
    headings: list[Heading]
    counts: dict[str, int]


def assemble_rows(rows: list[Row], root_en: str, root_ar: str) -> Assembly:
    """
    Turn repaired rows, in page and row order, into articles and headings.

    returns:
    - assembly (Assembly): the articles (sorted by number), every heading with
      the articles under it, and the stage counts
    """
    counts: Counter = Counter()
    root = Heading(ROOT_RANK, root_en, normalize_arabic(root_ar), 1, -1)
    headings: list[Heading] = [root]
    stack = HeadingStack(root)
    drafts: list[Draft] = []
    last: Draft | Heading | None = None  # the unit a continuation would join
    pending: Heading | None = None  # a keyword-only heading awaiting its title
    previous_page = None

    for row in rows:
        kind = classify_row(row)
        counts[f"rows_{kind}"] += 1
        new_page = row.page != previous_page
        previous_page = row.page

        if pending is not None and not (kind is RowClass.HEADING and new_page):
            _anomaly("Keyword-only heading never got its title", row, kind, None)
            counts["anomalies"] += 1
            pending = None

        if kind is RowClass.CONTINUATION:
            if isinstance(last, Draft):
                last.en_lines += _lines(row.en_text)
                last.ar_lines += _lines(row.ar_text)
                last.en_highlighted += row.en_highlighted
                last.ar_highlighted += row.ar_highlighted
                last.rows.append((row.page, row.row_index))
                counts["cross_page_merges"] += 1
                logger.info(
                    "Merged a continuation into article %d",
                    last.number,
                    extra={"page": row.page, "article_number": last.number},
                )
            else:
                _anomaly(
                    "Continuation with no article to merge into",
                    row,
                    kind,
                    None,
                    anomaly_type="orphan_continuation",
                )
                counts["anomalies"] += 1
            continue

        if kind is RowClass.ANOMALY:
            _anomaly("Row is not an article, heading or continuation", row, kind, None)
            counts["anomalies"] += 1
            continue

        if kind is RowClass.HEADING:
            en_units = heading_units(_lines(row.en_text), _en_unit)
            ar_units = heading_units(_lines(row.ar_text), _ar_unit)
            pairs, aligned = align_units(en_units, ar_units)
            if not aligned:
                _anomaly(
                    "Heading cell has different heading counts per language",
                    row,
                    kind,
                    None,
                    anomaly_type="heading_alignment",
                )
                counts["anomalies"] += 1
            if pending is not None and pairs:  # P19: the first heading is its title
                en_title, ar_title = pairs.pop(0)
                pending.en = " ".join(
                    filter(None, [pending.en, en_title and en_title.text])
                )
                pending.ar = " ".join(
                    filter(
                        None, [pending.ar, ar_title and normalize_arabic(ar_title.text)]
                    )
                )
                counts["titles_joined_across_pages"] += 1
            pending = None
            previous_rank = 0  # within one cell, each heading nests under the last
            for en, ar in pairs:
                rank, mismatch = heading_rank(en, ar)
                rank = max(rank, previous_rank + 1) if previous_rank else rank
                previous_rank = rank
                if mismatch:
                    _anomaly(
                        "Heading is numbered in one language only",
                        row,
                        kind,
                        None,
                        anomaly_type="numbering_mismatch",
                        en=en.text,
                        ar=ar.text,
                    )
                    counts["numbering_mismatches"] += 1
                heading = Heading(
                    rank,
                    en.text if en else "",
                    normalize_arabic(ar.text) if ar else "",
                    row.page,
                    row.row_index,
                )
                if row.en_highlighted and not ar:
                    heading.only_in = "en"
                elif row.ar_highlighted and not en:
                    heading.only_in = "ar"
                stack.push(heading)
                headings.append(heading)
                counts["headings"] += 1
                if (en and en.keyword_only) or (ar and ar.keyword_only):
                    pending = heading
            last = headings[-1]
            continue

        en_lines, ar_lines = _lines(row.en_text), _lines(row.ar_text)
        path_en, path_ar = stack.paths()
        if kind is RowClass.REPEAL:
            m = EN_RANGE.search(row.en_text)
            draft = Draft(
                int(m.group(1)),
                en_lines,
                ar_lines,
                path_en,
                path_ar,
                [(row.page, row.row_index)],
                repeal=(int(m.group(1)), int(m.group(2))),
            )
        else:
            m = EN_ARTICLE.match(en_lines[0])
            ar_number, ar_body = _ar_header_number(ar_lines)
            repeal = EN_RANGE.search(row.en_text)
            draft = Draft(
                int(m.group(3)),
                en_lines[1:],
                ar_body,
                path_en,
                path_ar,
                [(row.page, row.row_index)],
                ar_number=ar_number,
                repeal=(int(repeal.group(1)), int(repeal.group(2))) if repeal else None,
            )
        draft.en_highlighted = [
            h for h in row.en_highlighted if not HEADER_FRAGMENT.match(h)
        ]
        draft.ar_highlighted = [
            h for h in row.ar_highlighted if not HEADER_FRAGMENT.match(h)
        ]
        drafts.append(draft)
        first, final = draft.repeal or (draft.number, draft.number)
        stack.open[-1].articles += range(first, final + 1)
        last = draft

    articles = _records(drafts, rows, counts)
    counts["articles"] = len(articles)
    counts["repealed"] = sum(a.is_repealed for a in articles)
    return Assembly(articles, headings, dict(counts))


def _records(drafts: list[Draft], rows: list[Row], counts: Counter) -> list[Article]:
    """Expand repeals, check numbers, and build the ``Article`` records (R19, R20)."""
    by_position = {(r.page, r.row_index): r for r in rows}
    articles: dict[int, Article] = {}
    for d in drafts:
        numbers = range(d.repeal[0], d.repeal[1] + 1) if d.repeal else [d.number]
        first_row = by_position[d.rows[0]]
        if not d.repeal and d.ar_number is not None and d.ar_number != d.number:
            _anomaly(
                "Arabic header number differs from the English one",
                first_row,
                RowClass.ARTICLE,
                d.number,
                anomaly_type="number_mismatch",
                ar_number=d.ar_number,
            )
            counts["anomalies"] += 1
        only_in_en = _passages(
            d.en_lines, d.en_highlighted, first_row, d.number, counts
        )
        only_in_ar = _passages(
            d.ar_lines, d.ar_highlighted, first_row, d.number, counts
        )
        for number in numbers:
            if number in articles:
                _anomaly(
                    "Article number occurs twice",
                    first_row,
                    RowClass.ARTICLE,
                    number,
                    anomaly_type="duplicate_number",
                )
                counts["anomalies"] += 1
                continue
            articles[number] = Article(
                article_number=number,
                text_en="\n".join(d.en_lines),
                text_ar=normalize_arabic("\n".join(d.ar_lines)),
                heading_path=d.heading_path,
                heading_path_ar=d.heading_path_ar,
                source_pages=sorted({p for p, _ in d.rows}),
                source_rows=d.rows,
                is_repealed=d.repeal is not None,
                only_in_en=only_in_en,
                only_in_ar=[normalize_arabic(p) for p in only_in_ar],
            )
    return [articles[n] for n in sorted(articles)]


def _passages(
    lines: list[str], fragments: list[str], row: Row, number: int, counts: Counter
) -> list[str]:
    """
    Group an article's highlighted fragments into passages (R28).

    Fragments on consecutive lines of the article form one passage. A fragment
    found on no line is an anomaly, never dropped silently.

    returns:
    - passages (list[str]): each passage's text, lines joined with ``\n``
    """
    passages: list[list[str]] = []
    previous = None
    remaining = list(fragments)
    for i, line in enumerate(lines):
        hit = next((f for f in remaining if f in line), None)
        if hit is None:
            continue
        remaining.remove(hit)
        if previous is not None and i == previous + 1:
            passages[-1].append(hit)
        else:
            passages.append([hit])
        previous = i
    for fragment in remaining:
        _anomaly(
            "Highlighted text not found in its article",
            row,
            RowClass.ARTICLE,
            number,
            anomaly_type="highlight_not_found",
            fragment=fragment,
        )
        counts["anomalies"] += 1
    counts["untranslated_passages"] += len(passages)
    return ["\n".join(p) for p in passages]


def heading_tree(headings: list[Heading]) -> str:
    """
    Render the headings as an indented tree, for the owner to review (Gate 3).

    returns:
    - text (str): one heading per line, indented by rank, with its source row
      and the range of articles directly under it
    """
    lines = []
    for h in headings:
        span = f"{h.articles[0]}–{h.articles[-1]}" if h.articles else "–"
        where = "root" if h.row_index < 0 else f"p{h.page} r{h.row_index}"
        if h.only_in:
            where += f"; only in {'English' if h.only_in == 'en' else 'Arabic'}"
        lines.append(
            f"{'  ' * (h.rank - 1)}[{h.rank}] {h.en} | {h.ar}  ({where}; articles {span})"
        )
    return "\n".join(lines) + "\n"


def run_assemble(
    rows_in: Path, out: Path, tree_out: Path, metrics_out: Path, settings: Settings
) -> dict[str, int]:
    """
    Assemble ``rows_in`` into ``out`` (``Article`` records), the heading tree and counts.

    returns:
    - counts (dict[str, int]): what was written to ``metrics_out``
    """
    result = assemble_rows(
        read_records(rows_in, Row), settings.root_heading.en, settings.root_heading.ar
    )
    write_records(out, Article, result.articles)
    tree_out.parent.mkdir(parents=True, exist_ok=True)
    tree_out.write_text(heading_tree(result.headings), encoding="utf-8")
    metrics_out.parent.mkdir(parents=True, exist_ok=True)
    metrics_out.write_text(
        json.dumps(result.counts, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    logger.info(
        "Assembled %d articles under %d headings",
        result.counts["articles"],
        result.counts["headings"],
        extra={"event_type": LogEvent.STAGE_COMPLETED, **result.counts},
    )
    return result.counts


def main(argv: list[str] | None = None) -> None:
    settings = Settings()
    paths = settings.paths
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--rows", type=Path, default=paths.interim_dir / "rows_repaired.jsonl"
    )
    ap.add_argument("--out", type=Path, default=paths.corpus_dir / "articles.json")
    ap.add_argument("--tree", type=Path, default=paths.reports_dir / "heading_tree.txt")
    ap.add_argument("--metrics", type=Path, default=paths.metrics_dir / "assemble.json")
    args = ap.parse_args(argv)
    with stage_run(
        "assemble",
        input_hash=sha256_file(args.rows),
        output_model=Article,
        settings=settings,
    ) as run:
        counts = run_assemble(args.rows, args.out, args.tree, args.metrics, settings)
        run.log_metrics(counts)


if __name__ == "__main__":
    main()
