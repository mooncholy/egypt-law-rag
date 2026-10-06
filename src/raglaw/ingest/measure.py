"""Every measurement of the source PDF, in one place.

The ``profile`` stage runs these functions once per build. From that one pass
it writes the evidence behind ``docs/reports/0_source_pdf_analysis.md`` and
checks the same numbers against their baselines, so the report and the
pipeline's gates cannot drift apart.

The work splits in two. ``measure_document`` reads the PDF once into plain
per-page and per-row facts; everything after that (the summary, the evidence
files, the defect counts) is computed from those facts without touching the
PDF again.
"""

import collections
import csv
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pikepdf
import pymupdf

from raglaw.ingest.table import TableRow, iter_rows

ALEFS = frozenset("اأإآ")
AR_DIGIT_CHARS = "٠١٢٣٤٥٦٧٨٩"
AR_DIGITS = str.maketrans(AR_DIGIT_CHARS, "0123456789")
# Deliberately lenient and case-sensitive: the optional "A" and space catch the
# two header typos (P9), and a lowercase "article" is not a header.
EN_ARTICLE = re.compile(r"^(A?)rticle(\s*)(\d+)\b\s*(.*)$")
EN_RANGE = re.compile(r"Articles\s+(\d+)\s*[-–]\s*(\d+)[^\n]*repealed", re.IGNORECASE)
EN_NUMBERED = re.compile(r"^\d+\s*[.\-]")
AR_NUMBERED = re.compile(r"^[٠-٩]+\s*[-–—]")
KEYWORD = re.compile(r"\b(PART|BOOK|CHAPTER|SECTION)\b", re.IGNORECASE)
KEYWORD_ONLY = re.compile(
    r"^\s*(?:(?:PART|BOOK|CHAPTER|SECTION)\s+[IVXLC\d]+|[A-Z]+\s+PART)\s*$",
    re.IGNORECASE,
)
AR_INLINE_REF = re.compile(r"المادة\s*[٠-٩]+|المواد\s*[٠-٩]+")
# Tolerates spaces inside مادة and stray parentheses (P10).
AR_HEADER = re.compile(r"^[\s()]*م\s*ا\s*د\s*ة[\s()]*([٠-٩]+)")
# مادة with at least one space inside the word: م ادة, ما دة, ماد ة (P10).
AR_SPACED_MADA = re.compile(r"م\s+ا\s*د\s*ة|م\s*ا\s+د\s*ة|م\s*ا\s*د\s+ة")
AR_PARA_1 = re.compile(r"[()]\s*١\s*[()]")
EN_INLINE_REF = re.compile(r"(?<!^)\b[Aa]rticles?\s+\d+")
YELLOW = (1.0, 1.0, 0.0)


# --- Reading the page -------------------------------------------------------


def is_bold(span: dict) -> bool:
    return "Bold" in span["font"] or bool(span["flags"] & 16)


def spans_in(page: pymupdf.Page, rect: pymupdf.Rect) -> list[dict]:
    """Non-blank text spans inside ``rect``, with their font and flags."""
    d = page.get_text("dict", clip=rect)
    return [
        s
        for b in d["blocks"]
        for ln in b.get("lines", [])
        for s in ln["spans"]
        if s["text"].strip()
    ]


def lines_in(page: pymupdf.Page, rect: pymupdf.Rect) -> list[str]:
    """Stripped, non-empty text lines inside ``rect``, in logical order (P25)."""
    text = page.get_text("text", clip=rect)
    return [ln.strip() for ln in text.splitlines() if ln.strip()]


def lam_alef_hits(page: pymupdf.Page, rect: pymupdf.Rect) -> list[str]:
    """
    Find the broken lam-alef signature (P23) inside ``rect``.

    The signature is a zero-width alef variant directly before ``ل``, within
    one span, in ``rawdict`` character order.

    returns:
    - contexts (list[str]): a few characters around each hit, one per hit
    """
    hits = []
    raw = page.get_text("rawdict", clip=rect)
    for b in raw["blocks"]:
        for ln in b.get("lines", []):
            for s in ln["spans"]:
                chars = s["chars"]
                for i in range(len(chars) - 1):
                    c, nxt = chars[i], chars[i + 1]
                    width = c["bbox"][2] - c["bbox"][0]
                    if c["c"] in ALEFS and width < 0.01 and nxt["c"] == "ل":
                        ctx = "".join(ch["c"] for ch in chars[max(0, i - 6) : i + 8])
                        hits.append(ctx)
    return hits


def digit_run_order(page: pymupdf.Page, rect: pymupdf.Rect) -> collections.Counter:
    """
    Classify each multi-digit Arabic-Indic run inside ``rect`` by the x-order
    of its characters (P24).

    returns:
    - order (Counter): run counts under ``descending_x`` (stored right to
      left), ``ascending_x`` and ``mixed``
    """
    order: collections.Counter = collections.Counter()
    raw = page.get_text("rawdict", clip=rect)
    for b in raw["blocks"]:
        for ln in b.get("lines", []):
            for s in ln["spans"]:
                run: list[float] = []
                for ch in s["chars"] + [{"c": " ", "bbox": (0, 0, 0, 0)}]:
                    if ch["c"] in AR_DIGIT_CHARS:
                        run.append(ch["bbox"][0])
                        continue
                    if len(run) > 1:
                        if run == sorted(run, reverse=True):
                            order["descending_x"] += 1
                        elif run == sorted(run):
                            order["ascending_x"] += 1
                        else:
                            order["mixed"] += 1
                    run = []
    return order


def parse_ar_header(ar_lines: list[str]) -> str | None:
    """Digit string of the Arabic header; the number often sits on the next line."""
    m = AR_HEADER.match(" ".join(ar_lines[:3]))
    return m.group(1) if m else None


def header_shape(line: str) -> str:
    return re.sub(r"[٠-٩]+", "N", line)


# --- Reading the tag tree ---------------------------------------------------


def _element_pages(node: pikepdf.Dictionary, pageidx: dict, cur: int | None) -> set:
    """Pages an element's content sits on, from /Pg on it or its descendants."""
    pg = node.get("/Pg")
    cur = pageidx.get(pg.objgen, cur) if pg is not None else cur
    pages = {cur} if cur else set()
    kids = node.get("/K")
    kids = list(kids) if isinstance(kids, pikepdf.Array) else [kids]
    for k in kids:
        if isinstance(k, pikepdf.Dictionary):
            if k.get("/Type") == "/MCR" and k.get("/Pg") is not None:
                pages.add(pageidx.get(k.Pg.objgen))
            elif k.get("/Type") != "/MCR":
                pages |= _element_pages(k, pageidx, cur)
    return pages


def struct_tree_counts(path: Path, first_page: int = 1) -> dict:
    """
    Count the PDF's structure tags, and the pages each table row (``TR``) sits on.

    returns:
    - counts (dict): ``marked`` (tagged PDF flag), ``tags`` (count per tag
      name), ``tr_per_page`` and ``multi_page_trs`` (rows on more than one page)
    """
    with pikepdf.open(path) as pdf:
        pageidx = {pg.objgen: i for i, pg in enumerate(pdf.pages, start=first_page)}
        tr_pages: collections.Counter = collections.Counter()
        multi_page_trs = 0
        root = pdf.Root
        marked = bool(root.get("/MarkInfo", {}).get("/Marked", False))
        counts: collections.Counter = collections.Counter()
        if "/StructTreeRoot" in root:
            stack = [root.StructTreeRoot]
            while stack:
                node = stack.pop()
                if not isinstance(node, pikepdf.Dictionary):
                    continue
                if "/S" in node:
                    counts[str(node.S).lstrip("/")] += 1
                    if str(node.S) == "/TR":
                        pages = _element_pages(node, pageidx, None)
                        multi_page_trs += len(pages) > 1
                        tr_pages.update(pages)
                kids = node.get("/K")
                if isinstance(kids, pikepdf.Array):
                    stack.extend(k for k in kids if isinstance(k, pikepdf.Dictionary))
                elif isinstance(kids, pikepdf.Dictionary):
                    stack.append(kids)
    return {
        "marked": marked,
        "tags": dict(counts),
        "tr_per_page": dict(tr_pages),
        "multi_page_trs": multi_page_trs,
    }


# --- Facts ------------------------------------------------------------------


@dataclass
class RowFacts:
    """What one detected row with both sides holds."""

    page: int
    index: int
    is_last: bool
    table_bbox: tuple[float, float, float, float]
    ar_rect: pymupdf.Rect
    en_lines: list[str]
    ar_lines: list[str]
    en_all_bold: bool
    lam_alef_hits: list[str]
    digit_order: collections.Counter

    @property
    def article_match(self) -> re.Match | None:
        return EN_ARTICLE.match(self.en_lines[0]) if self.en_lines else None

    @property
    def kind(self) -> str:
        """``article`` (header-form first line), ``heading`` (all bold) or ``body``."""
        if self.article_match:
            return "article"
        return "heading" if self.en_all_bold else "body"


@dataclass
class PageFacts:
    """What one page holds: its fonts, tables, drawings and rows."""

    number: int
    fonts: set[str]
    has_text: bool
    n_tables: int
    col_counts: list[int]
    n_rows: int
    highlights: int
    default_strategy_rows: int
    text_outside_table: str
    rows: list[RowFacts]
    rows_missing_a_side: list[int] = field(default_factory=list)

    @property
    def first_row_class(self) -> str:
        """The first row's kind, with a leading body row named ``continuation`` (P5)."""
        if not self.rows:
            return ""
        kind = self.rows[0].kind
        return "continuation" if kind == "body" else kind


@dataclass
class SourceFacts:
    """Everything ``measure_document`` read from one PDF."""

    path: Path
    first_page: int
    metadata: dict[str, Any]
    page_count: int
    pages: list[PageFacts]
    struct: dict


def measure_row(page: pymupdf.Page, row: TableRow) -> RowFacts:
    """
    Read one row's text, fonts and glyph defects.

    returns:
    - facts (RowFacts): the row's lines per side, whether every English span
      is bold, and its lam-alef and digit-order defects
    """
    en_spans = spans_in(page, row.en_rect)
    return RowFacts(
        page=row.page,
        index=row.index,
        is_last=row.is_last,
        table_bbox=row.table_bbox,
        ar_rect=row.ar_rect,
        en_lines=lines_in(page, row.en_rect),
        ar_lines=lines_in(page, row.ar_rect),
        en_all_bold=bool(en_spans) and all(is_bold(s) for s in en_spans),
        lam_alef_hits=lam_alef_hits(page, row.ar_rect),
        digit_order=digit_run_order(page, row.ar_rect),
    )


def measure_page(page: pymupdf.Page, number: int) -> PageFacts:
    """
    Read one page's fonts, tables, highlight fills and rows.

    Rows missing a side are listed in ``rows_missing_a_side`` instead of being
    measured, so they can be reported.

    returns:
    - facts (PageFacts): the page's facts, ``number`` being its source page
    """
    # "lines_strict" ignores filled rectangles; the default "lines" strategy
    # turns yellow highlight fills into extra rows and columns (P6).
    tables = page.find_tables(strategy="lines_strict").tables
    table_rects = [pymupdf.Rect(t.bbox) for t in tables]
    outside = "".join(
        b[4]
        for b in page.get_text("blocks")
        if b[4].strip()
        and not any(pymupdf.Rect(b[:4]).intersects(r) for r in table_rects)
    ).strip()
    rows, missing = [], []
    for row in iter_rows(page, number):
        if row.has_both_sides:
            rows.append(measure_row(page, row))
        else:
            missing.append(row.index)
    return PageFacts(
        number=number,
        fonts={f[3].split("+")[-1] for f in page.get_fonts()},
        has_text=bool(page.get_text().strip()),
        n_tables=len(tables),
        col_counts=[t.col_count for t in tables],
        n_rows=sum(t.row_count for t in tables),
        highlights=sum(
            1
            for dr in page.get_drawings()
            if dr.get("fill") and tuple(round(c, 2) for c in dr["fill"]) == YELLOW
        ),
        default_strategy_rows=sum(t.row_count for t in page.find_tables().tables),
        text_outside_table=outside,
        rows=rows,
        rows_missing_a_side=missing,
    )


def measure_document(path: Path, first_page: int = 1) -> SourceFacts:
    """
    Read every page of a PDF into facts.

    ``first_page`` is the source page number of the file's first page, so an
    excerpt's facts carry the same page numbers as the full document.

    returns:
    - facts (SourceFacts): per-page facts plus the file's metadata and tag tree
    """
    with pymupdf.open(path) as doc:
        pages = [
            measure_page(page, number)
            for number, page in enumerate(doc, start=first_page)
        ]
        metadata, page_count = doc.metadata, doc.page_count
    return SourceFacts(
        path=path,
        first_page=first_page,
        metadata=metadata,
        page_count=page_count,
        pages=pages,
        struct=struct_tree_counts(path, first_page),
    )


def iter_row_facts(facts: SourceFacts):
    for page in facts.pages:
        yield from page.rows


# --- Defect counts (G6, C10) ------------------------------------------------


def defect_counts(facts: SourceFacts) -> dict[str, int]:
    """
    Count every defect class a repair targets.

    ``profile`` checks these against the raw baselines (G6).

    returns:
    - counts (dict[str, int]): ``lam_alef_signatures`` (P23),
      ``rtl_digit_runs`` (P24), ``header_typos`` (P9), ``same_line_headers``
      (P8) and ``spaced_mada_headers`` (P10)
    """
    counts = collections.Counter()
    for row in iter_row_facts(facts):
        counts["lam_alef_signatures"] += len(row.lam_alef_hits)
        counts["rtl_digit_runs"] += row.digit_order["descending_x"]
        m = row.article_match
        if not m:
            continue
        counts["header_typos"] += not m.group(1) or not m.group(2)
        counts["same_line_headers"] += bool(m.group(4))
        counts["spaced_mada_headers"] += bool(
            row.ar_lines and AR_SPACED_MADA.search(row.ar_lines[0])
        )
    keys = (
        "lam_alef_signatures",
        "rtl_digit_runs",
        "header_typos",
        "same_line_headers",
        "spaced_mada_headers",
    )
    return {k: counts[k] for k in keys}


# --- The analysis report's evidence ------------------------------------------


@dataclass
class Evidence:
    """The rows of each evidence file, and where the P25 sample cell sits."""

    pages: list[list] = field(default_factory=list)
    headings: list[list] = field(default_factory=list)
    shapes: collections.Counter = field(default_factory=collections.Counter)
    mismatches: list[list] = field(default_factory=list)
    lam_alef: list[list] = field(default_factory=list)
    citations: list[list] = field(default_factory=list)
    repeals: list[list] = field(default_factory=list)
    extractor_sample: RowFacts | None = None


def summarize(facts: SourceFacts) -> tuple[dict, Evidence]:
    """
    Compute the analysis summary (one entry per prerequisite) and its evidence.

    returns:
    - summary (dict): the content of ``summary.json``, keyed as the report cites it
    - evidence (Evidence): the rows behind each TSV file
    """
    ev = Evidence()
    kw_case: collections.Counter = collections.Counter()
    kw_pairs: collections.Counter = collections.Counter()
    digit_order: collections.Counter = collections.Counter()
    en_numbers: list[int] = []
    lam_total = ar_para_rows = en_para_rows = article_rows = header_on_own_line = 0
    same_line_headers: list[list] = []
    typo_headers: list[list] = []

    for page in facts.pages:
        pno = page.number
        for row in page.rows:
            ri, en_lines, ar_lines = row.index, row.en_lines, row.ar_lines
            en_text = "\n".join(en_lines)
            digit_order.update(row.digit_order)
            lam_total += len(row.lam_alef_hits)
            ev.lam_alef += [[pno, ri, h] for h in row.lam_alef_hits[:2]]

            for m in EN_RANGE.finditer(en_text):
                first = en_lines[0] if en_lines else ""
                ev.repeals.append([pno, ri, m.group(1), m.group(2), first])

            m_art = row.article_match
            if m_art:
                article_rows += 1
                en_num = int(m_art.group(3))
                if not m_art.group(4):
                    header_on_own_line += 1
                else:
                    same_line_headers.append([pno, ri, en_num, en_lines[0][:80]])
                if not m_art.group(1) or not m_art.group(2):
                    typo_headers.append([pno, ri, en_lines[0][:40]])
                en_numbers.append(en_num)
                ar_digits = parse_ar_header(ar_lines)
                if ar_lines:
                    ev.shapes[header_shape(ar_lines[0])] += 1
                if ar_digits is not None:
                    ar_num = int(ar_digits.translate(AR_DIGITS))
                    if ar_num != en_num:
                        rev = int(ar_digits[::-1].translate(AR_DIGITS))
                        ev.mismatches.append(
                            [pno, ri, en_num, ar_digits, ar_num, rev == en_num]
                        )
                else:
                    ev.mismatches.append(
                        [pno, ri, en_num, ar_lines[0] if ar_lines else "", "", ""]
                    )
                body_ar = "\n".join(ar_lines[1:])
                if AR_PARA_1.search(body_ar):
                    ar_para_rows += 1
                if "(1)" in "\n".join(en_lines[1:]):
                    en_para_rows += 1
                for m in AR_INLINE_REF.finditer(body_ar):
                    ev.citations.append([pno, ri, en_num, "ar", m.group(0)])
                for ln in en_lines[1:]:
                    if EN_INLINE_REF.search(ln):
                        ev.citations.append([pno, ri, en_num, "en", ln[:90]])
            elif row.en_all_bold:
                kw = KEYWORD.search(en_lines[0]) if en_lines else None
                if kw:
                    kw_case[kw.group(1)] += 1
                    ar_first = ar_lines[0].split()[0] if ar_lines else ""
                    kw_pairs[(kw.group(1).upper(), ar_first)] += 1
                ev.headings.append(
                    [
                        pno,
                        ri,
                        row.is_last,
                        len(en_lines),
                        " / ".join(en_lines),
                        " / ".join(ar_lines),
                        bool(en_lines and EN_NUMBERED.match(en_lines[0])),
                        bool(ar_lines and AR_NUMBERED.match(ar_lines[0])),
                        bool(en_lines and KEYWORD_ONLY.match(en_lines[0])),
                        [ln.endswith(":") for ln in en_lines],
                    ]
                )
                if (
                    ev.extractor_sample is None
                    and en_lines
                    and en_lines[0].upper().startswith("SECTION II")
                ):
                    ev.extractor_sample = row
        ev.pages.append(
            [
                pno,
                page.n_tables,
                page.col_counts,
                page.n_rows,
                page.first_row_class,
                page.highlights,
                page.default_strategy_rows,
                page.text_outside_table.replace("\n", " / ")[:300],
            ]
        )

    struct = facts.struct
    pages_rows, heading_rows = ev.pages, ev.headings
    cont_pages = [r[0] for r in pages_rows if r[4] == "continuation"]
    max_article = max(en_numbers) if en_numbers else 0
    repealed = {n for r in ev.repeals for n in range(int(r[2]), int(r[3]) + 1)}
    missing = sorted(set(range(1, max_article + 1)) - set(en_numbers) - repealed)
    dup = [n for n, c in collections.Counter(en_numbers).items() if c > 1]
    fonts = set().union(*(p.fonts for p in facts.pages))

    summary = {
        "P1_file": {
            "metadata": facts.metadata,
            "page_count": facts.page_count,
            "pages_with_text_layer": sum(p.has_text for p in facts.pages),
            "tagged_marked": struct["marked"],
        },
        "P2_fonts": sorted(fonts),
        "P3_tables_per_page": {
            "pages_with_exactly_one_table": sum(1 for r in pages_rows if r[1] == 1),
            "pages_all_tables_2_cols": sum(
                1 for r in pages_rows if r[2] and all(c == 2 for c in r[2])
            ),
            "total_pages": len(pages_rows),
        },
        "P4_struct_tree": {
            k: struct["tags"].get(k, 0) for k in ("Table", "TR", "TH", "TD")
        },
        "P4_rows": {
            "detected_lines_strict": sum(r[3] for r in pages_rows),
            "detected_default_strategy": sum(r[6] for r in pages_rows),
            "tagged_TR": sum(struct["tr_per_page"].values()),
            "tagged_TR_spanning_pages": struct["multi_page_trs"],
            "pages_where_detected_ne_tagged": [
                [r[0], r[3], struct["tr_per_page"].get(r[0], 0)]
                for r in pages_rows
                if r[3] != struct["tr_per_page"].get(r[0], 0)
            ],
            "pages_with_highlight_fills": [r[0] for r in pages_rows if r[5]],
            "pages_with_text_outside_table": [r[0] for r in pages_rows if r[7]],
        },
        "P5_continuation_first_rows": {
            "pages": len(cont_pages),
            "in_first_60_pages": sum(1 for p in cont_pages if p <= 60),
        },
        "P6_article_rows": {
            "total": article_rows,
            "header_alone_on_first_line": header_on_own_line,
            "header_and_body_on_same_line": same_line_headers,
            "header_typos": typo_headers,
        },
        "P7_ar_header_shapes": dict(ev.shapes.most_common()),
        "P8_P14_heading_rows": len(heading_rows),
        "P10_inline_citation_samples": len(ev.citations),
        "P11_paragraph_markers": {
            "article_rows_with_ar_marker_1": ar_para_rows,
            "article_rows_with_en_(1)": en_para_rows,
        },
        "P12_repeal_rows": [r[:4] for r in ev.repeals],
        "P13_article_numbers": {
            "max_english_number": max_article,
            "distinct": len(set(en_numbers)),
            "duplicates": dup,
            "missing_not_repealed": missing,
        },
        "P16_keyword_case_variants": dict(kw_case),
        "P17_multiline_heading_cells": sum(1 for h in heading_rows if h[3] > 1),
        "P18_keyword_only_heading_as_last_row": [
            [h[0], h[4]] for h in heading_rows if h[2] and h[8]
        ],
        "P19_unnumbered_headings_without_colon": sum(
            1 for h in heading_rows if not h[6] and not h[8] and not h[9][-1]
        ),
        "P20_numbering_mismatch_en_vs_ar": [
            [h[0], h[4], h[5]] for h in heading_rows if h[6] != h[7] and not h[8]
        ],
        "P21_keyword_pairs_en_ar": {f"{k[0]}|{k[1]}": v for k, v in kw_pairs.items()},
        "P22_P23_lam_alef_signature_count": lam_total,
        "P24_header_number_mismatches": {
            "total": len(ev.mismatches),
            "explained_by_digit_reversal": sum(
                1 for r in ev.mismatches if r[5] is True
            ),
            "unparsed_ar_header": sum(1 for r in ev.mismatches if r[4] == ""),
            "multi_digit_runs_by_x_order": dict(digit_order),
        },
    }
    return summary, ev


def write_tsv(path: Path, header: list[str], rows: list[list]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(header)
        w.writerows(rows)


def extractor_comparison(facts: SourceFacts, sample: RowFacts | None) -> str:
    """
    Read one Arabic heading cell through three extraction paths (P25).

    returns:
    - text (str): the content of ``extractor_comparison.txt``
    """
    lines = ["P25: one Arabic heading cell through three extraction paths\n"]
    if sample is None:
        return "\n".join(lines)
    path, index = facts.path, sample.page - facts.first_page
    with pymupdf.open(path) as doc:
        page = doc[index]
        lines.append(f"page {sample.page}, row {sample.index}")
        lines.append(
            "pymupdf get_text(clip):  "
            + repr(page.get_text("text", clip=sample.ar_rect))
        )
        for t in page.find_tables().tables:
            if tuple(t.bbox) == sample.table_bbox:
                lines.append(
                    "pymupdf Table.extract(): " + repr(t.extract()[sample.index][1])
                )
    try:
        import pdfplumber

        with pdfplumber.open(path) as pl:
            cells = pl.pages[index].extract_tables()[0]
            match = [
                r[1] for r in cells if r[0] and r[0].upper().startswith("SECTION II")
            ]
            lines.append(
                "pdfplumber extract:      " + repr(match[0] if match else None)
            )
    except ImportError:
        lines.append("pdfplumber not installed; comparison skipped")
    return "\n".join(lines)


def write_evidence(facts: SourceFacts, out: Path) -> dict:
    """
    Write ``summary.json`` and the evidence files behind the analysis report.

    The output must stay byte-identical for the same PDF:
    ``docs/reports/0_source_pdf_analysis.sha256`` pins it.

    returns:
    - summary (dict): what was written to ``summary.json``
    """
    out.mkdir(parents=True, exist_ok=True)
    summary, ev = summarize(facts)
    write_tsv(
        out / "pages.tsv", ["page", "tables", "cols", "rows", "first_row"], ev.pages
    )
    write_tsv(
        out / "headings.tsv",
        [
            "page",
            "row",
            "last_row_on_page",
            "n_lines",
            "en",
            "ar",
            "en_numbered",
            "ar_numbered",
            "keyword_only",
            "en_lines_end_colon",
        ],
        ev.headings,
    )
    write_tsv(
        out / "ar_header_variants.tsv",
        ["shape", "count"],
        [[k, v] for k, v in ev.shapes.most_common()],
    )
    write_tsv(
        out / "number_mismatches.tsv",
        ["page", "row", "en_number", "ar_digits_raw", "ar_number", "reversal_explains"],
        ev.mismatches,
    )
    write_tsv(
        out / "lam_alef_samples.tsv", ["page", "row", "context"], ev.lam_alef[:200]
    )
    write_tsv(
        out / "inline_citations.tsv",
        ["page", "row", "article", "lang", "text"],
        ev.citations[:200],
    )
    write_tsv(
        out / "repeal_rows.tsv",
        ["page", "row", "from", "to", "en_first_line"],
        ev.repeals,
    )
    (out / "extractor_comparison.txt").write_text(
        extractor_comparison(facts, ev.extractor_sample), encoding="utf-8"
    )
    (out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    return summary
