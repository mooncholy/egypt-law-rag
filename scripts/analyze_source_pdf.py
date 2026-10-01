"""Reproduce the source-document prerequisites (P1-P25) of the corpus build brief.

Usage:
    uv run python scripts/analyze_source_pdf.py data/raw/civil_code.pdf \
        --out docs/analysis/source_pdf

The output is gitignored. Check a rerun against the committed checksums with:
    sha256sum -c docs/reports/0_source_pdf_analysis.sha256

Outputs (in --out):
    summary.json            one entry per prerequisite ID, with the computed evidence
    pages.tsv               per page: tables, columns, rows, first-row class
    headings.tsv            every bold non-article row, with numbering/keyword features
    ar_header_variants.tsv  distinct shapes of the Arabic article header line
    number_mismatches.tsv   article rows whose Arabic header number != English number
    lam_alef_samples.tsv    sample contexts of the broken lam-alef glyph signature
    inline_citations.tsv    sample in-body references to other articles
    repeal_rows.tsv         rows that declare repealed article ranges
    extractor_comparison.txt  the same Arabic cell through three extraction paths

Dependencies: pymupdf, pikepdf; pdfplumber is optional (only for the comparison file).
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import re
from pathlib import Path

import pikepdf
import pymupdf

ALEFS = set("اأإآ")
AR_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
EN_ARTICLE = re.compile(r"^(A?)rticle(\s*)(\d+)\b\s*(.*)$")  # tolerates source typos
EN_RANGE = re.compile(
    r"Articles\s+(\d+)\s*[-\u2013]\s*(\d+)[^\n]*repealed", re.IGNORECASE
)
EN_NUMBERED = re.compile(r"^\d+\s*[.\-]")
AR_NUMBERED = re.compile(r"^[٠-٩]+\s*[-\u2013\u2014]")
KEYWORD = re.compile(r"\b(PART|BOOK|CHAPTER|SECTION)\b", re.IGNORECASE)
KEYWORD_ONLY = re.compile(
    r"^\s*(?:(?:PART|BOOK|CHAPTER|SECTION)\s+[IVXLC\d]+|[A-Z]+\s+PART)\s*$",
    re.IGNORECASE,
)
AR_INLINE_REF = re.compile(r"المادة\s*[٠-٩]+|المواد\s*[٠-٩]+")
AR_HEADER = re.compile(
    r"^[\s()]*م\s*ا\s*د\s*ة[\s()]*([٠-٩]+)"
)  # tolerates spaces inside مادة
AR_PARA_1 = re.compile(r"[()]\s*١\s*[()]")
EN_INLINE_REF = re.compile(r"(?<!^)\b[Aa]rticles?\s+\d+")


def is_bold(span: dict) -> bool:
    return "Bold" in span["font"] or bool(span["flags"] & 16)


def spans_in(page: pymupdf.Page, rect: pymupdf.Rect) -> list[dict]:
    d = page.get_text("dict", clip=rect)
    return [
        s
        for b in d["blocks"]
        for ln in b.get("lines", [])
        for s in ln["spans"]
        if s["text"].strip()
    ]


def lines_in(page: pymupdf.Page, rect: pymupdf.Rect) -> list[str]:
    text = page.get_text("text", clip=rect)
    return [ln.strip() for ln in text.splitlines() if ln.strip()]


def lam_alef_hits(page: pymupdf.Page, rect: pymupdf.Rect) -> list[str]:
    """Zero-width alef variant directly before lam, in rawdict character order."""
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
    """Classify multi-digit Arabic-Indic runs by the x-order of their characters."""
    order: collections.Counter = collections.Counter()
    raw = page.get_text("rawdict", clip=rect)
    for b in raw["blocks"]:
        for ln in b.get("lines", []):
            for s in ln["spans"]:
                run: list[float] = []
                for ch in s["chars"] + [{"c": " ", "bbox": (0, 0, 0, 0)}]:
                    if ch["c"] in "٠١٢٣٤٥٦٧٨٩":
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


def struct_tree_counts(path: Path) -> dict:
    with pikepdf.open(path) as pdf:
        pageidx = {pg.objgen: i for i, pg in enumerate(pdf.pages, start=1)}
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


def write_tsv(path: Path, header: list[str], rows: list[list]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(header)
        w.writerows(rows)


def analyze(pdf_path: Path, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    doc = pymupdf.open(pdf_path)

    fonts: set[str] = set()
    pages_rows, heading_rows, mismatch_rows = [], [], []
    repeal_rows, citation_rows, lam_rows = [], [], []
    shapes: collections.Counter = collections.Counter()
    kw_case: collections.Counter = collections.Counter()
    kw_pairs: collections.Counter = collections.Counter()
    en_numbers: list[int] = []
    pages_text = 0
    lam_total = 0
    ar_para_rows = en_para_rows = article_rows = 0
    header_on_own_line = 0
    extractor_sample = None
    same_line_headers: list[list] = []
    digit_order: collections.Counter = collections.Counter()
    typo_headers: list[list] = []

    for pno, page in enumerate(doc, start=1):
        fonts.update(f[3].split("+")[-1] for f in page.get_fonts())
        if page.get_text().strip():
            pages_text += 1
        # "lines_strict" ignores filled rectangles; the default "lines" strategy
        # turns yellow highlight fills into extra rows and columns.
        tables = page.find_tables(strategy="lines_strict").tables
        highlights = sum(
            1
            for dr in page.get_drawings()
            if dr.get("fill")
            and tuple(round(c, 2) for c in dr["fill"]) == (1.0, 1.0, 0.0)
        )
        default_rows = sum(t.row_count for t in page.find_tables().tables)
        table_rects = [pymupdf.Rect(t.bbox) for t in tables]
        outside = "".join(
            b[4]
            for b in page.get_text("blocks")
            if b[4].strip()
            and not any(pymupdf.Rect(b[:4]).intersects(r) for r in table_rects)
        ).strip()
        ncols = [t.col_count for t in tables]
        nrows = sum(t.row_count for t in tables)
        first_class = ""
        for t in tables:
            n_in_table = len(t.rows)
            for ri, row in enumerate(t.rows):
                cells = [pymupdf.Rect(c) for c in row.cells if c]
                mid = (t.bbox[0] + t.bbox[2]) / 2
                left = [c for c in cells if (c.x0 + c.x1) / 2 < mid]
                right = [c for c in cells if (c.x0 + c.x1) / 2 >= mid]
                if not left or not right:
                    continue
                # Some pages have spurious ruling lines that split a side into
                # several cells; each language is the union of its side.
                en_r, ar_r = pymupdf.Rect(left[0]), pymupdf.Rect(right[0])
                for c in left[1:]:
                    en_r |= c
                for c in right[1:]:
                    ar_r |= c
                en_lines, ar_lines = lines_in(page, en_r), lines_in(page, ar_r)
                en_spans = spans_in(page, en_r)
                en_text = "\n".join(en_lines)

                digit_order.update(digit_run_order(page, ar_r))
                hits = lam_alef_hits(page, ar_r)
                lam_total += len(hits)
                lam_rows += [[pno, ri, h] for h in hits[:2]]

                for m in EN_RANGE.finditer(en_text):
                    repeal_rows.append(
                        [
                            pno,
                            ri,
                            m.group(1),
                            m.group(2),
                            en_lines[0] if en_lines else "",
                        ]
                    )

                m_art = EN_ARTICLE.match(en_lines[0]) if en_lines else None
                if m_art:
                    cls = "article"
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
                        shapes[header_shape(ar_lines[0])] += 1
                    if ar_digits is not None:
                        ar_num = int(ar_digits.translate(AR_DIGITS))
                        if ar_num != en_num:
                            rev = int(ar_digits[::-1].translate(AR_DIGITS))
                            mismatch_rows.append(
                                [pno, ri, en_num, ar_digits, ar_num, rev == en_num]
                            )
                    else:
                        mismatch_rows.append(
                            [pno, ri, en_num, ar_lines[0] if ar_lines else "", "", ""]
                        )
                    body_ar = "\n".join(ar_lines[1:])
                    if AR_PARA_1.search(body_ar):
                        ar_para_rows += 1
                    if "(1)" in "\n".join(en_lines[1:]):
                        en_para_rows += 1
                    for m in AR_INLINE_REF.finditer(body_ar):
                        citation_rows.append([pno, ri, en_num, "ar", m.group(0)])
                    for ln in en_lines[1:]:
                        if EN_INLINE_REF.search(ln):
                            citation_rows.append([pno, ri, en_num, "en", ln[:90]])
                elif en_spans and all(is_bold(s) for s in en_spans):
                    cls = "heading"
                    kw = KEYWORD.search(en_lines[0]) if en_lines else None
                    if kw:
                        kw_case[kw.group(1)] += 1
                        ar_first = ar_lines[0].split()[0] if ar_lines else ""
                        kw_pairs[(kw.group(1).upper(), ar_first)] += 1
                    heading_rows.append(
                        [
                            pno,
                            ri,
                            ri == n_in_table - 1,
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
                        extractor_sample is None
                        and en_lines
                        and en_lines[0].upper().startswith("SECTION II")
                    ):
                        extractor_sample = (pno, t.bbox, ri, ar_r)
                else:
                    cls = "body"
                if not first_class:
                    first_class = "continuation" if cls == "body" else cls
        pages_rows.append(
            [
                pno,
                len(tables),
                ncols,
                nrows,
                first_class,
                highlights,
                default_rows,
                outside.replace("\n", " / ")[:300],
            ]
        )

    struct = struct_tree_counts(pdf_path)
    cont_pages = [r[0] for r in pages_rows if r[4] == "continuation"]
    max_article = max(en_numbers) if en_numbers else 0
    repealed = {n for r in repeal_rows for n in range(int(r[2]), int(r[3]) + 1)}
    missing = sorted(set(range(1, max_article + 1)) - set(en_numbers) - repealed)
    dup = [n for n, c in collections.Counter(en_numbers).items() if c > 1]

    summary = {
        "P1_file": {
            "metadata": doc.metadata,
            "page_count": doc.page_count,
            "pages_with_text_layer": pages_text,
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
        "P7_ar_header_shapes": dict(shapes.most_common()),
        "P8_P14_heading_rows": len(heading_rows),
        "P10_inline_citation_samples": len(citation_rows),
        "P11_paragraph_markers": {
            "article_rows_with_ar_marker_1": ar_para_rows,
            "article_rows_with_en_(1)": en_para_rows,
        },
        "P12_repeal_rows": [r[:4] for r in repeal_rows],
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
            "total": len(mismatch_rows),
            "explained_by_digit_reversal": sum(
                1 for r in mismatch_rows if r[5] is True
            ),
            "unparsed_ar_header": sum(1 for r in mismatch_rows if r[4] == ""),
            "multi_digit_runs_by_x_order": dict(digit_order),
        },
    }

    write_tsv(
        out / "pages.tsv", ["page", "tables", "cols", "rows", "first_row"], pages_rows
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
        heading_rows,
    )
    write_tsv(
        out / "ar_header_variants.tsv",
        ["shape", "count"],
        [[k, v] for k, v in shapes.most_common()],
    )
    write_tsv(
        out / "number_mismatches.tsv",
        ["page", "row", "en_number", "ar_digits_raw", "ar_number", "reversal_explains"],
        mismatch_rows,
    )
    write_tsv(out / "lam_alef_samples.tsv", ["page", "row", "context"], lam_rows[:200])
    write_tsv(
        out / "inline_citations.tsv",
        ["page", "row", "article", "lang", "text"],
        citation_rows[:200],
    )
    write_tsv(
        out / "repeal_rows.tsv",
        ["page", "row", "from", "to", "en_first_line"],
        repeal_rows,
    )

    comparison = ["P25: one Arabic heading cell through three extraction paths\n"]
    if extractor_sample:
        pno, tbbox, ri, ar_r = extractor_sample
        page = doc[pno - 1]
        comparison.append(f"page {pno}, row {ri}")
        comparison.append(
            "pymupdf get_text(clip):  " + repr(page.get_text("text", clip=ar_r))
        )
        for t in page.find_tables().tables:
            if t.bbox == tbbox:
                comparison.append(
                    "pymupdf Table.extract(): " + repr(t.extract()[ri][1])
                )
        try:
            import pdfplumber

            with pdfplumber.open(pdf_path) as pl:
                cells = pl.pages[pno - 1].extract_tables()[0]
                match = [
                    r[1]
                    for r in cells
                    if r[0] and r[0].upper().startswith("SECTION II")
                ]
                comparison.append(
                    "pdfplumber extract:      " + repr(match[0] if match else None)
                )
        except ImportError:
            comparison.append("pdfplumber not installed; comparison skipped")
    (out / "extractor_comparison.txt").write_text(
        "\n".join(comparison), encoding="utf-8"
    )
    (out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("pdf", type=Path)
    ap.add_argument("--out", type=Path, default=Path("docs/analysis/source_pdf"))
    args = ap.parse_args()
    summary = analyze(args.pdf, args.out)
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
