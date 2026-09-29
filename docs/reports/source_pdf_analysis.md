
# Source Document Analysis

Every decision taken in writing the unittests and ingestion pipeline is based on the following facts observed on `data/raw/civil_code.pdf`. `scripts/analyze_source_pdf.py` measures them over all 170 pages and writes the evidence to `docs/analysis/source_pdf/`.
- **Measured** means the script computes it; the Evidence column names the field in `summary.json` or the output file.
- **Judgment** means a person decided it; the gate that confirms it is named.

The evidence files are not committed. Regenerate them (needs `dvc pull` for the PDF) and check they match the analyzed version:

```bash
uv run python scripts/analyze_source_pdf.py data/raw/civil_code.pdf
sha256sum -c docs/reports/source_pdf_analysis.sha256
```

Every line must read `OK`. A mismatch means the PDF or the script changed since this report was written, and the facts below must be re-checked.

## 1. File and layout

| ID | Fact | Status | Evidence |
| --- | --- | --- | --- |
| P1 | Exported by Word for Office 365 (2019): 170 pages, a text layer on all 170, tagged PDF | Measured | `P1_file` |
| P2 | Fonts: Arial-BoldMT, ArialMT, Calibri, Calibri-Bold | Measured | `P2_fonts` |
| P3 | Every page is one ruled two-column table, English left and Arabic right, when detected with `find_tables(strategy="lines_strict")` | Measured (170 of 170) | `P3_tables_per_page` |
| P4 | The tag tree holds 171 `Table` and 1,464 `TR` elements. No `TR` spans two pages, so tags split page-crossing cells exactly as detection does. Detection finds 1,463 rows; the one extra `TR` is the page 1 block (P7) | Measured | `P4_rows`, `pages.tsv` |
| P5 | 146 of 170 pages open with a continuation row, the remainder of the previous page's last row | Measured (count); attachment confirmed by the gold set | `P5_continuation_first_rows` |
| P6 | Yellow highlight fills sit on 9 pages (8, 64, 81, 113, 140, 147, 148, 154, 158). The default `lines` strategy reads them as ruling lines and finds 1,483 rows with false columns, so detection must use `lines_strict` | Measured | `P4_rows`, `pages.tsv` |
| P7 | Page 1 opens with a borderless block outside the table. It holds the promulgation law (قانون الإصدار, Arabic only, with its own مادة ١ and مادة ٢), then the heading `باب تمهيدي / أحكام عامة`. The promulgation law is excluded from the corpus. The heading is kept as the fixed root of the preliminary section, at rank 1 | Measured (location); exclusion is the owner's decision | `pages.tsv` (`text_outside_table`) |

## 2. Articles

| ID | Fact | Status | Evidence |
| --- | --- | --- | --- |
| P8 | 1,094 article rows. The English cell starts with `Article N`: alone on its first line in 1,088 rows, followed by body text on the same line in 6 (Articles 277, 714, 746, 898, 908, 1090) | Measured | `P6_article_rows` |
| P9 | Two English headers have source typos: `rticle 452` (page 59) and `Article1022` (page 147) | Measured | `P6_article_rows.header_typos` |
| P10 | The Arabic header puts `مادة` on the first line and the number on the next, with stray parentheses. 13 headers have spaces inside the word: `م ادة` (7), `ما دة` (5), `ماد ة` (1) | Measured | `ar_header_variants.tsv` |
| P11 | English numbers run to 1,149 with no duplicates. 1,094 article rows plus 56 repealed articles, minus Article 54 (counted in both), gives exactly 1,149, with no other gaps | Measured | `P13_article_numbers` |
| P12 | Repealed ranges take two formats. Page 7: an `Article 54` row whose body reads `* Articles 54-80 have been repealed by Presidential Decree.` Page 53: a row reading `Articles 389-417 repealed`, with no header | Measured | `repeal_rows.tsv` |
| P13 | Bodies cite other articles inline, e.g., `paragraph 2 of Article 717.` (Article 194, page 23). An inline citation never starts a new article | Measured (116 samples) | `inline_citations.tsv` |
| P14 | 501 Arabic article bodies number their paragraphs `(١)`, `(٢)`, …, often with stray spaces (`(١ (`). No English body does | Measured | `P11_paragraph_markers` |

## 3. Headings

| ID | Fact | Status | Evidence |
| --- | --- | --- | --- |
| P15 | 222 heading rows, identified by every English span being bold | Measured | `P8_P14_heading_rows`, `headings.tsv` |
| P16 | Keyword case varies (`Section` 48, `SECTION` 5, `Chapter` 15, `CHAPTER` 2), and so do numbering separators (`1.`, `2-`, Arabic `١ -` and `١ –`) | Measured | `P16_keyword_case_variants` |
| P17 | English and Arabic keywords map consistently: SECTION to الفصل (53), CHAPTER to الباب (17), BOOK to الكتاب (4), PART to القسم (1) | Measured | `P21_keyword_pairs_en_ar` |
| P18 | 87 heading cells hold more than one line. A keyword-only line plus the next line forms one heading (`SECTION I` + `Laws and their Applications`). A numbered line plus the next line forms two (`1. Elements of Contracts` + `Consent:`) | Measured (count); reading is Judgment, confirmed at Gate 3 | `P17_multiline_heading_cells` |
| P19 | A keyword-only heading can end a page with its title opening the next (`Section II`, page 46 to 47) | Measured | `P18_keyword_only_heading_as_last_row` |
| P20 | 62 unnumbered headings have no trailing colon, so a colon is not a heading signal | Measured | `P19_unnumbered_headings_without_colon` |
| P21 | 9 headings disagree on numbering between the languages, in both directions (e.g., `Associations` / `٣ -الجمعيات` on page 7; `3. Insolvency` / `الإعسار` on page 31). Page 8 uses the Arabic ordinal word `أولاً` | Measured | `P20_numbering_mismatch_en_vs_ar` |
| P22 | Rank by type, highest first: PART and the page 1 root 1, BOOK 2, CHAPTER 3, SECTION 4, numbered 5, unnumbered 6 | Judgment, confirmed at Gate 3 | `headings.tsv` |

## 4. Text defects

| ID | Fact | Status | Evidence |
| --- | --- | --- | --- |
| P23 | Lam-alef is broken in every extractor because the PDF's font mapping is wrong (`السجالت` for `السجلات`). Its signature in PyMuPDF `rawdict`: a zero-width alef variant directly before `ل`. 3,500 occurrences | Measured (count). The correct spelling is Judgment; the swap fix is confirmed by the gold set | `P22_P23_lam_alef_signature_count`, `lam_alef_samples.tsv` |
| P24 | Every multi-digit Arabic-Indic run (1,167) is stored right to left. Sorting its digits by ascending `x0` restores the order, which explains 987 of 990 header mismatches. The other 3: the page 7 repeal row, the page 147 typo row, and Article 601 (page 81), whose number is split by a space (`٠٦ ١`) | Measured | `P24_header_number_mismatches`, `number_mismatches.tsv` |
| P25 | `find_tables().extract()` returns Arabic in reversed visual order; `get_text(clip=cell_bbox)` returns logical order | Measured | `extractor_comparison.txt` |
| P26 | Spurious spaces also occur inside ordinary Arabic words (`قان ون`, page 4). Not measured, and no safe rule exists to repair them, so they are left as they are | Judgment | — |
