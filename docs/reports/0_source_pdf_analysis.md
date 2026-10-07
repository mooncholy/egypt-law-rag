
# Source Document Analysis

Every decision taken in writing the unittests and ingestion pipeline is based on the following facts observed on `data/raw/civil_code.pdf`. The `profile` stage (`src/raglaw/ingest/profile.py`, with the measurements in `measure.py`) measures them over all 170 pages and writes the evidence to `docs/analysis/source_pdf/`.
- **Measured** means the code computes it; the Evidence column names the field in `summary.json` or the output file.
- **Judgment** means a person decided it; the gate that confirms it is named.

The evidence files are not committed. Regenerate them (needs `dvc pull` for the PDF) and check they match the analyzed version:

```bash
uv run dvc repro profile
sha256sum -c docs/reports/0_source_pdf_analysis.sha256
```

Every line must read `OK`. A mismatch means the PDF or the measurement code changed since this report was written, and the facts below must be re-checked.

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
| P10 | The Arabic header holds `مادة` and its number, with stray parentheses. Text extraction showed the number on the next line, but all 1,093 sit on the same visual line (corrected in Phase 2, P30). 13 headers have spaces inside the word: `م ادة` (7), `ما دة` (5), `ماد ة` (1). Those spaces are misplaced (P30): `extract` moves them and all 13 read `مادة` | Measured | `ar_header_variants.tsv` |
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
| P24 | Every multi-digit Arabic-Indic run (1,167) is stored right to left. Sorting its digits by ascending `x0` restores the order, which explains 987 of 990 header mismatches. The other 3: the page 7 repeal row, the page 147 typo row, and Article 601 (page 81), whose number is split by a misplaced space (`٠٦ ١`; P30 now joins it) | Measured | `P24_header_number_mismatches`, `number_mismatches.tsv` |
| P25 | `find_tables().extract()` returns Arabic in reversed visual order; `get_text(clip=cell_bbox)` returns logical order | Measured | `extractor_comparison.txt` |
| P26 | Spurious spaces also occur inside ordinary Arabic words (`قان ون`, page 4). Not measured, and no safe rule exists to repair them, so they are left as they are | Judgment | — |
| P27 | 25 private-use glyphs stand for a letter plus `ه`: U+E812 = `به` (×7, e.g., `بهذا`), U+E814 = `ته` (×13, e.g., `المرتهن`), U+E815 = `نه` (×4, e.g., `نهائي`), U+E811 = `لمج` (×1, `المجدد`, the owner's reading). Each is zero-width and begins its own text piece; in 17 of them a space stored right after the glyph belongs, visually, at the end of the piece (the word gap). `rawdict` keeps all 25; `get_text("text")` duplicates some. Fixed by errata E4–E28 | Measured (count); the letters are Judgment | `extract.json` (`private_use_glyphs`) |
| P28 | 26 zero-width alefs follow no lam after the swap, mostly at the end of headings (`حق الملكية أ`). They are invisible when printed, and each duplicates a lam-alef placeholder at the start of the next row. Dropped by R25 | Measured | `extract.json` (`stray_zero_width_alefs`) |
| P29 | 58 rows have an empty Arabic cell and 1 an empty English cell (page 81, row 5). Article 1022's Arabic cell (page 147, row 6) is empty | Measured | `extract.json` (`rows_empty_ar`, `rows_empty_en`) |
| P30 | PyMuPDF splits one visual line into several lines whenever a text run starts to the left of the previous one, which right-to-left text does constantly: 5,051 Arabic pieces (1,618 of them mid-word, e.g., `الثان` + `ي`) and 7 English pieces (stored right to left, e.g., Article 1022). `get_text("text")` splits the same way. Inside Arabic pieces, a space is often stored before characters drawn to its right (`' مصادر'` is drawn `'مصادر '`): 7,144 spaces, including the 13 spaced `مادة` (P10) and Article 601's split number (P24). `extract` joins the pieces of each visual line and moves those spaces (R24) | Measured | `extract.json` (`line_pieces_merged_ar`, `line_pieces_merged_en`, `piece_spaces_moved`) |
| P31 | 2 ligatures are drawn with no character at all, so their letters are missing from the text layer: page 21 (`اداما`, reads as `انهداما`) and page 169 (`أما`, reads as `أنهما`). Found in Phase 2; errata entries await the owner | Measured (by the gap at the seam); the letters are Judgment | — |
| P32 | The yellow fills (P6) mark text with no direct counterpart in the other language: English-only passages in Articles 84, 499, 1022 and 1060, and the heading `SECOND PART / REAL RIGHTS`; Arabic-only passages in Articles 970, 1021 (paragraphs ٢ and ٣) and 1085, and the heading `موت المستأجر أو إعساره`. Articles 1021 and 1022 are not treated as a misalignment: their paragraph numbers differ | Measured (location); the meaning is the owner's reading | `extract.json` (`highlighted_lines_en`, `highlighted_lines_ar`), `assemble.json` (`untranslated_passages`) |

# Subsequent Decisions

The pipeline follows these rules; each rests on the facts above. **Decided** rules follow directly from measured facts. **Judgment** rules are a reading of the source that the named gate confirms.

## 1. Extraction (`extract`)

| ID | Rule | Based on | Status |
| --- | --- | --- | --- |
| R1 | Segment by table rows only: one detected row is one unit. No regex over whole-page or whole-document text finds article boundaries | P3, P4, P13 | Decided |
| R2 | Detect tables with `find_tables(strategy="lines_strict")`, never the default `lines` strategy | P3, P6 | Decided |
| R3 | Read only inside the page's table. Text outside it is never read, which excludes the promulgation law | P7, D9 | Decided |
| R4 | Build cell text from `get_text("rawdict", clip=cell_bbox)`, never `Table.extract()`, joining the pieces of each visual line (P30) | P25, P30 | Decided |
| R5 | Lam-alef: a zero-width alef variant directly before `ل` swaps with it, before characters are joined. Exactly 3,500 swaps | P23 | Decided; spelling confirmed by the gold set |
| R6 | Digits: each multi-digit Arabic-Indic run is ordered by ascending `x0`. Exactly 1,167 runs | P24 | Decided |

## 2. Repair (`repair`, in this order)

| ID | Rule | Based on | Status |
| --- | --- | --- | --- |
| R7 | Errata first. An entry whose expected text is absent fails the stage | P9, P24 | Decided |
| R8 | Split same-line headers: `Article N` plus body text becomes a header line and a body line. Exactly 6 | P8 | Decided |
| R9 | Normalize the Arabic header: `مادة` with inner spaces or stray parentheses becomes one form, with the number on the same line. Exactly 13 spaced forms | P10 | Decided |
| R10 | No other edits to source text. Spurious spaces inside ordinary Arabic words stay as they are | P26 | Judgment |

## 3. Assembly (`assemble`)

| ID | Rule | Based on | Status |
| --- | --- | --- | --- |
| R11 | Classify each row: **article** if the English cell's first line is a header (`Article N`); **heading** if every English span is bold; **continuation** if neither and it is the first row on its page; otherwise **anomaly**, logged and never merged. A trailing colon is not a heading signal | P5, P8, P15, P20 | Decided |
| R12 | Only a header-form *first line* starts an article. Inline citations (`paragraph 2 of Article 717.`) never do | P13 | Decided |
| R13 | A continuation merges into the last row of the previous page, and every merge is logged. A continuation with no previous row to merge into is an `orphan_continuation` anomaly | P5 | Decided |
| R14 | Heading rank comes from the English keyword, case-insensitive: PART and the root 1, BOOK 2, CHAPTER 3, SECTION 4, numbered 5, unnumbered 6. A heading of rank *r* pops every open heading of rank ≥ *r* | P16, P17, P22 | Judgment, Gate 3 |
| R15 | A heading is numbered if either language numbers it; when they disagree, a `numbering_mismatch` anomaly is logged | P21 | Decided |
| R16 | In a multi-line heading cell, a keyword-only line plus the next line is one heading; a numbered line plus the next line is two | P18 | Judgment, Gate 3 |
| R17 | A keyword-only heading that ends a page waits for its title on the next page | P19 | Decided |
| R18 | The root heading `باب تمهيدي / أحكام عامة` opens every heading stack at rank 1; its English label comes from config | P7, D7 | Decided; label pending (D7) |
| R19 | Both repeal formats expand to one record per article (54–80, 389–417), each marked repealed | P12 | Decided |
| R20 | Records are keyed on the English article number; the Arabic number must agree after R6, R7 and R9 | P11, P24 | Decided |

## 4. Validation and chunking

| ID | Rule | Based on | Status |
| --- | --- | --- | --- |
| R21 | Article numbers run exactly 1 to 1,149, with no duplicates and no gaps | P11 | Decided |
| R22 | An article longer than `max_chars` splits only at Arabic paragraph markers (`(١)`, `(٢)`, …). A single paragraph longer than `max_chars` is an anomaly, not cut. The full English text goes on every part | P14, D4 | Decided |
| R23 | The `profile` stage re-measures the raw baselines above on every build; any mismatch stops `dvc repro` before a repair runs | All measured Ps | Decided |
| R24 | `extract` joins the pieces of each visual line (boxes overlapping vertically by at least half) in reading order, and moves each space of an Arabic piece past characters drawn to its right (never past a Latin letter or ASCII digit) | P30 | Decided (Phase 2) |
| R25 | `extract` drops a zero-width alef that follows no `ل`, before spaces move. Exactly 26 | P28 | Decided by the owner (D11) |
| R26 | A row with an empty English cell in the PDF is a heading when every Arabic span is bold; a row declaring a repealed range is a repeal, checked before headings | P12, P29 | Decided (Phase 3) |
| R27 | Refines R16: in a heading cell, a plain line is a wrapped title (joined onto the line before) when the other language has fewer headings; and a heading listed after another in the same cell nests under it | P18 | Judgment, Gate 3 |
| R28 | Highlighted text is kept, not corrected: each article carries its highlighted passages as `only_in_en` or `only_in_ar`, and a heading in one language keeps an empty label on the other side, marked in `heading_tree.txt`. Retrieval and answers surface them | P32 | Decided by the owner (Phase 3) |
| R29 | `chunk` never crosses an article and splits only before a line that opens with a paragraph marker. The `structural_semantic` variant also splits where neighbouring paragraphs drift apart, against one percentile threshold over the whole corpus. Every part carries the full English text (D4), both heading paths, a bilingual citation and its untranslated passages (R28) | R22, P14, P32 | Decided (Phase 5) |

## 5. Errata

Errata fix **one known error at one place** in the source, where a rule would be unsafe: each is a single occurrence with no pattern to generalize. They live in `data/errata.yaml`, tracked by git, not DVC.

**Rules for every entry**

- One entry per known source error: `page`, `row`, `side` (`en` or `ar`), `expect`, `replace`, `reason` (naming its P).
- `page` is the source page number (1-based). `row` is the 0-based index of the row in that page's `lines_strict` table, as in the analysis output.
- `expect` must equal a **whole line** of that cell, never part of one. `rticle 452` is a substring of its own fix, so substring matching would re-apply to corrected text. When the PDF split a word across a line break, `expect` may hold consecutive whole lines joined by `\n`, and `replace` the line or lines that take their place.
- An entry whose `expect` isn't found at its page, row and side **fails the stage**: a changed source stops the build instead of being patched in the wrong place.
- Errata apply first in `repair`, after the glyph repairs of `extract` (R5, R6), so `expect` is written in post-glyph-repair text.
- The owner approves every entry before it takes effect (Gate 2). The errata count in `docs/metrics/repair.json` must equal the number of entries.

**Entries**

| # | Page, row, side | Expect → replace | Reason | Status |
| --- | --- | --- | --- | --- |
| E1 | 59, 2, en | `rticle 452` → `Article 452` | Header typo: missing `A` (P9) | Proposed |
| E2 | 147, 6, en | `Article1022` → `Article 1022` | Header typo: missing space (P9) | Proposed |
| E3 | 81, 6, ar | — | Article 601's split number (P24) | Retired: R24 joins the number |
| E4–E28 | 25 rows, ar | Each visual line holding a private-use glyph, with the glyph mapped to its letters and its stored space moved back to the end of the word (listed in `data/errata.yaml`) | Private-use ligature glyphs (P27) | Proposed (pages 102 and 115 keep the owner's wording) |

E1 and E2 were re-checked against the PDF: the first English line at those rows is exactly `rticle 452` and `Article1022`.
