# Arabic Normalization

**Status:** Approved by the owner (2026-10-07). Implemented in `raglaw/text/normalize.py`.

The pipeline repairs the source's known defects by rule or errata (R5 to R9), and R10 forbids any other edit to the source text. Normalization therefore cannot be a general clean-up step. It has two separate jobs, and each gets its own function in `raglaw/text/normalize.py`:

1. `normalize_arabic`: makes the stored text consistent at the code-point level without changing a single word. It runs in `assemble`, on every `text_ar` and Arabic heading, and on every query at query time.
2. `fold_arabic`: builds a matching key for lexical search, where spelling variants should match. It runs at index time and query time only. Its output is never stored as article text or shown to a user.

Both functions are idempotent (`f(f(x)) == f(x)`, test U15), and query-time code imports the same functions, so ingestion and queries cannot drift apart.

## 1. `normalize_arabic` (stored text)

Each rule removes or unifies code points that carry no meaning in this source. None of them changes a letter a reader would see.

| ID | Rule | Why |
| --- | --- | --- |
| N1 | Map Arabic presentation forms (U+FB50 to U+FDFF, U+FE70 to U+FEFF) to their base letters through NFKC, applied to those characters only | C13 requires zero presentation forms. Full-text NFKC would also rewrite unrelated characters (e.g., superscripts, full-width punctuation), so it is restricted to these two blocks |
| N2 | Apply NFC to the whole text | Composes letter + combining hamza or madda into the single code point (e.g., `ا` + U+0653 to `آ`), so identical-looking words compare equal |
| N3 | Remove tatweel (U+0640) | A justification stretch, not a letter |
| N4 | Remove zero-width and bidi control characters: U+200B to U+200F, U+202A to U+202E, U+2066 to U+2069, U+061C, U+FEFF | C13 allows no control characters other than `\n`. These only affect rendering |
| N5 | Replace every Unicode space (e.g., U+00A0, U+2009) with U+0020, collapse runs of spaces and tabs to one space, strip each line, drop empty lines, and remove `\r` | Makes whitespace comparable without touching line structure |

**Deliberately left unchanged** in stored text:

- Alef variants (`أ إ آ ا`), `ى` and `ي`, `ة` and `ه`: different spellings are different words in legal text, and R10 forbids rewriting them.
- Diacritics (tashkeel): kept as printed.
- Arabic-Indic digits: kept, since articles cite each other with them (`المادة ٧١٧`).
- Spaces inside ordinary words (`قان ون`, P26): no safe rule exists to repair them.
- Paragraph markers with stray spaces (`(١ (`, P14): chunking tolerates them, so they stay as printed.
- Line breaks: kept as `\n`. Whether `assemble` joins the PDF's line wraps into running text is an assembly decision, not a normalization one.

## 2. `fold_arabic` (matching key only)

`fold_arabic(x)` first applies `normalize_arabic`, then the rules below. They conflate spellings on purpose, so a query typed without hamza or with Western digits still matches the article.

| ID | Rule | Example |
| --- | --- | --- |
| F1 | Unify alef variants `أ إ آ ٱ` to `ا` | `الأشخاص` to `الاشخاص` |
| F2 | Map alef maqsura `ى` to `ي` | `على` to `علي` |
| F3 | Map ta marbuta `ة` to `ه` | `مادة` to `ماده` |
| F4 | Remove diacritics (U+064B to U+0652, U+0670) | `عَقْد` to `عقد` |
| F5 | Map Arabic-Indic (U+0660 to U+0669) and Extended Arabic-Indic (U+06F0 to U+06F9) digits to ASCII | `١٤٧` to `147` |

`fold_arabic` is for lexical matching (e.g., a BM25 index and its queries). Embedding models are trained on natural text, so embeddings use `normalize_arabic` output, not the folded key.

## 3. Decisions

The owner approved the draft as written, which settles its three open questions:

1. **F3 (`ة` to `ه`)** is kept: users often type `ه` for `ة`.
2. **Hamza on carriers (`ؤ`, `ئ`)** is not folded.
3. **Diacritics** are folded only in the matching key (F4). Their frequency in `text_ar` is reported by `validate`.
