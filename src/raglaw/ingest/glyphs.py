"""Turn a cell's ``rawdict`` characters into text lines, repairing glyph defects.

The two glyph-level defects need character positions, which exist only here,
before characters are joined into text (R4):

- **Lam-alef (R5, P23):** the font draws لا as a zero-width alef variant
  stored *before* ``ل``. Swapping each such pair restores the spelling
  (``السجالت`` becomes ``السجلات``).
- **Digit order (R6, P24):** every multi-digit Arabic-Indic run is stored right
  to left. Sorting a run's digits by ascending ``x0`` restores the number.

PyMuPDF also splits one visual line into several "lines" whenever a text run
starts to the left of the previous one, which right-to-left text does
constantly (P30): ``الفصل الثان`` and ``ي`` on the same baseline. Pieces whose
boxes overlap vertically are joined back into one line in reading order, as
stored, before the line is stripped, so the spaces at each seam survive.

Two artifacts are counted, not changed, because no approved rule covers them
yet (R10):

- **Private-use glyphs** (U+E000 to U+F8FF): ligatures of a letter plus ``ه``
  (e.g., U+E812 for ``به``) that the font maps to no Unicode letter.
- **Stray zero-width alefs:** invisible alef glyphs with no lam, at the end of
  a line (mostly headings).
"""

import re
from dataclasses import dataclass, field

import pymupdf

from raglaw.ingest.measure import ALEFS, AR_DIGIT_CHARS

ZERO_WIDTH = 0.01  # a glyph narrower than this (in points) is drawn with no advance
# Two pieces belong to one visual line when their boxes overlap vertically by at
# least this share of the shorter box. A ligature glyph's origin can sit 2.9 pt
# lower than its line (P27), so equal baselines are too strict a test.
SAME_LINE_OVERLAP = 0.5
PRIVATE_USE = re.compile(r"[-]")


@dataclass(frozen=True)
class CellText:
    """A cell's repaired lines, and how many repairs each defect class needed."""

    lines: list[str]
    pieces_merged: int = 0
    lam_alef_swaps: int = 0
    digit_runs_reordered: int = 0
    stray_zero_width_alefs: int = 0
    private_use_glyphs: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n".join(self.lines)


def _is_zero_width_alef(char: dict) -> bool:
    return char["c"] in ALEFS and char["bbox"][2] - char["bbox"][0] < ZERO_WIDTH


def _is_broken_lam_alef(char: dict, following: dict) -> bool:
    return _is_zero_width_alef(char) and following["c"] == "ل"


def stray_zero_width_alefs(chars: list[dict]) -> int:
    """
    Count zero-width alefs that don't directly follow ``ل``.

    After the swap, every zero-width alef should sit right after its lam. A
    plain "before ``ل``" check can't tell: a repaired ``لال`` still has one.

    returns:
    - count (int): zero-width alefs not preceded by ``ل``
    """
    return sum(
        _is_zero_width_alef(c) and (i == 0 or chars[i - 1]["c"] != "ل")
        for i, c in enumerate(chars)
    )


def swap_lam_alef(chars: list[dict]) -> tuple[list[dict], int]:
    """
    Swap each zero-width alef variant with the ``ل`` that follows it (R5).

    returns:
    - chars (list[dict]): the span's characters with every pair swapped
    - swaps (int): how many pairs were swapped
    """
    chars, swaps, i = list(chars), 0, 0
    while i < len(chars) - 1:
        if _is_broken_lam_alef(chars[i], chars[i + 1]):
            chars[i], chars[i + 1] = chars[i + 1], chars[i]
            swaps += 1
            i += 2
        else:
            i += 1
    return chars, swaps


def order_digit_runs(chars: list[dict]) -> tuple[list[dict], int]:
    """
    Order each run of two or more Arabic-Indic digits by ascending ``x0`` (R6).

    returns:
    - chars (list[dict]): the span's characters with every run in reading order
    - reordered (int): how many runs changed order
    """
    out: list[dict] = []
    run: list[dict] = []
    reordered = 0
    for char in [*chars, None]:
        if char is not None and char["c"] in AR_DIGIT_CHARS:
            run.append(char)
            continue
        if len(run) > 1:
            ordered = sorted(run, key=lambda c: c["bbox"][0])
            reordered += ordered != run
            run = ordered
        out += run
        run = []
        if char is not None:
            out.append(char)
    return out, reordered


def vertical_overlap(a: tuple, b: tuple) -> float:
    """
    Measure how much two boxes overlap vertically.

    returns:
    - share (float): the overlap as a share of the shorter box's height, 0 to 1
    """
    top, bottom = max(a[1], b[1]), min(a[3], b[3])
    shorter = min(a[3] - a[1], b[3] - b[1])
    return max(0.0, bottom - top) / shorter if shorter > 0 else 0.0


def visual_lines(raw: dict, rtl: bool) -> tuple[list[list[dict]], int]:
    """
    Group a cell's ``rawdict`` lines into the visual lines they belong to.

    Consecutive pieces whose boxes overlap vertically form one visual line,
    ordered for reading: right to left when ``rtl``, else left to right.

    returns:
    - lines (list[list[dict]]): each visual line's pieces, in reading order
    - merged (int): how many pieces were joined onto a line before them
    """
    groups: list[list[dict]] = []
    for line in (ln for b in raw["blocks"] for ln in b.get("lines", [])):
        if groups and vertical_overlap(groups[-1][-1]["bbox"], line["bbox"]) >= (
            SAME_LINE_OVERLAP
        ):
            groups[-1].append(line)
        else:
            groups.append([line])
    if rtl:
        ordered = [sorted(g, key=lambda ln: -ln["bbox"][2]) for g in groups]
    else:
        ordered = [sorted(g, key=lambda ln: ln["bbox"][0]) for g in groups]
    return ordered, sum(len(g) - 1 for g in groups)


def cell_text(raw: dict, rtl: bool = True) -> CellText:
    """
    Join a cell's ``rawdict`` characters into visual lines, repairing glyphs.

    Glyph repairs run per span. Pieces of one visual line are joined exactly as
    stored, with no space added, and only then stripped; empty lines are
    dropped.

    returns:
    - cell (CellText): the repaired lines and the repair counts
    """
    lines: list[str] = []
    swaps = reordered = remaining = 0
    private: list[str] = []
    groups, merged = visual_lines(raw, rtl)
    for pieces in groups:
        text = ""
        for span in (s for piece in pieces for s in piece["spans"]):
            chars, n_swaps = swap_lam_alef(span["chars"])
            chars, n_runs = order_digit_runs(chars)
            swaps += n_swaps
            reordered += n_runs
            remaining += stray_zero_width_alefs(chars)
            text += "".join(c["c"] for c in chars)
        private += PRIVATE_USE.findall(text)
        if text.strip():
            lines.append(text.strip())
    return CellText(lines, merged, swaps, reordered, remaining, private)


def read_cell(page: pymupdf.Page, rect: pymupdf.Rect | None, rtl: bool) -> CellText:
    """
    Read one side of a row from its clipped ``rawdict``.

    ``rtl`` is True for the Arabic side, so a line's pieces read right to left.

    returns:
    - cell (CellText): the repaired text, or no lines when ``rect`` is None
    """
    if rect is None:
        return CellText([])
    return cell_text(page.get_text("rawdict", clip=rect), rtl=rtl)
