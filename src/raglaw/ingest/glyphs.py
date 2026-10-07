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
stored, before the line is stripped, so the spaces at each seam survive. An
Arabic piece often stores a space before characters drawn to its right
(``' مصادر'`` is drawn ``'مصادر '``); each space is moved to where it is drawn.

- **Stray zero-width alefs (D11, P28):** after the swap, a zero-width alef that
  follows no ``ل`` belongs to no letter. It is invisible on the page and
  duplicates a lam-alef placeholder of the next row, so it is dropped.

Private-use glyphs (U+E000 to U+F8FF, P27) are counted, not changed: they are
ligatures of a letter plus ``ه`` that the font maps to no Unicode letter, and
errata fix them in ``repair``.
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
# Characters of left-to-right runs inside right-to-left text.
LEFT_TO_RIGHT = re.compile(r"[A-Za-z0-9]")
PRIVATE_USE = re.compile(r"[-]")


@dataclass(frozen=True)
class CellText:
    """A cell's repaired lines, and how many repairs each defect class needed."""

    lines: list[str]
    pieces_merged: int = 0
    spaces_moved: int = 0
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


def drop_stray_zero_width_alefs(
    chars: list[dict], previous: dict | None = None
) -> tuple[list[dict], int]:
    """
    Drop zero-width alefs that don't directly follow ``ل`` (D11).

    After the swap, every zero-width alef that belongs to a lam-alef sits right
    after its lam. A plain "before ``ل``" check can't tell the two apart: a
    repaired ``لال`` still has a zero-width alef before a lam.

    It runs before spaces are moved (``reorder_spaces``): moving a space can
    put a stray alef right after a lam (``العمل أ`` would become ``العملأ``).
    ``previous`` is the character before ``chars`` on the same line.

    returns:
    - chars (list[dict]): the characters without the stray alefs
    - dropped (int): how many were dropped
    """
    context = [previous, *chars]
    kept = [
        c
        for i, c in enumerate(chars)
        if not (
            _is_zero_width_alef(c) and (context[i] is None or context[i]["c"] != "ل")
        )
    ]
    return kept, len(chars) - len(kept)


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


def reorder_spaces(chars: list[dict]) -> tuple[list[dict], int]:
    """
    Move each space of a right-to-left piece to where it is drawn (P30).

    In right-to-left text the next character is drawn to the *left*. A space
    stored before characters drawn to its right (e.g., ``' مصادر'``, drawn
    ``'مصادر '``) is moved past them. Only spaces move: letters keep their
    stored order, which the lam-alef and digit repairs rely on. A space never
    moves past a Latin letter or an ASCII digit: those runs read left to right,
    so their next character is *meant* to be on the right.

    returns:
    - chars (list[dict]): the piece's characters, spaces where they're drawn
    - moved (int): how many spaces moved
    """
    chars, moved = list(chars), 0
    i = len(chars) - 1
    while i >= 0:
        if chars[i]["c"].isspace():
            j = i
            while (
                j + 1 < len(chars)
                and not LEFT_TO_RIGHT.match(chars[j + 1]["c"])
                and chars[j + 1]["bbox"][0] > chars[j]["bbox"][0] + 0.5
            ):
                chars[j], chars[j + 1] = chars[j + 1], chars[j]
                j += 1
            moved += j != i
        i -= 1
    return chars, moved


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
    swaps = reordered = remaining = moved = 0
    private: list[str] = []
    groups, merged = visual_lines(raw, rtl)
    for pieces in groups:
        line: list[dict] = []
        for piece in pieces:
            piece_chars: list[dict] = []
            for span in piece["spans"]:
                chars, n_swaps = swap_lam_alef(span["chars"])
                chars, n_runs = order_digit_runs(chars)
                swaps += n_swaps
                reordered += n_runs
                piece_chars += chars
            piece_chars, n_dropped = drop_stray_zero_width_alefs(
                piece_chars, line[-1] if line else None
            )
            remaining += n_dropped
            if rtl:
                piece_chars, n_moved = reorder_spaces(piece_chars)
                moved += n_moved
            line += piece_chars
        text = "".join(c["c"] for c in line)
        private += PRIVATE_USE.findall(text)
        if text.strip():
            lines.append(text.strip())
    return CellText(lines, merged, moved, swaps, reordered, remaining, private)


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
