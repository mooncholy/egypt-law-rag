import pytest

from raglaw.ingest.glyphs import (
    cell_text,
    order_digit_runs,
    swap_lam_alef,
    vertical_overlap,
)

pytestmark = pytest.mark.unit


def char(c: str, x0: float, width: float = 5.0) -> dict:
    return {"c": c, "bbox": (x0, 0.0, x0 + width, 10.0)}


def word(text: str, zero_width: set[int] = frozenset(), x0: float = 0.0) -> list[dict]:
    """Characters in stream order; positions in ``zero_width`` get no width."""
    return [
        char(c, x0 + 5 * i, 0.0 if i in zero_width else 5.0) for i, c in enumerate(text)
    ]


def piece(chars: list[dict], x0: float, x1: float, top: float) -> dict:
    """A ``rawdict`` line ("piece") spanning ``x0``-``x1`` and 12 pt from ``top``."""
    return {"bbox": (x0, top, x1, top + 12.0), "spans": [{"chars": chars}]}


def raw(*lines: list[dict]) -> dict:
    """A ``rawdict`` with one block, one line per argument, each on its own row."""
    pieces = [piece(ln, 0.0, 100.0, 20.0 * i) for i, ln in enumerate(lines)]
    return {"blocks": [{"lines": pieces}]}


def text(chars: list[dict]) -> str:
    return "".join(c["c"] for c in chars)


# --- Lam-alef (U1) ------------------------------------------------------------


def test_zero_width_alef_before_lam_swaps_with_it():
    # ا, zero-width أ, ل, ش, خ, ا, ص (the brief's U1 sequence)
    chars, swaps = swap_lam_alef(word("األشخاص", zero_width={1}))

    assert text(chars) == "الأشخاص"
    assert swaps == 1


def test_correct_text_is_unchanged():
    chars, swaps = swap_lam_alef(word("الأشخاص"))

    assert text(chars) == "الأشخاص"
    assert swaps == 0


def test_an_alef_with_width_before_lam_is_not_a_signature():
    chars, swaps = swap_lam_alef(word("الل"))  # ا ل ل, all with width

    assert text(chars) == "الل"
    assert swaps == 0


def test_repaired_lal_is_not_counted_as_a_stray_alef():
    # الالتزامات stored as ا, zero-width ا, ل, ل, ...: after the swap a
    # zero-width alef still precedes a lam, but it follows its own lam.
    cell = cell_text(raw(word("االلتزامات", zero_width={1})))

    assert cell.lines == ["الالتزامات"]
    assert (cell.lam_alef_swaps, cell.stray_zero_width_alefs) == (1, 0)


def test_trailing_zero_width_alef_is_counted_and_kept():
    cell = cell_text(raw(word("حق الملكية أ", zero_width={11})))

    assert cell.lines == ["حق الملكية أ"]
    assert cell.stray_zero_width_alefs == 1


# --- Digit order (U2) ---------------------------------------------------------


def test_digit_run_is_ordered_by_x0():
    # ١ at x=530 and ٢ at x=524, stored in that order: the number is ٢١.
    chars, reordered = order_digit_runs([char("١", 530), char("٢", 524)])

    assert text(chars) == "٢١"
    assert reordered == 1


def test_ordered_runs_and_single_digits_are_unchanged():
    stream = [char("١", 500), char("٢", 505), char(" ", 510), char("٣", 520)]

    chars, reordered = order_digit_runs(stream)

    assert text(chars) == "١٢ ٣"
    assert reordered == 0


def test_runs_are_ordered_independently():
    stream = [char("٢", 510), char("١", 505), char(" ", 500), char("٤", 490)]
    stream += [char("٣", 485)]

    chars, reordered = order_digit_runs(stream)

    assert text(chars) == "١٢ ٣٤"
    assert reordered == 2


# --- Cells ------------------------------------------------------------------


def test_cell_lines_are_stripped_and_empty_lines_dropped():
    cell = cell_text(raw(word(" مادة "), word("   "), word("١٤٧")))

    assert cell.lines == ["مادة", "١٤٧"]
    assert cell.text == "مادة\n١٤٧"


def test_private_use_glyphs_are_counted_and_kept():
    cell = cell_text(raw(word("يعلم"), word("ذا")))

    assert cell.lines == ["يعلم", "ذا"]
    assert cell.private_use_glyphs == [""]


# --- Visual lines (P30) -------------------------------------------------------


def test_pieces_on_one_visual_line_are_joined_as_stored():
    """`الفصل الثان` and `ي` share a baseline: PyMuPDF splits them, the merge rejoins."""
    cell = cell_text(
        {
            "blocks": [
                {
                    "lines": [
                        piece(word("الفصل الثان"), 520.9, 559.5, 293.1),
                        piece(word("ي "), 508.4, 521.0, 293.1),
                        piece(word("تعدد محل"), 492.9, 559.5, 305.7),
                    ]
                }
            ]
        }
    )

    assert cell.lines == ["الفصل الثاني", "تعدد محل"]
    assert cell.pieces_merged == 1


def test_spaces_at_a_seam_survive_the_merge():
    cell = cell_text(
        {
            "blocks": [
                {
                    "lines": [
                        piece(word("قواعد العدالة ."), 400.0, 560.0, 0.0),
                        piece(word(" "), 395.0, 400.0, 0.0),
                        piece(word("فإذا"), 350.0, 395.0, 0.0),
                    ]
                }
            ]
        }
    )

    assert cell.lines == ["قواعد العدالة . فإذا"]


def test_a_lowered_ligature_piece_still_joins_its_line():
    """A ligature glyph's origin sits 2.9 pt lower (P27); its box still overlaps."""
    cell = cell_text(
        {
            "blocks": [
                {
                    "lines": [
                        piece(word("الدائن المر"), 400.0, 560.0, 0.0),
                        piece(word("\ue814 ن"), 380.0, 405.0, 2.9),
                    ]
                }
            ]
        }
    )

    assert cell.lines == ["الدائن المر\ue814 ن"]


def test_english_pieces_are_put_in_left_to_right_order():
    """Pages 8, 64, 147, 148 and 154 store some English pieces right to left."""
    cell = cell_text(
        {
            "blocks": [
                {
                    "lines": [
                        piece(word("to be consumable."), 200.0, 300.0, 0.0),
                        piece(word("are deemed "), 140.0, 200.0, 0.0),
                    ]
                }
            ]
        },
        rtl=False,
    )

    assert cell.lines == ["are deemed to be consumable."]


@pytest.mark.parametrize(
    ("a", "b", "share"),
    [
        ((0, 0, 1, 12), (0, 0, 1, 12), 1.0),
        ((0, 0, 1, 12), (0, 2.9, 1, 14.9), (12 - 2.9) / 12),
        ((0, 0, 1, 12), (0, 12.3, 1, 24.3), 0.0),
    ],
)
def test_vertical_overlap_is_a_share_of_the_shorter_box(a, b, share):
    assert vertical_overlap(a, b) == pytest.approx(share)
