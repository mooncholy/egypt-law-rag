import pytest

from raglaw.text.normalize import fold_arabic, normalize_arabic

pytestmark = pytest.mark.unit

# One sample per rule of docs/normalization.md, each with what it must become.
# Invisible and look-alike characters are built with chr(), so no editor can
# turn them into literal characters.
RLM, RLE, LRI, NBSP, THIN = (chr(c) for c in (0x200F, 0x202B, 0x2066, 0x00A0, 0x2009))
LAM_ALEF_FORM, ALEF_FINAL_FORM, HAMZA_ABOVE, TATWEEL = (
    chr(c) for c in (0xFEFB, 0xFE8E, 0x0654, 0x0640)
)
NORMALIZE_CASES = {
    "N1 presentation forms": (LAM_ALEF_FORM + ALEF_FINAL_FORM, "لاا"),
    "N2 composed hamza": ("ا" + HAMZA_ABOVE, "أ"),
    "N3 tatweel": ("م" + TATWEEL + "ادة", "مادة"),
    "N4 control characters": (RLM + "مادة" + RLE + " ١" + LRI, "مادة ١"),
    "N5 whitespace": (
        "  مادة" + NBSP + THIN + "١ \r\n\n  نص\t\tم  ",
        "مادة ١\nنص م",
    ),
}
# Alef variants, ى, ة, diacritics, digits, P26 spaces and P14 markers stay as printed.
KEPT_AS_PRINTED = "أإآا ى ة عَقْد ١٤٧ قان ون (١ ("


@pytest.mark.parametrize(
    ("text", "expected"), NORMALIZE_CASES.values(), ids=NORMALIZE_CASES
)
def test_each_normalization_rule(text, expected):
    assert normalize_arabic(text) == expected


def test_normalization_changes_no_letter():
    assert normalize_arabic(KEPT_AS_PRINTED) == KEPT_AS_PRINTED


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("الأشخاص إلى آخر", "الاشخاص الي اخر"),  # F1, F2
        ("مادة", "ماده"),  # F3
        ("عَقْدٌ", "عقد"),  # F4
        ("المادة ١٤٧ و۱۲", "الماده 147 و12"),  # F5, both digit blocks
    ],
)
def test_each_folding_rule(text, expected):
    assert fold_arabic(text) == expected


@pytest.mark.parametrize(
    "text",
    [case[0] for case in NORMALIZE_CASES.values()] + [KEPT_AS_PRINTED, "عَقْدٌ ١٤٧ إلى"],
)
def test_both_functions_are_idempotent(text):
    """U15."""
    assert normalize_arabic(normalize_arabic(text)) == normalize_arabic(text)
    assert fold_arabic(fold_arabic(text)) == fold_arabic(text)
