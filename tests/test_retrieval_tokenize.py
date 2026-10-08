import pytest

from raglaw.retrieval.tokenize import fold_tokens

pytestmark = pytest.mark.unit

# U25: one sample per fold, each with the tokens it must become.
FOLD_CASES = {
    "alef variants": ("أحكام إلى آثار ٱلقانون", ["احكام", "الي", "اثار", "القانون"]),
    "alef maqsura as ya": ("على", ["علي"]),
    "ta marbuta as ha": ("المادة", ["الماده"]),
    "diacritics": ("عَقْدٌ مَدَنِيّ", ["عقد", "مدني"]),
    "Arabic-Indic digits": ("المادة ١٤٧", ["الماده", "147"]),
    "tatweel": ("مـــادة", ["ماده"]),
}


@pytest.mark.parametrize(("text", "expected"), FOLD_CASES.values(), ids=FOLD_CASES)
def test_spelling_variants_fold_to_one_token(text, expected):
    assert fold_tokens(text) == expected


def test_arabic_and_english_split_alike_on_non_word_characters():
    assert fold_tokens("Article 147, Civil-Code؟ العقد، شريعة (المتعاقدين).") == [
        "article",
        "147",
        "civil",
        "code",
        "العقد",
        "شريعه",
        "المتعاقدين",
    ]


def test_underscores_and_punctuation_are_never_tokens():
    assert fold_tokens("a_b ... ؛ — ") == ["a", "b"]


def test_folding_a_question_and_its_article_text_gives_the_same_key():
    """A question typed without hamza or diacritics matches the printed text."""
    assert fold_tokens("الأهلية") == fold_tokens("الاهليه")


# --- Light prefix stripping (a measured variant, off by default) -------------


def test_prefixes_are_kept_by_default():
    assert fold_tokens("والعقد بالتزام") == ["والعقد", "بالتزام"]


@pytest.mark.parametrize(
    ("word", "stem"),
    [
        ("والعقد", "عقد"),  # و + ال
        ("فالعقد", "عقد"),  # ف + ال
        ("بالعقد", "عقد"),  # بال
        ("العقد", "عقد"),  # ال
        ("وعقد", "عقد"),  # و alone
    ],
)
def test_light_stripping_removes_conjunctions_and_the_article(word, stem):
    assert fold_tokens(word, strip_prefixes=True) == [stem]


@pytest.mark.parametrize("word", ["وقف", "بيع", "فرد", "الي"])
def test_light_stripping_keeps_short_words_and_a_bare_ba_whole(word):
    """A word of three letters keeps its first letter, and ب alone is never stripped."""
    assert fold_tokens(word, strip_prefixes=True) == [word]


def test_light_stripping_leaves_english_alone():
    assert fold_tokens("Alliance ball", strip_prefixes=True) == ["alliance", "ball"]
