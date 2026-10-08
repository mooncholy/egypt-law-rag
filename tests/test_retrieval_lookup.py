import pytest

from raglaw.retrieval.lookup import article_references

pytestmark = pytest.mark.unit


# U26: the ways the eval set names an article.
@pytest.mark.parametrize(
    ("question", "numbers"),
    [
        ("ما نص المادة ٢٢٢ من القانون المدني؟", [222]),
        ("اشرح مادة 563 مدني", [563]),
        ("What does Article 147 of the Civil Code say?", [147]),
        ("Art. 1143: which claims does it make privileged?", [1143]),
        ("المادة ١٦٣ مدني بتقول إيه؟", [163]),
        ("ما مضمون المادة (٨٠٧)؟", [807]),
        ("وفقا للمادة ٤٤ وبالمادة ٤٥", [44, 45]),
        ("article 1 and ARTICLE 2", [1, 2]),
        ("art 651", [651]),
    ],
)
def test_article_references_are_found(question, numbers):
    assert article_references(question) == numbers


@pytest.mark.parametrize(
    "question",
    [
        "Can interest exceed 7 percent?",
        "هل يجوز الاتفاق على فائدة تزيد على ٧ في المائة؟",
        "Was the Civil Code amended in 1948?",
        "صدر القانون سنة ١٩٤٨",
        "Who is liable after 10 years?",
        "The artist painted 3 walls.",
        "What do Articles 221 and 222 say?",  # plural: not a single-article lookup
    ],
)
def test_numbers_that_are_not_article_references_are_ignored(question):
    assert article_references(question) == []


def test_a_repeated_reference_is_returned_once_in_order_of_appearance():
    assert article_references("Article 5, then Article 3, then Article 5") == [5, 3]
