import pytest

from raglaw.retrieval.lookup import article_references, cited_articles

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


# --- Citations inside article text (search.cite_expansion) -------------------------


@pytest.mark.parametrize(
    ("text", "numbers"),
    [
        ("as provided in Article 563.", [563]),
        ("under Article\n221, be made by a third party", [221]),  # broken across lines
        ("Articles 221 and 22 apply", [221, 22]),  # as printed (Art. 170)
        ("Articles 1033, 1040 and 1041 apply", [1033, 1040, 1041]),
        ("Articles 466 to 468", [466, 468]),  # a range: its ends, never its middle
        ("وفقا للمادة ٥٦٣", [563]),
        ("مع مراعاة المادتين ٢٢١ ، ٢٢٢", [221, 222]),
        ("أحكام المادتين ٢٢١ و ٢٢٢ مراعيا", [221, 222]),
        ("المادتين ٢٢١-٢٢٢ من", [221, 222]),
        ("المادة ١١٤ فقرة ٢", [114]),  # the second number is a paragraph
        (
            "أحكام المادة ١٠٣٣ وأحكام المواد من ١٠٤٠\nإلى ١٠٤٢ المتعلقة",
            [1033, 1040, 1042],
        ),
        ("يقدم المقاول مواد من عنده", []),  # materials, not articles
    ],
)
def test_citations_are_found_in_every_printed_form(text, numbers):
    assert cited_articles(text) == numbers


def test_no_citation_means_nothing_is_cited():
    assert cited_articles("The debtor pays 7 percent within 15 days.") == []
