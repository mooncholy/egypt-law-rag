import pytest
from pydantic import ValidationError

from raglaw.retrieval.repealed import (
    RepealedMatcher,
    RepealedSet,
    passages,
    repeal_answer,
)

pytestmark = pytest.mark.unit


def test_each_passage_carries_its_section_headings(repealed_set):
    pieces = passages(repealed_set.sections, size=10)

    texts = [t for _, t in pieces]
    assert texts[0] == "Associations\nالجمعيات\nالجمعية جم"
    assert all(t.startswith("Foundations\nالمؤسسات\n") for t in texts[3:])
    assert "".join(t.split("\n", 2)[2] for t in texts[:3]) == (
        repealed_set.sections[0].text
    )


def test_the_matcher_returns_the_best_passage_and_its_range(
    repealed_set, stub_reranker
):
    foundations = passages(repealed_set.sections)[1][1]
    matcher = RepealedMatcher(repealed_set, stub_reranker({foundations: 0.4}))

    match = matcher.match("What does the Code say about foundations?")

    assert (match.record.first, match.score, match.section) == (5, 0.4, "Foundations")
    assert match.to_metadata()["record"]["last"] == 6


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (
            {
                "sections": [
                    {"first": 9, "heading_en": "x", "heading_ar": "x", "text": ""}
                ]
            },
            "no repeal record",
        ),
        ({"ranges": []}, "at least 1"),
    ],
)
def test_every_section_belongs_to_a_range_and_every_range_has_one(
    repealed_set, change, message
):
    with pytest.raises(ValidationError, match=message):
        RepealedSet.model_validate(repealed_set.model_dump() | change)


def test_the_reply_states_the_range_the_date_and_the_law(repealed_set):
    record = repealed_set.ranges[0]

    assert repeal_answer(record, "en") == (
        "The Civil Code's provisions on associations and foundations were in "
        "Articles 5-6. They were repealed as of 3 November 1956 by Decree-Law "
        "No. 348 of 1956. The current Code keeps only a note that these articles "
        "are repealed."
    )
    arabic = repeal_answer(record, "ar")
    assert "المواد من ٥ إلى ٦" in arabic
    assert "اعتبارًا من ٣ نوفمبر ١٩٥٦ بالقانون رقم ٣٤٨ لسنة ١٩٥٦" in arabic
