import asyncio
import math

import pytest

from raglaw.retrieval import distill

pytestmark = pytest.mark.unit


def test_no_eval_article_is_a_training_article(make_article, make_question):
    """The student never trains on an article an eval question expects or cites."""
    questions = [make_question("q001", [2], supporting_articles=[3])]
    articles = [make_article(n) for n in range(1, 8)]
    articles.append(make_article(8, is_repealed=True))
    articles.append(make_article(9, ""))

    exclude = distill.held_out_articles(questions)
    picked = distill.pick_articles(articles, exclude, 4, seed=1)

    assert exclude == {2, 3}
    numbers = [a.article_number for a in picked]
    assert len(numbers) == 4 and set(numbers) <= {1, 4, 5, 6, 7}
    assert numbers == sorted(numbers)
    assert picked == distill.pick_articles(articles, exclude, 4, seed=1)
    with pytest.raises(ValueError, match="only 5 articles qualify"):
        distill.pick_articles(articles, exclude, 6, seed=1)


def test_questions_alternate_languages_and_keep_the_first_line(make_article, stub_llm):
    llm = stub_llm("1 + 1?\nExtra line.")  # a valid reply in either script

    rows = asyncio.run(distill.write_questions([make_article(1), make_article(2)], llm))

    assert [(r["article_number"], r["language"]) for r in rows] == [
        (1, "ar"),
        (2, "en"),
    ]
    assert rows[0]["question"] == "1 + 1?"
    assert "in Arabic" in llm.calls[0][1]["content"]
    assert "in English" in llm.calls[1][1]["content"]


def test_pairs_carry_the_teachers_logit_and_the_source_label(stub_reranker):
    teacher = distill.RecordingScorer(stub_reranker({"a": 0.5, "b": 0.9}))
    teacher.score("q", ["a", "b", "c"])
    query, texts, scores = teacher.calls[0]

    pairs = distill.pairs_for(query, texts, scores, [7, 8, 7], source=7)

    assert [p["label"] for p in pairs] == [1.0, 0.0, 1.0]
    assert pairs[0]["teacher_logit"] == pytest.approx(0.0)
    assert pairs[1]["teacher_logit"] == pytest.approx(math.log(9))
    assert math.isfinite(pairs[2]["teacher_logit"])  # a score of exactly 0


@pytest.mark.parametrize(
    ("reply", "language", "kept"),
    [
        ("ما هي الكفالة؟\nشرح", "ar", "ما هي الكفالة؟"),
        ("كيف يُقبل عرض في اتفاق如果没有直接接受要约？", "ar", None),
        ("كيف يمكن أن يExecuting thePromise؟", "ar", None),
        ("ما معنى GPS؟", "ar", "ما معنى GPS؟"),  # a short Latin term is fine
        ("Who owns the fruits?", "en", "Who owns the fruits?"),
        ("Who owns the ثمار?", "en", None),
    ],
)
def test_a_question_in_a_mixed_script_is_rejected(reply, language, kept):
    assert distill.clean_question(reply, language) == kept


def test_a_mixed_script_reply_is_asked_for_again(make_article, stub_llm):
    llm = stub_llm("ما الحكم如果?")

    rows = asyncio.run(distill.write_questions([make_article(1)], llm))

    assert rows == []
    assert len(llm.calls) == distill.ATTEMPTS


@pytest.mark.parametrize(
    ("ranked", "expected", "precision"),
    [
        ([1, 2, 3], {1}, 1.0),
        ([2, 1, 3], {1}, 0.5),
        ([1, 2, 3], {1, 3}, (1 + 2 / 3) / 2),
        ([2, 3, 4, 5, 6, 1], {1}, 0.0),  # beyond the top 5
    ],
)
def test_context_precision_rewards_expected_articles_ranked_early(
    ranked, expected, precision
):
    assert distill.context_precision(ranked, expected) == pytest.approx(precision)
