import asyncio

import pytest

from raglaw.rag import (
    ANSWER_EVENT,
    NO_ANSWER,
    check_citations,
    context_articles,
    is_unaddressed,
    load_prompt,
    question_language,
)
from raglaw.schema import LogEvent

pytestmark = pytest.mark.unit


def ask(pipeline, question="What does the Code say about majority?"):
    return asyncio.run(pipeline.answer(question))


# --- The question's language --------------------------------------------------


@pytest.mark.parametrize(
    ("question", "language"),
    [
        ("ما هي شروط الأهلية؟", "ar"),
        ("What is the age of majority?", "en"),
        ("ما حكم Article 147؟", "ar"),  # Arabic words outnumber the Latin ones
        ("What does المادة ١٤٧ say about contracts?", "en"),
        ("147?", "en"),
    ],
)
def test_the_answer_is_in_the_questions_language(question, language):
    assert question_language(question) == language


# --- The no-answer gate (D21) -----------------------------------------------------


def test_a_low_rank1_reranker_score_means_the_code_does_not_address_it(
    retrieved, make_chunk
):
    assert is_unaddressed(retrieved([make_chunk(1)], rerank_score=0.29), 0.3)
    assert not is_unaddressed(retrieved([make_chunk(1)], rerank_score=0.3), 0.3)


def test_an_article_the_question_names_is_always_answered(retrieved, make_chunk):
    documents = retrieved([make_chunk(147)], rerank_score=0.01, lookup=True)

    assert not is_unaddressed(documents, 0.3)


def test_nothing_retrieved_means_unaddressed_and_no_reranker_means_no_gate(
    retrieved, make_chunk
):
    assert is_unaddressed([], 0.3)
    assert not is_unaddressed(retrieved([make_chunk(1)], rerank_score=None), 0.3)


def test_u32_below_the_threshold_the_no_answer_reply_has_no_sources(
    make_pipeline, retrieved, make_chunk
):
    pipeline, llm = make_pipeline(
        retrieved([make_chunk(1)], rerank_score=0.05), reply="Article 1."
    )

    answer = ask(pipeline, "ما حكم الطائرات المسيرة؟")

    assert (answer.text, answer.sources) == (NO_ANSWER["ar"], [])
    assert answer.unaddressed
    assert llm.calls == []  # no LLM call is made


# --- What the LLM is given ------------------------------------------------------


def test_an_articles_parts_reach_the_llm_once_in_part_order(make_chunk, retrieved):
    documents = retrieved(
        [
            make_chunk(4, "(٢) الجزء الثاني", chunk_id="art-4-p2", part_index=2),
            make_chunk(2),
            make_chunk(4, "(١) الجزء الأول", part_index=1),
        ]
    )

    [first, second] = context_articles(documents, max_sources=5)

    assert (first.first, second.first) == (4, 2)
    assert first.text_ar == "(١) الجزء الأول\n(٢) الجزء الثاني"
    assert first.text_en == "English text of article 4."


def test_only_max_sources_articles_reach_the_llm(make_chunk, retrieved):
    documents = retrieved([make_chunk(n) for n in (5, 4, 3, 2, 1)])

    articles = context_articles(documents, max_sources=3)

    assert [a.first for a in articles] == [5, 4, 3]


def test_the_prompt_marks_repealed_and_one_language_passages(
    make_pipeline, retrieved, make_chunk
):
    documents = retrieved(
        [
            make_chunk(54, "ملغاة", is_repealed=True),
            make_chunk(7, only_in_en=["A passage printed in English only."]),
        ]
    )
    pipeline, llm = make_pipeline(documents, reply="See Article 7.")

    ask(pipeline)

    [[system, user]] = llm.calls
    assert system["content"] == load_prompt("v1")
    assert "Answer in English." in user["content"]
    assert "### Article 54 (repealed)" in user["content"]
    assert (
        "Only in the English text:\nA passage printed in English only."
        in (user["content"])
    )


def test_a_missing_prompt_version_fails_loudly():
    with pytest.raises(FileNotFoundError):
        load_prompt("v999")


# --- The answer's citations -----------------------------------------------------


def test_u30_sources_are_the_retrieved_articles_the_answer_cites(
    make_pipeline, retrieved, make_chunk
):
    documents = retrieved([make_chunk(n) for n in (44, 45, 46)])
    reply = "Under Article 45 and المادة ٤٤, majority is twenty-one."
    pipeline, _ = make_pipeline(documents, reply=reply)

    answer = ask(pipeline)

    assert answer.text == reply
    assert answer.sources == [
        "Egyptian Civil Code, Article 45",
        "Egyptian Civil Code, Article 44",
    ]  # in the order the answer cites them; 46 isn't cited


def test_u31_an_article_cited_but_not_retrieved_is_dropped_and_logged(
    make_pipeline, retrieved, make_chunk, rag_log_records
):
    pipeline, _ = make_pipeline(
        retrieved([make_chunk(44)]), reply="See Article 44 and Article 999."
    )

    answer = ask(pipeline)

    assert answer.sources == ["Egyptian Civil Code, Article 44"]
    assert answer.hallucinated == [999]
    [anomaly] = rag_log_records(LogEvent.ANOMALY)
    assert anomaly["level"] == "WARNING"
    assert (anomaly["anomaly"], anomaly["article_number"]) == (
        "hallucinated_citation",
        999,
    )


def test_an_article_retrieved_but_not_given_to_the_llm_is_not_a_source(
    make_chunk, retrieved
):
    articles = context_articles(
        retrieved([make_chunk(n) for n in (1, 2, 3)]), max_sources=2
    )

    assert check_citations("Articles 1, 2 and 3.", articles) == ([1, 2], [3])


def test_a_repealed_range_chunk_covers_every_article_in_it(make_chunk, retrieved):
    articles = context_articles(
        retrieved([make_chunk(54, "ملغاة", is_repealed=True, range_end=80)]),
        max_sources=5,
    )

    assert check_citations("Article 60 was repealed.", articles) == ([60], [])


def test_each_question_logs_one_answer_line(
    make_pipeline, retrieved, make_chunk, rag_log_records
):
    pipeline, _ = make_pipeline(retrieved([make_chunk(44)]), reply="Article 44.")

    ask(pipeline)

    [line] = rag_log_records(ANSWER_EVENT)
    assert (line["articles"], line["cited"], line["prompt_version"]) == (
        [44],
        [44],
        "v1",
    )
