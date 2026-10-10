import re

import pytest
from ragas.run_config import RunConfig

from raglaw.answer_compare import answer_kind, comparison_report
from raglaw.answer_eval import (
    AnswerSetError,
    ci_subset,
    ragas_scores,
    read_answer_set,
    summarize,
)
from raglaw.config import Settings
from raglaw.retrieval.scoring import read_questions, split_questions

# A failed stub call is final; RAGAS's default retries take minutes.
ONE_TRY = RunConfig(max_retries=1, max_wait=1)

# --- U36: the scorer, with a stub judge ---------------------------------------


@pytest.mark.unit
def test_u36_faithfulness_is_1_when_supported_and_0_when_not(stub_judge):
    context = "Majority is fixed at twenty one years. It is counted in Gregorian years."
    samples = [
        {
            "user_input": "What is the age of majority?",
            "response": "Majority is fixed at twenty one years. It is counted in "
            "Gregorian years.",
            "retrieved_contexts": [context],
            "reference": "Twenty-one years.",
        },
        {
            "user_input": "What is the age of majority?",
            "response": "Majority is eighteen years.",
            "retrieved_contexts": [context],
            "reference": "Twenty-one years.",
        },
    ]

    scores = ragas_scores(
        samples, stub_judge, None, metrics=["faithfulness"], run_config=ONE_TRY
    )

    assert scores == [{"faithfulness": 1.0}, {"faithfulness": 0.0}]


@pytest.mark.unit
def test_a_metric_the_judge_cannot_score_is_none_not_a_crash(stub_judge):
    sample = {
        "user_input": "Q?",
        "response": "A.",
        "retrieved_contexts": ["A."],
        "reference": "A.",
    }

    [scores] = ragas_scores(
        [sample], stub_judge, None, metrics=["context_recall"], run_config=ONE_TRY
    )

    assert scores == {"context_recall": None}  # the stub doesn't answer it


# --- The answer set -----------------------------------------------------------------


@pytest.mark.unit
def test_each_reference_is_joined_to_its_question(answer_set, make_question):
    answers, eval_set = answer_set(
        [make_question("q002"), make_question("q001", language="ar", register="msa")]
    )

    items = read_answer_set(answers, eval_set)

    assert [i.question.id for i in items] == ["q001", "q002"]
    assert items[0].question.language == "ar"
    assert items[0].reference == "Reference for q001."


@pytest.mark.unit
@pytest.mark.parametrize(
    ("references", "message"),
    [
        ({"q009": "x"}, "not in"),
        ({"q002": "x"}, "out-of-scope"),
    ],
)
def test_a_reference_must_name_an_in_scope_question(
    answer_set, make_question, references, message
):
    questions = [
        make_question("q001"),
        make_question("q002", [], kind="out_of_scope", match="any"),
    ]
    answers, eval_set = answer_set(questions, references)

    with pytest.raises(AnswerSetError, match=message):
        read_answer_set(answers, eval_set)


@pytest.mark.unit
def test_a_question_has_one_reference(answer_set, make_question):
    answers, eval_set = answer_set([make_question("q001")])
    answers.write_text(answers.read_text() * 2)

    with pytest.raises(AnswerSetError, match="duplicate"):
        read_answer_set(answers, eval_set)


# --- U37: the CI subset ---------------------------------------------------------------


def _items(make_question, answer_set, counts):
    """{(language, kind): n} -> answer items, ids in stratum order."""
    questions, n = [], 0
    for (language, kind), count in counts.items():
        for _ in range(count):
            n += 1
            register = "english" if language == "en" else "msa"
            questions.append(
                make_question(
                    f"q{n:03d}", language=language, register=register, kind=kind
                )
            )
    return read_answer_set(*answer_set(questions))


STRATA = {
    ("en", "rule"): 6,
    ("ar", "rule"): 5,
    ("en", "multi_article"): 4,
    ("ar", "multi_article"): 3,
    ("en", "repealed"): 2,
    ("ar", "lay_term"): 4,
}


@pytest.mark.unit
def test_u37_the_ci_subset_has_its_size_and_every_stratum(make_question, answer_set):
    subset = ci_subset(_items(make_question, answer_set, STRATA), 10)

    assert len(subset) == 10
    picked = {(i.question.language, i.question.kind) for i in subset}
    assert picked == set(STRATA)


@pytest.mark.unit
def test_u37_appending_questions_leaves_the_subset_unchanged(make_question, answer_set):
    before = [
        i.question.id for i in ci_subset(_items(make_question, answer_set, STRATA), 10)
    ]
    items = _items(make_question, answer_set, STRATA)
    extra_questions = [
        make_question(f"q{900 + k:03d}", language="en", register="english", kind="rule")
        for k in range(3)
    ]
    answers, eval_set = answer_set([i.question for i in items] + extra_questions)
    after = [i.question.id for i in ci_subset(read_answer_set(answers, eval_set), 10)]

    assert after == before


@pytest.mark.unit
def test_a_set_smaller_than_the_size_is_taken_whole(make_question, answer_set):
    items = _items(make_question, answer_set, {("en", "rule"): 3})

    assert len(ci_subset(items, 20)) == 3


# --- Summaries ----------------------------------------------------------------------


@pytest.mark.unit
def test_refused_answers_are_counted_not_averaged():
    def row(qid, language, faithfulness, unaddressed=False):
        scores = (
            {}
            if unaddressed
            else {
                "faithfulness": faithfulness,
                "answer_relevancy": 0.5,
                "context_precision": None,  # unscored: left out of the mean
                "context_recall": 1.0,
            }
        )
        return {
            "id": qid,
            "language": language,
            "unaddressed": unaddressed,
            "scores": scores,
        }

    summary = summarize(
        [
            row("q1", "en", 1.0),
            row("q2", "en", 0.5),
            row("q3", "ar", 0.0),
            row("q4", "ar", None, True),
        ]
    )

    assert (summary["questions"], summary["answered"], summary["refused"]) == (
        4,
        3,
        ["q4"],
    )
    assert summary["overall"]["faithfulness"] == 0.5
    assert summary["overall"]["context_precision"] is None
    assert summary["by_language"]["en"]["faithfulness"] == 0.75
    assert summary["by_language"]["ar"]["faithfulness"] == 0.0


# --- The real answer set (DVC-tracked) -------------------------------------------------


@pytest.mark.corpus
def test_every_tuning_question_in_scope_has_a_reference_in_its_language():
    gold = Settings(_env_file=None).paths.gold_dir
    items = read_answer_set(gold / "answer_eval.jsonl", gold / "retrieval_eval.jsonl")
    tuning = split_questions(read_questions(gold / "retrieval_eval.jsonl"))["tuning"]

    assert [i.question.id for i in items] == sorted(q.id for q in tuning if q.in_scope)
    assert len(items) >= 50  # the Project 2 checklist's minimum
    for item in items:
        arabic = bool(re.search(r"[ء-ي]{3}", item.reference))
        assert arabic == (item.question.language == "ar"), item.question.id


@pytest.mark.corpus
def test_u37_the_real_ci_subset_is_20_questions_across_both_languages():
    gold = Settings(_env_file=None).paths.gold_dir
    items = read_answer_set(gold / "answer_eval.jsonl", gold / "retrieval_eval.jsonl")

    subset = ci_subset(items, 20)

    assert len(subset) == 20
    assert {i.question.language for i in subset} == {"ar", "en"}
    assert {i.question.kind for i in subset} == {i.question.kind for i in items}


# --- Comparing runs ------------------------------------------------------------------


def _row(qid, answer, language="en", unaddressed=False, faithfulness=1.0, given=(44,)):
    return {
        "id": qid,
        "language": language,
        "answer": answer,
        "unaddressed": unaddressed,
        "expected": [44],
        "articles": list(given),
        "scores": {}
        if unaddressed
        else {"faithfulness": faithfulness, "context_recall": 1.0},
    }


SUBSTANCE = "Article 44 sets the age of majority at twenty-one Gregorian years."


@pytest.mark.unit
@pytest.mark.parametrize(
    ("row", "kind"),
    [
        (_row("q1", SUBSTANCE), "answer"),
        (_row("q1", "The articles found do not address this question."), "refusal"),
        (_row("q1", "المواد المذكورة لا تتناول هذا الموضوع.", "ar"), "refusal"),
        (
            _row("q1", "Article 812 | المادة ٨١٢"),
            "refusal",
        ),  # a bare label says nothing
        (_row("q1", "", unaddressed=True), "gate"),
    ],
)
def test_each_reply_is_an_answer_a_refusal_or_a_gate_refusal(row, kind):
    assert answer_kind(row) == kind


@pytest.mark.unit
def test_the_comparison_counts_refusals_and_compares_shared_questions():
    a = [
        _row("q1", SUBSTANCE),
        _row("q2", "The articles found do not address it.", faithfulness=0.0),
    ]
    b = [_row("q1", SUBSTANCE, faithfulness=0.5), _row("q2", "", unaddressed=True)]
    runs = {
        "a": ({"overall.faithfulness": 0.5}, a),
        "b": ({"overall.faithfulness": 0.49}, b),
    }

    report = comparison_report(runs, repeat=("a", "b"))

    assert (
        "| a | 0.500 | n/a | n/a | n/a | 2 | 0 | 1 | 1 |" in report
    )  # refused with 44 given
    assert "| b | 0.490 | n/a | n/a | n/a | 1 | 1 | 0 | 0 |" in report
    assert "## On the 1 questions every run answered with substance" in report
    assert "| b | 0.500 | n/a | 0.500 |" in report
    assert "a difference of 0.0100 (C20 allows 0.02). 1 of 2 answers" in report
