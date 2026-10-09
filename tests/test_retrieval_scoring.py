import json

import pytest
from langchain_core.documents import Document
from pydantic import ValidationError

from raglaw.retrieval.scoring import (
    EvalSetError,
    Retrieved,
    bilingual_agreement,
    read_questions,
    score_question,
    split_questions,
    summarize,
)

pytestmark = pytest.mark.unit

KS = (1, 3, 5, 10)


def retrieved(articles, dense=0.5, bm25=None, rrf=None):
    return Retrieved(
        articles=articles,
        top_scores={"dense_score": dense, "bm25_score": bm25, "rrf_score": rrf},
    )


def hit(article, rank, dense=None, bm25=None, rrf=None):
    """A retriever ``Document`` for one chunk of ``article``."""
    return Document(
        page_content="",
        metadata={
            "article_number": article,
            "retrieval": {
                "rank": rank,
                "dense_score": dense,
                "bm25_score": bm25,
                "rrf_score": rrf,
                "lookup": False,
            },
        },
    )


# --- U28: scoring one question ---------------------------------------------------


def test_match_all_needs_every_expected_article_within_k(make_question):
    score = score_question(make_question("q001", [1, 2]), retrieved([1, 3, 2]), KS)

    assert score.hits == {1: False, 3: True, 5: True, 10: True}
    assert score.ranks == {1: 1, 2: 3}
    assert score.reciprocal_rank == 1.0  # the first expected article found


def test_match_any_needs_one_expected_article_within_k(make_question):
    question = make_question("q001", [54, 55, 56], match="any", kind="repealed")

    score = score_question(question, retrieved([9, 55]), KS)

    assert score.hits == {1: False, 3: True, 5: True, 10: True}
    assert score.reciprocal_rank == 0.5


def test_a_question_whose_articles_never_come_back_scores_zero(make_question):
    score = score_question(make_question("q001", [7]), retrieved([1, 2, 3]), KS)

    assert score.hits == dict.fromkeys(KS, False)
    assert score.ranks == {7: None}
    assert score.reciprocal_rank == 0.0


def test_parts_of_one_article_collapse_to_the_article():
    """Article 1143 has two parts; its second part must not push others down."""
    documents = [
        hit(1143, 1, dense=0.9),
        hit(1143, 2),
        hit(5, 3),
        hit(1143, 4),
        hit(8, 5),
    ]

    found = Retrieved.from_documents(documents)

    assert found.articles == [1143, 5, 8]
    assert found.top_scores == {
        "dense_score": 0.9,
        "bm25_score": None,
        "rrf_score": None,
    }


def test_a_range_chunk_stands_for_every_article_it_covers(make_question):
    """One chunk for repealed 54-80 (chunking.repealed=per_range) at rank 2."""
    documents = [hit(7, 1), hit(54, 2), hit(9, 3)]
    documents[1].metadata["range_end"] = 80

    found = Retrieved.from_documents(documents)
    score = score_question(
        make_question("q001", [60, 61], match="any", kind="repealed"), found, KS
    )

    assert found.articles == [7, 54, 9]
    assert found.ranges == {54: 80}
    assert score.ranks == {60: 2, 61: 2}
    assert score.hits == {1: False, 3: True, 5: True, 10: True}


def test_nothing_retrieved_has_no_top_scores():
    assert Retrieved.from_documents([]) == Retrieved(articles=[], top_scores={})


# --- U28: the summary ------------------------------------------------------------


def test_out_of_scope_questions_are_left_out_of_recall(make_question):
    scores = [
        score_question(make_question("q001", [1]), retrieved([1], dense=0.8), KS),
        score_question(make_question("q002", [2]), retrieved([1, 2], dense=0.7), KS),
        score_question(
            make_question("q003", [], kind="out_of_scope"),
            retrieved([4], dense=0.4),
            KS,
        ),
    ]

    summary = summarize(scores, KS)

    assert summary["overall"]["questions"] == 2
    assert summary["overall"]["recall_at_1"] == 0.5
    assert summary["overall"]["recall_at_3"] == 1.0
    assert summary["overall"]["mrr"] == 0.75
    assert "out_of_scope" not in summary["by_kind"]
    assert summary["out_of_scope_questions"] == 1
    # The evidence for a later "no answer" threshold: rank-1 scores, both sides
    top1 = summary["top1_scores"]
    assert top1["out_of_scope"]["dense_score"] == {
        "min": 0.4,
        "median": 0.4,
        "max": 0.4,
    }
    assert top1["in_scope"]["dense_score"]["median"] == 0.75
    assert top1["out_of_scope"]["bm25_score"] is None  # the mode gave none


def test_the_summary_breaks_down_by_kind_language_and_register(make_question):
    scores = [
        score_question(make_question("q001", [1]), retrieved([1]), KS),
        score_question(
            make_question(
                "q002", [2], language="ar", register="colloquial", kind="lay_term"
            ),
            retrieved([9]),
            KS,
        ),
    ]

    summary = summarize(scores, KS)

    assert summary["by_kind"]["rule"]["recall_at_1"] == 1.0
    assert summary["by_kind"]["lay_term"]["recall_at_1"] == 0.0
    assert summary["by_language"] == {
        "en": {
            "questions": 1,
            "recall_at_1": 1.0,
            "recall_at_3": 1.0,
            "recall_at_5": 1.0,
            "recall_at_10": 1.0,
            "mrr": 1.0,
        },
        "ar": {
            "questions": 1,
            "recall_at_1": 0.0,
            "recall_at_3": 0.0,
            "recall_at_5": 0.0,
            "recall_at_10": 0.0,
            "mrr": 0.0,
        },
    }
    assert summary["by_register"]["colloquial"]["questions"] == 1


def test_a_summary_without_out_of_scope_questions_reports_none(make_question):
    summary = summarize(
        [score_question(make_question("q001", [1]), retrieved([1]), KS)], KS
    )

    assert summary["out_of_scope_questions"] == 0
    assert summary["top1_scores"]["out_of_scope"]["dense_score"] is None


# --- Bilingual agreement -----------------------------------------------------------


def test_bilingual_agreement_is_the_share_of_pairs_with_the_same_top_article(
    make_question,
):
    pairs = [
        (make_question("q001", [1], pair_id="p1"), [1, 2]),
        (make_question("q002", [1], pair_id="p1", language="ar", register="msa"), [1]),
        (make_question("q003", [5], pair_id="p2", kind="lay_term"), [5]),
        (
            make_question(
                "q004",
                [5],
                pair_id="p2",
                kind="lay_term",
                language="ar",
                register="msa",
            ),
            [6, 5],
        ),
        (make_question("q005", [9]), [9]),  # not in a pair
    ]
    scores = [score_question(q, retrieved(found), KS) for q, found in pairs]

    agreement = bilingual_agreement(scores)

    assert agreement["pairs"] == 2
    assert agreement["same_top1"] == 0.5
    assert agreement["by_kind"] == {
        "rule": {"pairs": 1, "same_top1": 1.0},
        "lay_term": {"pairs": 1, "same_top1": 0.0},
    }


def test_a_pair_with_one_member_missing_is_not_counted(make_question):
    """A split keeps pairs together, but a filtered run may hold only one side."""
    score = score_question(make_question("q001", [1], pair_id="p1"), retrieved([1]), KS)

    assert bilingual_agreement([score]) == {
        "pairs": 0,
        "same_top1": None,
        "by_kind": {},
    }


# --- The tuning / held-out split (D15) ---------------------------------------------


def test_questions_alternate_when_every_one_is_alike(make_question):
    questions = [make_question(f"q{i:03d}", [i]) for i in range(1, 6)]

    split = split_questions(questions)

    assert [q.id for q in split["tuning"]] == ["q001", "q003", "q005"]
    assert [q.id for q in split["heldout"]] == ["q002", "q004"]


def test_each_half_holds_about_half_of_every_kind_and_register(make_question):
    kinds = ["rule", "lay_term", "repealed"]
    registers = [("en", "english"), ("ar", "msa"), ("ar", "colloquial")]
    questions = [
        make_question(
            f"q{i:03d}",
            [i],
            kind=kinds[i % 3],
            language=registers[(i // 3) % 3][0],
            register=registers[(i // 3) % 3][1],
        )
        for i in range(1, 37)
    ]

    split = split_questions(questions)

    for attr in ("kind", "speech_register"):
        for value in {getattr(q, attr) for q in questions}:
            sizes = [
                sum(getattr(q, attr) == value for q in half) for half in split.values()
            ]
            assert abs(sizes[0] - sizes[1]) <= 1, (attr, value)


def test_a_pair_stays_in_one_half_and_counts_as_two(make_question):
    """q001 and its twin q009 move together, at q001's place in the order; with
    two questions already in tuning, both q002 and q003 go to held-out."""
    questions = [
        make_question("q001", [1], pair_id="p1"),
        make_question("q002", [2]),
        make_question("q003", [3]),
        make_question("q009", [1], pair_id="p1", language="ar", register="msa"),
    ]

    split = split_questions(questions)

    assert [q.id for q in split["tuning"]] == ["q001", "q009"]
    assert [q.id for q in split["heldout"]] == ["q002", "q003"]


def test_the_split_is_the_same_whatever_the_input_order(make_question):
    questions = [make_question(f"q{i:03d}", [i]) for i in range(1, 8)]

    assert split_questions(questions) == split_questions(questions[::-1])


def test_appending_questions_never_moves_an_existing_one(make_question):
    kinds = ["rule", "lay_term"]
    questions = [
        make_question(f"q{i:03d}", [i], kind=kinds[i % 2]) for i in range(1, 21)
    ]

    before = split_questions(questions[:12])
    after = split_questions(questions)

    for half in ("tuning", "heldout"):
        assert {q.id for q in before[half]} <= {q.id for q in after[half]}


# --- Reading the eval set ----------------------------------------------------------


def write_jsonl(path, questions):
    path.write_text(
        "".join(
            json.dumps(q.model_dump(exclude_none=True, by_alias=True)) + "\n"
            for q in questions
        ),
        encoding="utf-8",
    )
    return path


def test_the_eval_set_reads_back_in_file_order(tmp_path, make_question):
    questions = [make_question("q002", [2]), make_question("q001", [1])]

    assert read_questions(write_jsonl(tmp_path / "e.jsonl", questions)) == questions


def test_an_unknown_field_is_rejected(tmp_path):
    path = tmp_path / "e.jsonl"
    path.write_text('{"id": "q001", "surprise": 1}\n', encoding="utf-8")

    with pytest.raises(ValidationError):
        read_questions(path)


def test_out_of_scope_means_no_expected_articles(make_question):
    with pytest.raises(ValidationError, match="out_of_scope"):
        make_question("q001", [3], kind="out_of_scope")
    with pytest.raises(ValidationError, match="out_of_scope"):
        make_question("q002", [])


@pytest.mark.parametrize(
    ("questions", "problem"),
    [
        ([("q001", {}), ("q001", {})], "duplicate"),
        ([("q001", {"pair_id": "p1"})], "pair p1"),
        (
            [("q001", {"pair_id": "p1"}), ("q002", {"pair_id": "p1"})],
            "pair p1",  # both English
        ),
        (
            [
                ("q001", {"pair_id": "p1"}),
                (
                    "q002",
                    {
                        "pair_id": "p1",
                        "language": "ar",
                        "register": "msa",
                        "kind": "lay_term",
                    },
                ),
            ],
            "pair p1",  # kinds differ
        ),
        (
            [
                ("q001", {}),
                (
                    "q002",
                    {"language": "ar", "register": "msa", "translated_from": "q001"},
                ),
            ],
            "translated_from",  # not in a pair with its source
        ),
    ],
)
def test_a_broken_eval_set_is_refused(tmp_path, make_question, questions, problem):
    built = [make_question(qid, [1], **fields) for qid, fields in questions]

    with pytest.raises(EvalSetError, match=problem):
        read_questions(write_jsonl(tmp_path / "e.jsonl", built))


# --- Corpus: the real eval set -------------------------------------------------------


@pytest.mark.corpus
def test_the_real_eval_set_splits_into_two_balanced_halves(repo_root):
    questions = read_questions(repo_root / "data/gold/retrieval_eval.jsonl")

    split = split_questions(questions)

    assert len(questions) == 142
    assert sorted(q.id for half in split.values() for q in half) == sorted(
        q.id for q in questions
    )
    for attr in ("kind", "language"):
        for value in {getattr(q, attr) for q in questions}:
            sizes = [
                sum(getattr(q, attr) == value for q in half) for half in split.values()
            ]
            assert abs(sizes[0] - sizes[1]) <= 2, (attr, value)  # a pair moves two


# --- One record per question (the comparison's input) --------------------------------


def test_a_question_score_becomes_one_flat_record(make_question):
    question = make_question(
        "q001", [54, 55], match="any", kind="repealed", pair_id="p1"
    )
    documents = [hit(7, 1, dense=0.6), hit(54, 2)]
    documents[1].metadata["range_end"] = 80

    record = score_question(
        question, Retrieved.from_documents(documents), KS
    ).to_record()

    assert record == {
        "id": "q001",
        "kind": "repealed",
        "language": "en",
        "register": "english",
        "pair_id": "p1",
        "in_scope": True,
        "match": "any",
        "expected": [54, 55],
        "first_rank": 2,
        "hits": {"1": False, "3": True, "5": True, "10": True},
        "returned": ["7", "54-80"],
        "top_scores": {"dense_score": 0.6, "bm25_score": None, "rrf_score": None},
    }


def test_an_out_of_scope_record_has_no_rank_or_hits(make_question):
    record = score_question(
        make_question("q001", [], kind="out_of_scope"), retrieved([3]), KS
    ).to_record()

    assert (record["in_scope"], record["first_rank"], record["hits"]) == (
        False,
        None,
        {},
    )
