"""C19: the end-to-end answer check that ``scripts/check_answers.py`` writes."""

import pytest

pytestmark = pytest.mark.corpus


def test_c19_every_source_is_a_well_formed_citation_of_a_given_article(stage_metrics):
    check = stage_metrics("answer_check")

    assert check["questions"] == 20
    for answer in check["answers"]:
        for source in answer["sources"]:
            prefix, _, number = source.rpartition(" ")
            assert prefix == "Egyptian Civil Code, Article"
            assert int(number) in answer["articles"], (answer["id"], source)
        if answer["unaddressed"]:
            assert answer["sources"] == []


def test_c19_the_out_of_scope_questions_are_refused_or_listed(stage_metrics):
    check = stage_metrics("answer_check")
    out_of_scope = [a for a in check["answers"] if not a["in_scope"]]

    assert len(out_of_scope) == check["out_of_scope"] == 11
    assert check["out_of_scope_answered"] == [
        a["id"] for a in out_of_scope if not a["unaddressed"]
    ]
