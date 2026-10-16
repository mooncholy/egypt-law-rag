import pytest

from raglaw.config import Gate
from raglaw.gate import evaluate_gate

pytestmark = pytest.mark.unit

GATE = Gate(min_faithfulness=0.75, max_recall_drop=1)


def answers(faithfulness, subset="ci"):
    return {
        "overall": {"faithfulness": faithfulness},
        "subset": subset,
        "questions": 20,
    }


def retrieval(recall, questions=65):
    return {"overall": {"recall_at_5": recall, "questions": questions}}


def test_a_good_build_passes():
    result = evaluate_gate(answers(0.86), retrieval(53 / 65), retrieval(53 / 65), GATE)

    assert result.passed
    assert all(line.startswith("pass") for line in result.lines)


@pytest.mark.parametrize(
    ("scored", "why"),
    [
        (answers(0.74), "faithfulness 0.74"),
        (answers(None), "faithfulness None"),
        (answers(0.9, subset="full"), "on the full set"),  # not the CI set
    ],
)
def test_answers_below_the_bar_or_off_the_ci_set_fail(scored, why):
    result = evaluate_gate(scored, retrieval(0.8), None, GATE)

    assert not result.passed
    assert why in result.lines[0]


def test_recall_may_lose_one_question_but_not_two():
    base = retrieval(53 / 65)

    assert evaluate_gate(answers(0.9), retrieval(52 / 65), base, GATE).passed
    lost_two = evaluate_gate(answers(0.9), retrieval(51 / 65), base, GATE)
    assert not lost_two.passed
    assert "+2 questions lost" in lost_two.lines[1]


def test_without_a_base_score_recall_is_reported_not_gated():
    result = evaluate_gate(answers(0.9), retrieval(0.1), None, GATE)

    assert result.passed
    assert result.lines[1].startswith("skip")
