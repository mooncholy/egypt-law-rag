import pytest

from raglaw.config import Gate
from raglaw.gate import evaluate_gate

pytestmark = pytest.mark.unit

GATE = Gate(min_faithfulness=0.75, max_recall_drop=1)
CI_SIZE = 20


def answers(faithfulness, subset="ci", questions=CI_SIZE):
    return {
        "overall": {"faithfulness": faithfulness},
        "subset": subset,
        "questions": questions,
    }


def retrieval(recall, questions=65, split="tuning"):
    return {
        "split": split,
        "overall": {"recall_at_5": recall, "questions": questions},
    }


def gate(scored, found, base):
    return evaluate_gate(scored, found, base, GATE, CI_SIZE)


def test_a_good_build_passes():
    result = gate(answers(0.86), retrieval(53 / 65), retrieval(53 / 65))

    assert result.passed
    assert all(line.startswith("pass") for line in result.lines)


@pytest.mark.parametrize(
    ("scored", "why"),
    [
        (answers(0.74), "faithfulness 0.74"),
        (answers(None), "faithfulness None"),
        (answers(0.9, subset="full"), "on the full set"),  # not the CI set
        (answers(0.9, questions=12), "12 of its 20 questions"),  # a cut CI set
    ],
)
def test_answers_below_the_bar_or_off_the_ci_set_fail(scored, why):
    result = gate(scored, retrieval(0.8), None)

    assert not result.passed
    assert why in result.lines[0]


def test_recall_may_lose_one_question_but_not_two():
    base = retrieval(53 / 65)

    assert gate(answers(0.9), retrieval(52 / 65), base).passed
    lost_two = gate(answers(0.9), retrieval(51 / 65), base)
    assert not lost_two.passed
    assert "+2 questions lost" in lost_two.lines[1]


def test_without_a_base_score_recall_is_reported_not_gated():
    result = gate(answers(0.9), retrieval(0.1), None)

    assert result.passed
    assert result.lines[1].startswith("skip")


def test_retrieval_off_the_tuning_half_fails_and_a_heldout_base_is_not_compared():
    heldout = gate(answers(0.9), retrieval(0.95, split="heldout"), retrieval(0.8))
    no_base = gate(answers(0.9), retrieval(0.1), retrieval(0.9, split="heldout"))

    assert not heldout.passed
    assert "on the heldout half" in heldout.lines[1]
    assert no_base.passed and no_base.lines[1].startswith("skip")
