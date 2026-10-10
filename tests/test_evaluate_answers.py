import json

import pytest

from raglaw.answer_eval import read_answer_set
from raglaw.ingest import evaluate_answers
from raglaw.ingest.evaluate_answers import (
    EndpointError,
    check_endpoints,
    pick,
    run_evaluate_answers,
    run_params,
)
from raglaw.retrieval.scoring import EvalQuestion

pytestmark = pytest.mark.unit


def fake_scores(samples, judge, embeddings):
    """Faithfulness 1.0 when the answer cites Article 44, else 0.0."""
    return [
        {
            "faithfulness": 1.0 if "Article 44" in s["response"] else 0.0,
            "answer_relevancy": 0.9,
            "context_precision": 1.0,
            "context_recall": 1.0,
        }
        for s in samples
    ]


@pytest.fixture
def two_questions(answer_set, make_question):
    """q001 (English) is answered; q002 (Arabic) gets nothing back."""
    answers, eval_set = answer_set(
        [
            make_question("q001", [44]),
            make_question(
                "q002", [45], language="ar", register="msa", question="سؤال؟"
            ),
        ]
    )
    return read_answer_set(answers, eval_set)


def test_answers_are_scored_and_refusals_counted(
    two_questions,
    retrieved,
    make_chunk,
    stub_llm,
    tracking_settings,
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(evaluate_answers, "ragas_scores", fake_scores)

    def retrieve(question):
        return [] if question == "سؤال؟" else retrieved([make_chunk(44)])

    out = {name: tmp_path / name for name in ("answers.json", "report.md", "rows.json")}
    metrics = run_evaluate_answers(
        two_questions,
        retrieve,
        stub_llm("Article 44 sets majority at twenty-one."),
        judge=None,
        embeddings=None,
        settings=tracking_settings,
        metrics_out=out["answers.json"],
        report_out=out["report.md"],
        questions_out=out["rows.json"],
    )

    assert (metrics["questions"], metrics["answered"], metrics["refused"]) == (
        2,
        1,
        ["q002"],
    )
    assert metrics["overall"]["faithfulness"] == 1.0
    assert metrics["by_language"]["ar"]["faithfulness"] is None  # nothing answered
    assert json.loads(out["answers.json"].read_text()) == metrics
    rows = json.loads(out["rows.json"].read_text())
    assert rows[0]["contexts"][0].startswith("### Article 44")
    assert rows[1]["scores"] == {}
    assert "## Lowest faithfulness (1)" in out["report.md"].read_text()


def test_the_stage_stops_without_a_generator_or_a_judge(tracking_settings):
    with pytest.raises(EndpointError, match="RAGLAW_JUDGE_MODEL"):
        check_endpoints(tracking_settings)


def test_a_run_is_compared_by_its_answer_config_and_judge(tracking_settings):
    settings = tracking_settings.model_copy(
        update={"llm_model": "qwen2.5:3b-instruct-fp16", "judge_model": "judge-model"}
    )

    params = run_params(
        settings, chunks_sha256="0" * 64, index_device="cuda", questions=65
    )

    assert "split" not in params  # the answer set is its own
    assert params["llm_model"] == "qwen2.5:3b-instruct-fp16"
    assert params["judge_model"] == "judge-model"
    assert params["prompt_version"] == settings.answer.prompt_version
    assert (params["subset"], params["questions"]) == ("full", 65)
    assert params["retrieval_mode"] == settings.search.mode


def test_the_ci_subset_is_picked_by_the_config(two_questions):
    assert pick(two_questions, "full", 1) == two_questions
    assert len(pick(two_questions, "ci", 1)) == 1
    assert isinstance(pick(two_questions, "ci", 1)[0].question, EvalQuestion)
