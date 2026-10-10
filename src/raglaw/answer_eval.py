"""Answer evaluation (Phase 9): the answer set, its CI subset, and RAGAS scores.

The answer set (``data/gold/answer_eval.jsonl``) gives a reference answer to
each in-scope question of the retrieval eval set's tuning half, keyed by the
question's ``id``: the question, its language and its governing articles stay
in ``retrieval_eval.jsonl``, so the two can't drift apart. The references were
drafted by an LLM from the governing articles' text and aren't yet reviewed
(``docs/reports/evaluation_data.md``).

RAGAS scores four things, each from 0 to 1:

- ``faithfulness``: the share of the answer's claims that follow from the
  articles given to the generator; a low score blames generation;
- ``answer_relevancy``: how directly the answer addresses the question;
- ``context_precision``: whether the useful articles were ranked first;
- ``context_recall``: the share of the reference's claims the articles support;
  a low score blames retrieval.

RAGAS is imported only when scoring (the ``ingest`` group), so this module's
data helpers load anywhere.
"""

import math
import statistics
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from itertools import zip_longest
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from raglaw.retrieval.scoring import EvalQuestion, read_questions

METRICS = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")


class AnswerSetError(ValueError):
    """The answer set doesn't match the retrieval eval set."""


class AnswerReference(BaseModel):
    """One line of ``answer_eval.jsonl``."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(
        pattern=r"^q\d{3}$", description="The question's id in the eval set."
    )
    reference: str = Field(
        min_length=1,
        description="What a correct answer says, in the question's language, "
        "drawn from the governing articles only.",
    )


@dataclass(frozen=True)
class AnswerItem:
    """A question with the answer it should get."""

    question: EvalQuestion
    reference: str


def read_answer_set(answers_path: Path, questions_path: Path) -> list[AnswerItem]:
    """
    Read the reference answers and join each to its question.

    returns:
    - items (list[AnswerItem]): in ``id`` order

    exceptions:
    - pydantic.ValidationError: a line doesn't fit ``AnswerReference``
    - AnswerSetError: a duplicate id, an id the eval set lacks, or an
      out-of-scope question (it has no governing article to answer from)
    """
    lines = answers_path.read_text(encoding="utf-8").splitlines()
    references = [AnswerReference.model_validate_json(line) for line in lines if line]
    ids = [r.id for r in references]
    if duplicates := sorted({i for i in ids if ids.count(i) > 1}):
        raise AnswerSetError(f"duplicate ids: {', '.join(duplicates)}")
    questions = {q.id: q for q in read_questions(questions_path)}
    if unknown := sorted(set(ids) - set(questions)):
        raise AnswerSetError(f"ids not in {questions_path.name}: {', '.join(unknown)}")
    if out := sorted(i for i in ids if not questions[i].in_scope):
        raise AnswerSetError(
            f"out-of-scope questions have no reference: {', '.join(out)}"
        )
    return [
        AnswerItem(questions[r.id], r.reference)
        for r in sorted(references, key=lambda r: r.id)
    ]


def ci_subset(items: Sequence[AnswerItem], size: int) -> list[AnswerItem]:
    """
    A fixed, stratified subset for CI: round-robin over (language, kind) strata.

    Each stratum yields its questions in ``id`` order, and the strata take turns
    in sorted order, so every stratum is represented before any gives a second
    question. A question appended with a later id never displaces one already
    in the subset, unless its stratum had run out.

    returns:
    - subset (list[AnswerItem]): ``size`` items (fewer if the set is smaller)
    """
    strata: dict[tuple[str, str], list[AnswerItem]] = {}
    for item in sorted(items, key=lambda i: i.question.id):
        key = (item.question.language, item.question.kind)
        strata.setdefault(key, []).append(item)
    ordered = [strata[k] for k in sorted(strata)]
    picked = [
        item for round_ in zip_longest(*ordered) for item in round_ if item is not None
    ]
    return sorted(picked[:size], key=lambda i: i.question.id)


def ragas_scores(
    samples: Sequence[dict[str, Any]],
    llm: Any,
    embeddings: Any,
    metrics: Iterable[str] = METRICS,
    run_config: Any = None,
) -> list[dict[str, float | None]]:
    """
    Score each sample with RAGAS, with ``llm`` as the judge.

    Fresh metric objects each call: RAGAS's module-level metrics keep the first
    judge they are given, so a stub in one call would leak into the next.

    ``samples`` hold ``user_input``, ``response``, ``retrieved_contexts`` and
    ``reference``; ``llm`` is a LangChain chat model and ``embeddings`` a
    LangChain ``Embeddings`` (answer relevancy compares question embeddings).
    ``run_config`` (a ``ragas.RunConfig``) overrides RAGAS's retries: by
    default a failed judge call is retried up to 10 times, waiting up to 60 s.

    returns:
    - scores (list[dict[str, float | None]]): per sample, each metric; None
      where RAGAS couldn't score it (a judge reply it couldn't parse)
    """
    from ragas import EvaluationDataset, evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import (
        AnswerRelevancy,
        ContextPrecision,
        ContextRecall,
        Faithfulness,
    )

    builders = {
        "faithfulness": Faithfulness,
        "answer_relevancy": AnswerRelevancy,
        "context_precision": ContextPrecision,
        "context_recall": ContextRecall,
    }
    names = list(metrics)
    result = evaluate(
        EvaluationDataset.from_list(list(samples)),
        metrics=[builders[n]() for n in names],
        llm=LangchainLLMWrapper(llm),
        embeddings=LangchainEmbeddingsWrapper(embeddings) if embeddings else None,
        run_config=run_config,
        show_progress=False,
        raise_exceptions=False,
    )
    return [{n: _score(row.get(n)) for n in names} for row in result.scores]


def _score(value: Any) -> float | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    return float(value)


def summarize(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """
    Mean scores over the scored rows, overall and per language.

    A row the no-answer gate refused has no scores; it is counted, not
    averaged. A metric RAGAS couldn't score on a row is left out of that mean.

    returns:
    - summary (dict[str, Any]): ``questions``, ``answered``, ``refused`` (ids),
      ``overall`` and ``by_language`` (metric -> mean, or None)
    """

    def means(subset: list[dict[str, Any]]) -> dict[str, float | None]:
        out: dict[str, float | None] = {}
        for metric in METRICS:
            values = [
                r["scores"][metric]
                for r in subset
                if r["scores"].get(metric) is not None
            ]
            out[metric] = round(statistics.fmean(values), 4) if values else None
        return out

    scored = [r for r in rows if not r["unaddressed"]]
    return {
        "questions": len(rows),
        "answered": len(scored),
        "refused": [r["id"] for r in rows if r["unaddressed"]],
        "overall": means(scored),
        "by_language": {
            lang: means([r for r in scored if r["language"] == lang])
            for lang in sorted({r["language"] for r in rows})
        },
    }
