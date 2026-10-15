"""The ``evaluate_answers`` stage (Phase 9): RAGAS scores for ``/ask``'s answers.

For each question with a reference answer (``data/gold/answer_eval.jsonl``), or
the CI subset of them, it runs ``/ask``'s pipeline over the index with the
config in ``params.yaml``, as ``evaluate_retrieval`` does, so
``dvc exp run -S`` compares configs on faithfulness. RAGAS then scores each
answer with a judge LLM (D23: a stronger model than the generator, so the
generator doesn't grade itself) and the pinned embedder.

It calls two LLMs, so it needs ``RAGLAW_LLM_*`` (the generator) and
``RAGLAW_JUDGE_*`` in ``.env``, and stops before any work without them. DVC
can't see either endpoint, so each run logs both.

Writes ``docs/metrics/answers.json``, ``docs/reports/answer_eval.md`` (the
lowest-faithfulness answers with what they were given) and
``docs/reports/answer_questions.json`` (one record per question), and logs one
run to the ``answers`` experiment.
"""

import argparse
import asyncio
import json
import logging
import os
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

from raglaw.answer_eval import (
    METRICS,
    AnswerItem,
    ci_subset,
    ragas_scores,
    read_answer_set,
    summarize,
)
from raglaw.config import Settings
from raglaw.ingest.evaluate_retrieval import (
    CONFIG_SECTIONS,
    flatten_metrics,
    inputs_hash,
    repealed_inputs,
    run_identity,
)
from raglaw.ingest.evaluate_retrieval import run_params as retrieval_params
from raglaw.llm import ChatModel, LLMError
from raglaw.rag import Answer, AnswerPipeline
from raglaw.tracking import params_diff, stage_run

logger = logging.getLogger(__name__)

# Answers shown in the report, lowest faithfulness first.
REPORT_WORST = 10
# Seconds to wait before each retry of a failed generator call.
GENERATOR_RETRY_WAITS = (10, 30)


class EndpointError(RuntimeError):
    """The generator or the judge isn't configured."""


def run_params(
    settings: Settings,
    *,
    chunks_sha256: str,
    index_device: str | None,
    questions: int,
) -> dict[str, object]:
    """
    The whole answer config a run is compared by.

    returns:
    - params (dict[str, object]): the retrieval config (``evaluate_retrieval``'s
      params, without its split), the generator, the prompt and gate, the
      judge, and which questions were scored. Each LLM is named by its model
      and its endpoint (the base URL, never the key), since one model name can
      be served by different providers
    """
    params = retrieval_params(
        settings, chunks_sha256=chunks_sha256, index_device=index_device
    )
    params.pop("split")
    answer = settings.answer
    return params | {
        "llm_model": settings.llm_model,
        "llm_base_url": settings.llm_base_url and str(settings.llm_base_url),
        "prompt_version": answer.prompt_version,
        "no_answer_threshold": answer.no_answer_threshold,
        "max_sources": answer.max_sources,
        "temperature": answer.temperature,
        "max_tokens": answer.max_tokens,
        "judge_model": settings.judge_model,
        "judge_base_url": settings.judge_base_url and str(settings.judge_base_url),
        "subset": settings.answer_evaluation.subset,
        "questions": questions,
    }


def pick(items: Sequence[AnswerItem], subset: str, ci_size: int) -> list[AnswerItem]:
    """The questions ``answer_evaluation.subset`` names."""
    return list(items) if subset == "full" else ci_subset(items, ci_size)


def record(item: AnswerItem, answer: Answer) -> dict[str, Any]:
    """One question's outcome, before scoring."""
    q = item.question
    return {
        "id": q.id,
        "language": q.language,
        "register": q.speech_register,
        "kind": q.kind,
        "question": q.question,
        "reference": item.reference,
        "expected": q.expected_articles,
        "match": q.match,
        "answer": answer.text,
        "sources": answer.sources,
        "unaddressed": answer.unaddressed,
        "articles": answer.articles,
        "hallucinated": answer.hallucinated,
        "contexts": answer.contexts,
        "scores": {},
    }


def score(rows: list[dict[str, Any]], judge: Any, embeddings: Any) -> None:
    """Attach RAGAS scores to each answered row; a refused row keeps none."""
    answered = [r for r in rows if not r["unaddressed"]]
    if not answered:
        return
    samples = [
        {
            "user_input": r["question"],
            "response": r["answer"],
            "retrieved_contexts": r["contexts"],
            "reference": r["reference"],
        }
        for r in answered
    ]
    for row, scores in zip(
        answered, ragas_scores(samples, judge, embeddings), strict=True
    ):
        row["scores"] = scores


def _value(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.3f}"


def report(rows: Sequence[dict[str, Any]], summary: dict[str, Any], header: str) -> str:
    """The scores, then the lowest-faithfulness answers with their articles."""
    lines = ["# Answer evaluation", "", header, ""]
    lines += [
        "| Scope | " + " | ".join(METRICS) + " |",
        "| --- |" + " --- |" * len(METRICS),
    ]
    groups = {"overall": summary["overall"], **summary["by_language"]}
    for name, means in groups.items():
        lines.append(
            f"| {name} | " + " | ".join(_value(means[m]) for m in METRICS) + " |"
        )
    lines += [
        "",
        (
            f"{summary['answered']} of {summary['questions']} questions answered; "
            f"refused by the no-answer gate: {', '.join(summary['refused']) or 'none'}."
        ),
        "",
    ]
    scored = [r for r in rows if r["scores"].get("faithfulness") is not None]
    worst = sorted(scored, key=lambda r: (r["scores"]["faithfulness"], r["id"]))
    lines += [f"## Lowest faithfulness ({min(REPORT_WORST, len(worst))})", ""]
    for r in worst[:REPORT_WORST]:
        s = r["scores"]
        lines += [
            f"### {r['id']}: {r['kind']}, {r['register']}",
            "",
            "- Scores: " + ", ".join(f"{m} {_value(s.get(m))}" for m in METRICS),
            f"- Articles given: {r['articles']}; expected: {r['expected']}",
            "",
            "```text",
            r["question"],
            "```",
            "",
            "Answer:",
            "",
            "```text",
            r["answer"],
            "```",
            "",
            "Reference:",
            "",
            "```text",
            r["reference"],
            "```",
            "",
        ]
    return "\n".join(lines)


async def _answer(pipeline: AnswerPipeline, question: str) -> Answer:
    """
    One answer, retrying a failed generator call after each wait in
    ``GENERATOR_RETRY_WAITS``: one slow reply shouldn't discard a whole run.

    exceptions:
    - LLMError: the generator still failed after every retry; the run stops,
      since dropping a question would make configs incomparable
    """
    for wait in (*GENERATOR_RETRY_WAITS, None):
        try:
            return await pipeline.answer(question)
        except LLMError as exc:
            if wait is None:
                raise
            logger.warning("Generator failed (%s); retrying in %s s", exc, wait)
            await asyncio.sleep(wait)
    raise AssertionError("unreachable")


async def _answer_all(
    pipeline: AnswerPipeline, items: Sequence[AnswerItem]
) -> list[Answer]:
    return [await _answer(pipeline, item.question.question) for item in items]


def run_evaluate_answers(
    items: Sequence[AnswerItem],
    retrieve: Callable[[str], list[Document]],
    generator: ChatModel,
    judge: Any,
    embeddings: Any,
    settings: Settings,
    *,
    metrics_out: Path,
    report_out: Path,
    questions_out: Path,
) -> dict[str, Any]:
    """
    Answer each question with ``/ask``'s pipeline, score the answers, and write
    the metrics, the report and the per-question records.

    returns:
    - metrics (dict[str, Any]): what was written to ``metrics_out``
    """
    pipeline = AnswerPipeline(retrieve, generator, settings.answer)
    answers = asyncio.run(_answer_all(pipeline, items))
    rows = [record(i, a) for i, a in zip(items, answers, strict=True)]
    score(rows, judge, embeddings)
    summary = summarize(rows)
    metrics = {
        "llm_model": settings.llm_model,
        "judge_model": settings.judge_model,
        "prompt_version": settings.answer.prompt_version,
        "subset": settings.answer_evaluation.subset,
        "hallucinated_citations": sum(len(r["hallucinated"]) for r in rows),
        **summary,
    }
    for path in (metrics_out, report_out, questions_out):
        path.parent.mkdir(parents=True, exist_ok=True)
    metrics_out.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        "utf-8",
    )
    questions_out.write_text(
        json.dumps(rows, indent=2, ensure_ascii=False) + "\n", "utf-8"
    )
    header = (
        f"Generator {settings.llm_model}, judge {settings.judge_model}, prompt "
        f"{settings.answer.prompt_version}, {settings.answer_evaluation.subset} set. "
        "Generated by `evaluate_answers`; scores are RAGAS's, from 0 to 1."
    )
    report_out.write_text(report(rows, summary, header), "utf-8")
    logger.info(
        "Scored %d answers (%d refused): faithfulness %s",
        summary["answered"],
        len(summary["refused"]),
        summary["overall"]["faithfulness"],
    )
    return metrics


def check_endpoints(settings: Settings) -> None:
    """
    exceptions:
    - EndpointError: naming each generator or judge setting that is missing,
      or the judge being the generator (D23: it would grade itself)
    """
    values = {
        "RAGLAW_LLM_BASE_URL": settings.llm_base_url,
        "RAGLAW_LLM_MODEL": settings.llm_model,
        "RAGLAW_LLM_API_KEY": settings.llm_api_key,
        "RAGLAW_JUDGE_BASE_URL": settings.judge_base_url,
        "RAGLAW_JUDGE_MODEL": settings.judge_model,
        "RAGLAW_JUDGE_API_KEY": settings.judge_api_key,
    }
    if missing := [name for name, value in values.items() if not value]:
        raise EndpointError(
            f"evaluate_answers calls a generator and a judge; set {', '.join(missing)} "
            "in .env"
        )
    if settings.judge_model == settings.llm_model:
        raise EndpointError(
            f"the judge and the generator are both {settings.llm_model}; D23 needs "
            "a judge that doesn't grade its own answers"
        )


def main(argv: list[str] | None = None) -> None:
    settings = Settings()
    paths = settings.paths
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--answers", type=Path, default=paths.gold_dir / "answer_eval.jsonl"
    )
    ap.add_argument(
        "--eval", type=Path, default=paths.gold_dir / "retrieval_eval.jsonl"
    )
    ap.add_argument("--index", type=Path, default=paths.index_dir)
    ap.add_argument("--metrics", type=Path, default=paths.metrics_dir / "answers.json")
    ap.add_argument("--report", type=Path, default=paths.reports_dir / "answer_eval.md")
    ap.add_argument(
        "--questions", type=Path, default=paths.reports_dir / "answer_questions.json"
    )
    args = ap.parse_args(argv)
    try:
        check_endpoints(settings)
    except EndpointError as exc:
        raise SystemExit(str(exc)) from exc

    from langchain_openai import ChatOpenAI  # ingest group (ragas)

    from raglaw.llm import OpenAIChat
    from raglaw.retrieval.dense import DenseManifest
    from raglaw.retrieval.embeddings import huggingface_embeddings, model_tokenizer
    from raglaw.retrieval.manifest import read_manifest
    from raglaw.retrieval.repealed import load_repealed
    from raglaw.retrieval.rerank import CrossEncoderReranker
    from raglaw.retrieval.retriever import HybridRetriever

    evaluation = settings.answer_evaluation
    items = pick(
        read_answer_set(args.answers, args.eval), evaluation.subset, evaluation.ci_size
    )
    dense_dir, bm25_dir = args.index / "dense", args.index / "bm25"
    manifest = read_manifest(dense_dir, DenseManifest)
    baseline_rev = os.environ.get("DVC_EXP_BASELINE_REV")
    sections = (*CONFIG_SECTIONS, "answer", "answer_evaluation")
    run_name, tags = run_identity(
        os.environ.get("DVC_EXP_NAME"),
        baseline_rev,
        params_diff(baseline_rev or "HEAD", sections=sections),
    )
    embeddings = huggingface_embeddings(settings.embedding, settings.device)
    with stage_run(
        "evaluate_answers",
        input_hash=inputs_hash(
            args.answers,
            args.eval,
            dense_dir / "manifest.json",
            bm25_dir / "manifest.json",
            *repealed_inputs(settings),
        ),
        settings=settings,
        experiment="answers",
        run_name=run_name,
        tags=tags,
    ) as run:
        run.log_params(
            run_params(
                settings,
                chunks_sha256=manifest.chunks_sha256,
                index_device=manifest.device,
                questions=len(items),
            )
        )
        with HybridRetriever.from_index(
            dense_dir,
            bm25_dir,
            embedding=settings.embedding,
            retrieval=settings.retrieval,
            search=settings.search,
            embeddings=embeddings,
            tokenizer=(
                model_tokenizer(settings.embedding)
                if settings.retrieval.bm25_tokenizer == "model_subwords"
                else None
            ),
            reranker=(
                CrossEncoderReranker(settings.reranker, settings.device)
                if settings.search.rerank
                else None
            ),
            repealed=(
                load_repealed(Path(settings.paths.repealed))
                if settings.search.repealed_fallback
                else None
            ),
        ) as retriever:
            metrics = run_evaluate_answers(
                items,
                retriever.invoke,
                OpenAIChat(
                    settings.llm_base_url,
                    settings.llm_api_key.get_secret_value(),
                    settings.llm_model,
                ),
                ChatOpenAI(
                    base_url=settings.judge_base_url,
                    model=settings.judge_model,
                    api_key=settings.judge_api_key.get_secret_value(),
                    temperature=0,
                ),
                embeddings,
                settings,
                metrics_out=args.metrics,
                report_out=args.report,
                questions_out=args.questions,
            )
        run.log_metrics(flatten_metrics(metrics))
        run.log_artifact(args.report, artifact_path="reports")
        run.log_artifact(args.questions, artifact_path="questions")


if __name__ == "__main__":
    main()
