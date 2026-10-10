"""C19: twenty real questions through ``/ask``'s pipeline, end to end.

Builds the service as the API does (``load_service``: ``champion`` by alias,
the ``models`` service, the LLM endpoint), then answers every out-of-scope
question in the eval set (11), one in-scope question of each kind in Arabic and
English by turns, and one colloquial question (9). It writes:

- ``docs/metrics/answer_check.json``: per question, the gate's decision, the
  articles given to the LLM, the sources and any hallucinated citation;
  ``tests/test_rag.py`` (C19) checks it;
- ``docs/reports/answer_check.md``: each question with its answer, to read.

It sits outside ``dvc repro``, since it calls the LLM (no LLM calls in the
pipeline). Run as ``uv run python scripts/check_answers.py``, with the
``models`` service up and the LLM set in ``.env``.
"""

import os

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import asyncio
import json
import logging
from pathlib import Path

from raglaw.api.main import load_service
from raglaw.config import Settings
from raglaw.rag import Answer, AnswerPipeline
from raglaw.retrieval.scoring import EvalQuestion, read_questions

logger = logging.getLogger("check_answers")


def pick_questions(questions: list[EvalQuestion]) -> list[EvalQuestion]:
    """Every out-of-scope question; then, for each in-scope kind in order of
    first appearance, its first question by id in Arabic and English by turns;
    then the first colloquial question not already picked."""
    ordered = sorted(questions, key=lambda q: q.id)
    picked = [q for q in ordered if not q.in_scope]
    kinds = list(dict.fromkeys(q.kind for q in ordered if q.in_scope))
    for i, kind in enumerate(kinds):
        language = "ar" if i % 2 == 0 else "en"
        picked.append(
            next(q for q in ordered if q.kind == kind and q.language == language)
        )
    picked.append(
        next(
            q for q in ordered if q.speech_register == "colloquial" and q not in picked
        )
    )
    return picked


def row(question: EvalQuestion, answer: Answer) -> dict:
    return {
        "id": question.id,
        "kind": question.kind,
        "language": question.language,
        "register": question.speech_register,
        "in_scope": question.in_scope,
        "expected": question.expected_articles,
        "unaddressed": answer.unaddressed,
        "rank1_rerank_score": answer.rank1_rerank_score,
        "articles": answer.articles,
        "sources": answer.sources,
        "hallucinated": answer.hallucinated,
    }


def report(pairs: list[tuple[EvalQuestion, Answer]], header: str) -> str:
    lines = ["# Answer check (C19)", "", header, ""]
    for question, answer in pairs:
        scope = "in scope" if question.in_scope else "out of scope"
        gate = "refused by the no-answer gate" if answer.unaddressed else "answered"
        lines += [
            f"## {question.id}: {question.kind}, {question.speech_register} ({scope}, {gate})",
            "",
            "```text",
            question.question,
            "```",
            "",
            "```text",
            answer.text,
            "```",
            "",
            f"- Sources: {', '.join(answer.sources) or 'none'}",
            f"- Expected articles: {question.expected_articles or 'none'}",
            f"- Given to the LLM: {answer.articles or 'none'}",
        ]
        if answer.hallucinated:
            lines.append(f"- Hallucinated citations, dropped: {answer.hallucinated}")
        lines.append("")
    return "\n".join(lines)


async def answer_all(
    pipeline: AnswerPipeline, questions: list[EvalQuestion]
) -> list[tuple[EvalQuestion, Answer]]:
    return [(q, await pipeline.answer(q.question)) for q in questions]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    settings = Settings()
    paths = settings.paths
    service = load_service(settings)
    try:
        if service.pipeline is None:
            reasons = {
                n: c.detail for n, c in service.components.items() if not c.ready
            }
            raise SystemExit(f"The service isn't ready: {reasons}")
        questions = pick_questions(
            read_questions(paths.gold_dir / "retrieval_eval.jsonl")
        )
        pairs = asyncio.run(answer_all(service.pipeline, questions))
    finally:
        service.close()
    rows = [row(q, a) for q, a in pairs]
    out_of_scope = [r for r in rows if not r["in_scope"]]
    metrics = {
        "questions": len(rows),
        "llm_model": settings.llm_model,
        "prompt_version": settings.answer.prompt_version,
        "no_answer_threshold": settings.answer.no_answer_threshold,
        "out_of_scope": len(out_of_scope),
        "out_of_scope_refused": sum(r["unaddressed"] for r in out_of_scope),
        "out_of_scope_answered": [
            r["id"] for r in out_of_scope if not r["unaddressed"]
        ],
        "in_scope_refused": [
            r["id"] for r in rows if r["in_scope"] and r["unaddressed"]
        ],
        "hallucinated_citations": sum(len(r["hallucinated"]) for r in rows),
        "answers": rows,
    }
    metrics_out = paths.metrics_dir / "answer_check.json"
    metrics_out.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", "utf-8"
    )
    header = (
        f"{settings.llm_model}, prompt {settings.answer.prompt_version}, no-answer "
        f"threshold {settings.answer.no_answer_threshold}. Generated by "
        "`scripts/check_answers.py`."
    )
    report_out = Path(paths.reports_dir) / "answer_check.md"
    report_out.write_text(report(pairs, header), "utf-8")
    logger.info(
        "%d questions: %d of %d out-of-scope refused; wrote %s and %s",
        len(rows),
        metrics["out_of_scope_refused"],
        len(out_of_scope),
        metrics_out,
        report_out,
    )


if __name__ == "__main__":
    main()
