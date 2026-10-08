"""The ``evaluate_retrieval`` stage: does a question find the articles that govern it?

Before any LLM is involved, every question of one half of the eval set
(``evaluation.split``, D15) is run through ``HybridRetriever`` with the
configured ``search`` choices and scored on articles (``raglaw.retrieval.scoring``):
recall@1/3/5/10 and MRR overall and per kind, language and register, the
rank-1 scores of in- and out-of-scope questions, and bilingual agreement.

It writes the metrics (no timings, so two runs on one index are identical, C18)
and a misses report: every in-scope question missed at 5 with what came back
instead, every pair whose languages disagree on the top article, and what each
out-of-scope question retrieved first. Each run is one MLflow run in the
``retrieval`` experiment, with the whole config as params, so configs are
compared there.

Run as ``python -m raglaw.ingest.evaluate_retrieval``.
"""

import argparse
import hashlib
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from langchain_core.embeddings import Embeddings

from raglaw.config import Embedding, Evaluation, Retrieval, Search, Settings
from raglaw.ingest import chunk
from raglaw.retrieval.dense import DenseManifest
from raglaw.retrieval.embeddings import (
    ModelTokenizer,
    huggingface_embeddings,
    model_tokenizer,
)
from raglaw.retrieval.manifest import read_manifest
from raglaw.retrieval.retriever import HybridRetriever
from raglaw.retrieval.scoring import (
    KS,
    EvalQuestion,
    QuestionScore,
    Retrieved,
    bilingual_agreement,
    dump,
    read_questions,
    score_question,
    split_questions,
    summarize,
)
from raglaw.schema import LogEvent
from raglaw.tracking import sha256_file, stage_run

logger = logging.getLogger(__name__)

# The headline cut-off (recall@5, D17): a question not found within it is a miss.
MISS_K = 5


def evaluate(
    questions: Sequence[EvalQuestion],
    retriever: HybridRetriever,
    ks: Sequence[int] = KS,
) -> list[QuestionScore]:
    """
    Retrieve for every question and score it.

    returns:
    - scores (list[QuestionScore]): in question order
    """
    return [
        score_question(q, Retrieved.from_documents(retriever.invoke(q.question)), ks)
        for q in questions
    ]


def _question_block(question: EvalQuestion) -> list[str]:
    # A code block, so right-to-left text never reorders the English around it.
    return ["```text", question.question, "```", ""]


def _score(value: float | None) -> str:
    return "none" if value is None else f"{value:.3f}"


def _ranges(numbers: list[int]) -> str:
    """``[54, 55, 56, 80]`` as ``54–56, 80``."""
    runs: list[list[int]] = []
    for n in sorted(numbers):
        if runs and n == runs[-1][-1] + 1:
            runs[-1].append(n)
        else:
            runs.append([n])
    return ", ".join(f"{r[0]}–{r[-1]}" if len(r) > 1 else str(r[0]) for r in runs)


def _expected(ranks: Mapping[int, int | None]) -> str:
    """Found articles with their ranks, then the missing ones as ranges."""
    found = "; ".join(
        f"{a} (rank {r})"
        for a, r in sorted(ranks.items(), key=lambda x: x[1] or 0)
        if r is not None
    )
    missing = [a for a, r in ranks.items() if r is None]
    parts = [found] if found else []
    if missing:
        parts.append(f"{_ranges(missing)} (not returned)")
    return "; ".join(parts)


def misses_report(scores: Sequence[QuestionScore], title: str) -> str:
    """
    Write the misses report for one run.

    returns:
    - text (str): Markdown: in-scope questions missed at ``MISS_K``, with their
      expected articles' ranks and what came back; pairs whose languages
      disagree on the top article; out-of-scope questions' top article and
      scores
    """
    in_scope = [s for s in scores if s.question.in_scope]
    missed = [s for s in in_scope if not s.hits[MISS_K]]
    lines = [f"# Retrieval misses: {title}", ""]
    lines += [f"## Missed at {MISS_K}: {len(missed)} of {len(in_scope)}", ""]
    for s in missed:
        q = s.question
        expected = _expected(s.ranks)
        returned = ", ".join(str(a) for a in s.retrieved.articles) or "nothing"
        lines += [f"### {q.id} ({q.kind}, {q.language}, {q.speech_register})", ""]
        lines += _question_block(q)
        lines += [
            f"- Expected: {expected}",
            f"- Match: {q.match}",
            f"- Returned: {returned}",
            f"- Legal basis: {q.legal_basis}",
            "",
        ]
    pairs: dict[str, list[QuestionScore]] = {}
    for s in scores:
        if s.question.pair_id is not None:
            pairs.setdefault(s.question.pair_id, []).append(s)
    complete = {p: m for p, m in sorted(pairs.items()) if len(m) == 2}
    disagree = {
        p: m for p, m in complete.items() if m[0].top_article != m[1].top_article
    }
    lines += [
        (
            "## Pairs that disagree on the top article: "
            f"{len(disagree)} of {len(complete)}"
        ),
        "",
    ]
    for pair_id, members in disagree.items():
        lines += [f"### {pair_id} ({members[0].question.kind})", ""]
        for s in members:
            expected = ", ".join(str(a) for a in s.question.expected_articles)
            lines += [
                (
                    f"- {s.question.id} ({s.question.language}): "
                    f"top {s.top_article}, expected {expected}"
                ),
                "",
            ]
            lines += _question_block(s.question)
    out = [s for s in scores if not s.question.in_scope]
    lines += [f"## Out of scope: what came back first ({len(out)})", ""]
    for s in out:
        top = s.retrieved.top_scores
        lines += [
            (
                f"- {s.question.id}: Article {s.top_article} "
                f"(dense {_score(top.get('dense_score'))}, "
                f"bm25 {_score(top.get('bm25_score'))}, "
                f"rrf {_score(top.get('rrf_score'))})"
            ),
            "",
        ]
        lines += _question_block(s.question)
    return "\n".join(lines).rstrip("\n") + "\n"


def flatten_metrics(metrics: Mapping[str, Any], prefix: str = "") -> dict[str, float]:
    """
    The numeric leaves of nested metrics, as MLflow metric names.

    returns:
    - flat (dict[str, float]): ``by_kind.rule.recall_at_5``-style keys;
      strings, booleans and None are left out
    """
    flat: dict[str, float] = {}
    for key, value in metrics.items():
        name = f"{prefix}{key}"
        if isinstance(value, Mapping):
            flat |= flatten_metrics(value, f"{name}.")
        elif isinstance(value, int | float) and not isinstance(value, bool):
            flat[name] = value
    return flat


def run_evaluate_retrieval(
    eval_path: Path,
    dense_dir: Path,
    bm25_dir: Path,
    metrics_out: Path,
    report_out: Path,
    *,
    embedding: Embedding,
    retrieval: Retrieval,
    search: Search,
    evaluation: Evaluation,
    embeddings: Embeddings,
    tokenizer: ModelTokenizer | None = None,
) -> dict[str, Any]:
    """
    Score one half of the eval set against the index, writing metrics and a report.

    returns:
    - metrics (dict[str, Any]): what was written to ``metrics_out``
    """
    questions = split_questions(read_questions(eval_path))[evaluation.split]
    with HybridRetriever.from_index(
        dense_dir,
        bm25_dir,
        embedding=embedding,
        retrieval=retrieval,
        search=search,
        embeddings=embeddings,
        tokenizer=tokenizer,
    ) as retriever:
        scores = evaluate(questions, retriever)
    metrics: dict[str, Any] = {
        "split": evaluation.split,
        "questions": len(questions),
        "mode": search.mode,
        "document_text": retrieval.document_text,
        "bm25_tokenizer": retrieval.bm25_tokenizer,
        **summarize(scores),
        "bilingual": bilingual_agreement(scores),
    }
    for path in (metrics_out, report_out):
        path.parent.mkdir(parents=True, exist_ok=True)
    metrics_out.write_text(dump(metrics), encoding="utf-8")
    report_out.write_text(
        misses_report(scores, f"{search.mode}, {evaluation.split} half"),
        encoding="utf-8",
    )
    overall = metrics["overall"]
    logger.info(
        "Recall@5 %.3f, MRR %.3f over %d in-scope questions (%s half)",
        overall["recall_at_5"],
        overall["mrr"],
        overall["questions"],
        evaluation.split,
        extra={"event_type": LogEvent.STAGE_COMPLETED, **flatten_metrics(overall)},
    )
    return metrics


def run_params(settings: Settings, *, chunks_sha256: str) -> dict[str, object]:
    """
    The whole retrieval config a run is compared by, under the names used across runs.

    returns:
    - params (dict[str, object]): the chunking (``chunk.run_params``), the
      embedding model, the index's text and tokenizer, every search choice,
      the split, and the hash of the chunks the index holds
    """
    search = settings.search
    return {
        **chunk.run_params(settings.chunking, settings.embedding),
        "embedding_model": settings.embedding.model,
        "embedding_revision": settings.embedding.revision,
        "document_text": settings.retrieval.document_text,
        "bm25_tokenizer": settings.retrieval.bm25_tokenizer,
        "retrieval_mode": search.mode,
        "article_lookup": search.article_lookup,
        "candidates": search.candidates,
        "rrf_k": search.rrf_k,
        "top_k": search.top_k,
        "split": settings.evaluation.split,
        "chunks_sha256": chunks_sha256,
    }


def _input_hash(*paths: Path) -> str:
    """One hash over the eval set and both manifests, naming what was scored."""
    digest = hashlib.sha256()
    for path in paths:
        digest.update(sha256_file(path).encode())
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> None:
    settings = Settings()
    paths = settings.paths
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--eval", type=Path, default=paths.gold_dir / "retrieval_eval.jsonl"
    )
    ap.add_argument("--index", type=Path, default=paths.index_dir)
    ap.add_argument(
        "--metrics", type=Path, default=paths.metrics_dir / "retrieval.json"
    )
    ap.add_argument(
        "--report", type=Path, default=paths.reports_dir / "retrieval_misses.md"
    )
    args = ap.parse_args(argv)
    dense_dir, bm25_dir = args.index / "dense", args.index / "bm25"
    tokenizer = (
        model_tokenizer(settings.embedding)
        if settings.retrieval.bm25_tokenizer == "model_subwords"
        else None
    )
    with stage_run(
        "evaluate_retrieval",
        input_hash=_input_hash(
            args.eval, dense_dir / "manifest.json", bm25_dir / "manifest.json"
        ),
        settings=settings,
        experiment="retrieval",
    ) as run:
        manifest = read_manifest(dense_dir, DenseManifest)
        run.log_params(run_params(settings, chunks_sha256=manifest.chunks_sha256))
        metrics = run_evaluate_retrieval(
            args.eval,
            dense_dir,
            bm25_dir,
            args.metrics,
            args.report,
            embedding=settings.embedding,
            retrieval=settings.retrieval,
            search=settings.search,
            evaluation=settings.evaluation,
            embeddings=huggingface_embeddings(settings.embedding),
            tokenizer=tokenizer,
        )
        run.log_metrics(flatten_metrics(metrics))
        run.log_artifact(args.report, artifact_path="reports")


if __name__ == "__main__":
    main()
