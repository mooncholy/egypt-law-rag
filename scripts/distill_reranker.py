"""Distill the reranker (Phase 13): questions, teacher labels, two students, one comparison.

Each step reads the previous one's output (``raglaw.retrieval.distill``):

- ``questions``: the generator (``RAGLAW_LLM_*`` in ``.env``) writes one question
  per sampled article into ``data/distill/questions.jsonl``.
- ``label``: the teacher reranks each question's fused candidates; the pairs,
  with the teacher's logits and the hard labels, go to ``data/distill/pairs.jsonl``.
- ``train``: the soft (teacher's logits) and hard (labels only) students, in
  ``build/distill/{soft,hard}``.
- ``evaluate``: the teacher, the untrained student and both trained ones on the
  whole retrieval eval set: recall@5, MRR, context precision@5, and the latency
  of one question's ``rerank_depth`` candidates (500 timed iterations, run
  twice) into ``docs/metrics/distillation.json`` and ``docs/reports/distillation.md``.

Run on the GPU: ``uv run --group distill --no-group torch-cpu --group torch-gpu python
scripts/distill_reranker.py <step>``. The generator's GPU memory is needed by
the other steps, so stop it after ``questions``.
"""

import argparse
import asyncio
import json
import logging
import platform
import statistics
from collections.abc import Callable
from pathlib import Path
from typing import Any

from raglaw.bench import benchmark
from raglaw.config import Settings
from raglaw.ingest.evaluate_retrieval import evaluate, repealed_ranges
from raglaw.llm import OpenAIChat
from raglaw.records import read_records
from raglaw.retrieval.distill import (
    LocalCrossEncoder,
    RecordingScorer,
    context_precision,
    held_out_articles,
    pairs_for,
    pick_articles,
    train_student,
    write_questions,
)
from raglaw.retrieval.embeddings import huggingface_embeddings, model_tokenizer
from raglaw.retrieval.rerank import CrossEncoderReranker, RerankScorer
from raglaw.retrieval.retriever import HybridRetriever
from raglaw.retrieval.scoring import read_questions, summarize
from raglaw.schema import Article

logger = logging.getLogger("distill")

DATA = Path("data/distill")
MODELS = Path("build/distill")
WARMUP, ITERATIONS = 20, 500


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text("utf-8").splitlines()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), "utf-8"
    )


def open_retriever(
    settings: Settings, reranker: RerankScorer, *, fallback: bool
) -> HybridRetriever:
    index = Path(settings.paths.index_dir)
    search = settings.search
    if not fallback:
        search = search.model_copy(update={"repealed_fallback": False})
    return HybridRetriever.from_index(
        index / "dense",
        index / "bm25",
        embedding=settings.embedding,
        retrieval=settings.retrieval,
        search=search,
        embeddings=huggingface_embeddings(settings.embedding, settings.device),
        tokenizer=(
            model_tokenizer(settings.embedding)
            if settings.retrieval.bm25_tokenizer == "model_subwords"
            else None
        ),
        reranker=reranker,
        repealed=repealed_ranges(settings) if fallback else None,
    )


def eval_questions(settings: Settings):
    return read_questions(Path(settings.paths.gold_dir) / "retrieval_eval.jsonl")


def step_questions(settings: Settings) -> None:
    d = settings.distillation
    articles = read_records(Path(settings.paths.corpus_dir) / "articles.json", Article)
    picked = pick_articles(
        articles, held_out_articles(eval_questions(settings)), d.questions, d.seed
    )
    if settings.llm_base_url is None or settings.llm_model is None:
        raise SystemExit("set RAGLAW_LLM_BASE_URL and RAGLAW_LLM_MODEL in .env")
    llm = OpenAIChat(
        str(settings.llm_base_url),
        settings.llm_api_key.get_secret_value() if settings.llm_api_key else "none",
        settings.llm_model,
    )
    rows = asyncio.run(write_questions(picked, llm))
    write_jsonl(DATA / "questions.jsonl", rows)
    logger.info(
        "%d questions (of %d articles) by %s", len(rows), len(picked), llm.model
    )


def step_label(settings: Settings) -> None:
    questions = read_jsonl(DATA / "questions.jsonl")
    teacher = RecordingScorer(CrossEncoderReranker(settings.reranker, settings.device))
    pairs = []
    with open_retriever(settings, teacher, fallback=False) as retriever:
        article_of = retriever.text_articles()
        for q in questions:
            teacher.calls.clear()
            retriever.invoke(q["question"])
            (query, texts, scores) = teacher.calls[0]
            pairs += pairs_for(
                query,
                texts,
                scores,
                [article_of[t] for t in texts],
                q["article_number"],
            )
    write_jsonl(DATA / "pairs.jsonl", pairs)
    found = sum(p["label"] for p in pairs)
    logger.info(
        "%d pairs from %d questions; the source article among the candidates %d times",
        len(pairs),
        len(questions),
        int(found),
    )


def step_train(settings: Settings) -> None:
    pairs = read_jsonl(DATA / "pairs.jsonl")
    for target in ("soft", "hard"):
        train_student(
            pairs,
            MODELS / target,
            target=target,
            distillation=settings.distillation,
            device=settings.device,
        )
        logger.info("Trained the %s student into %s", target, MODELS / target)


def _hardware(device: str) -> str:
    if device.startswith("cuda"):
        import torch

        return torch.cuda.get_device_name(0)
    return f"CPU ({platform.processor() or platform.machine()})"


def _synchronize(device: str) -> Callable[[], None]:
    if device.startswith("cuda"):
        import torch

        return torch.cuda.synchronize
    return lambda: None


def _parameters(reranker: RerankScorer) -> int:
    return sum(p.numel() for p in reranker.model.parameters())  # type: ignore[attr-defined]


def measure(settings: Settings, name: str, reranker: RerankScorer) -> dict[str, Any]:
    """One reranker on the whole eval set, then its latency on the recorded calls."""
    recorder = RecordingScorer(reranker)
    questions = eval_questions(settings)
    with open_retriever(settings, recorder, fallback=True) as retriever:
        scores = evaluate(questions, retriever)
    in_scope = [s for s in scores if s.question.in_scope]
    overall = summarize(scores)["overall"]
    depth = settings.search.rerank_depth
    calls = [(q, t) for q, t, _ in recorder.calls if len(t) == depth]
    sync = _synchronize(settings.device)
    runs = [
        benchmark(
            lambda i: reranker.score(*calls[i % len(calls)]),
            warmup=WARMUP,
            iterations=ITERATIONS,
            sync=sync,
        )
        for _ in range(2)
    ]
    row = {
        "reranker": name,
        "parameters": _parameters(reranker),
        "questions": overall["questions"],
        "recall_at_5": overall["recall_at_5"],
        "mrr": overall["mrr"],
        "context_precision_at_5": statistics.fmean(
            context_precision(s.retrieved.articles, set(s.question.expected_articles))
            for s in in_scope
        ),
        "latency": [r.to_dict() for r in runs],
        "p95_rerun_change": runs[1].p95_ms / runs[0].p95_ms - 1,
        "candidates": depth,
        "hardware": _hardware(settings.device),
    }
    logger.info(
        "%s: recall@5 %.3f, CP@5 %.3f, p95 %.1f ms",
        name,
        row["recall_at_5"],
        row["context_precision_at_5"],
        runs[0].p95_ms,
    )
    return row


def report(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Reranker distillation",
        "",
        (
            "Generated by `scripts/distill_reranker.py evaluate`. Every row "
            f"reranks the same {rows[0]['questions']} in-scope questions of the "
            "retrieval eval set (both halves) through the production retriever; "
            "context precision@5 uses the gold articles as the reference. "
            "Latency: one question against its "
            f"{rows[0]['candidates']} fused candidates, {WARMUP} warmup and "
            f"{ITERATIONS} timed iterations, synchronized, measured twice."
        ),
        "",
        (
            "| Reranker | Parameters | Recall@5 | MRR | Context precision@5 "
            "| p50 ms | p95 ms | p95 on rerun | Hardware |"
        ),
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in rows:
        first = r["latency"][0]
        lines.append(
            f"| {r['reranker']} | {r['parameters'] / 1e6:.0f}M | {r['recall_at_5']:.3f} "
            f"| {r['mrr']:.3f} | {r['context_precision_at_5']:.3f} "
            f"| {first['p50_ms']:.1f} | {first['p95_ms']:.1f} "
            f"| {r['p95_rerun_change']:+.1%} | {r['hardware']} |"
        )
    return "\n".join(lines) + "\n"


def step_evaluate(settings: Settings) -> None:
    d, device = settings.distillation, settings.device
    student = {"max_length": d.student.max_length, "batch_size": d.student.batch_size}
    candidates = [
        (
            "teacher (bge-reranker-v2-m3)",
            lambda: CrossEncoderReranker(settings.reranker, device),
        ),
        ("student, untrained", lambda: CrossEncoderReranker(d.student, device)),
        (
            "student, teacher's scores",
            lambda: LocalCrossEncoder(MODELS / "soft", device=device, **student),
        ),
        (
            "student, hard labels (control)",
            lambda: LocalCrossEncoder(MODELS / "hard", device=device, **student),
        ),
    ]
    rows = [measure(settings, name, load()) for name, load in candidates]
    metrics = Path(settings.paths.metrics_dir) / "distillation.json"
    metrics.write_text(json.dumps(rows, indent=2) + "\n", "utf-8")
    out = Path(settings.paths.reports_dir) / "distillation.md"
    out.write_text(report(rows), "utf-8")
    logger.info("Wrote %s and %s", metrics, out)


STEPS = {
    "questions": step_questions,
    "label": step_label,
    "train": step_train,
    "evaluate": step_evaluate,
}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("step", choices=STEPS)
    STEPS[ap.parse_args().step](Settings())


if __name__ == "__main__":
    main()
