"""The ``bm25`` stage: chunks become the lexical half of the search index.

Each chunk's document text (``retrieval.document_text``) is split into terms by
``retrieval.bm25_tokenizer`` and BM25-indexed (``raglaw.retrieval.lexical``). It
takes seconds, so comparing tokenizers never waits on the ``embed`` stage.

A chunk that yields no terms can never be found by BM25; each one is logged as
an anomaly and counted, not dropped.

Run as ``python -m raglaw.ingest.bm25``.
"""

import argparse
import json
import logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from raglaw.config import Embedding, Retrieval, Settings
from raglaw.logging_setup import log_anomaly
from raglaw.records import read_records
from raglaw.retrieval.document_text import document_text
from raglaw.retrieval.embeddings import ModelTokenizer, model_tokenizer
from raglaw.retrieval.lexical import build_bm25
from raglaw.retrieval.tokenize import bm25_tokenizer
from raglaw.schema import Chunk, LogEvent
from raglaw.tracking import sha256_file, stage_run

logger = logging.getLogger(__name__)


def _count_terms(
    chunks: list[Chunk], retrieval: Retrieval, tokenize: Callable[[str], list[str]]
) -> tuple[int, int]:
    """The most terms in one chunk, and how many chunks have none (each logged)."""
    counts = [len(tokenize(document_text(c, retrieval.document_text))) for c in chunks]
    for c, n in zip(chunks, counts, strict=True):
        if n == 0:
            log_anomaly(
                logger,
                "Chunk yields no BM25 terms",
                page=c.source_pages[0],
                row_index=-1,
                row_class="article",
                article_number=c.article_number,
                anomaly_type="chunk_without_terms",
                chunk_id=c.chunk_id,
                bm25_tokenizer=retrieval.bm25_tokenizer,
            )
    return max(counts, default=0), counts.count(0)


def run_bm25(
    chunks_in: Path,
    bm25_dir: Path,
    metrics_out: Path,
    *,
    embedding: Embedding,
    retrieval: Retrieval,
    tokenizer: ModelTokenizer | None = None,
) -> dict[str, Any]:
    """
    BM25-index ``chunks_in`` into ``bm25_dir``, with metrics.

    ``tokenizer`` is the model's own, needed only for ``model_subwords``.

    returns:
    - metrics (dict[str, Any]): what was written to ``metrics_out``
    """
    chunks = read_records(chunks_in, Chunk)
    tokenize = bm25_tokenizer(retrieval.bm25_tokenizer, tokenizer)
    max_terms, without_terms = _count_terms(chunks, retrieval, tokenize)
    started = time.perf_counter()
    manifest = build_bm25(
        chunks,
        bm25_dir,
        tokenize=tokenize,
        retrieval=retrieval,
        embedding=embedding,
        chunks_sha256=sha256_file(chunks_in),
    )
    seconds = time.perf_counter() - started
    metrics: dict[str, Any] = {
        "chunks_indexed": len(manifest.chunk_ids),
        "vocabulary_size": manifest.vocabulary_size,
        "max_chunk_terms": max_terms,
        "chunks_without_terms": without_terms,
        "seconds": round(seconds, 1),
        "document_text": retrieval.document_text,
        "bm25_tokenizer": retrieval.bm25_tokenizer,
    }
    metrics_out.parent.mkdir(parents=True, exist_ok=True)
    metrics_out.write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    logger.info(
        "BM25-indexed %d chunks, %d terms",
        metrics["chunks_indexed"],
        manifest.vocabulary_size,
        extra={"event_type": LogEvent.STAGE_COMPLETED, **metrics},
    )
    return metrics


def run_params(retrieval: Retrieval) -> dict[str, object]:
    """
    The BM25 config a run is compared by, under the names used across runs.

    returns:
    - params (dict[str, object]): the document text variant and the tokenizer
    """
    return {
        "document_text": retrieval.document_text,
        "bm25_tokenizer": retrieval.bm25_tokenizer,
    }


def main(argv: list[str] | None = None) -> None:
    settings = Settings()
    paths = settings.paths
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--chunks", type=Path, default=paths.corpus_dir / "chunks.json")
    ap.add_argument("--out", type=Path, default=paths.index_dir / "bm25")
    ap.add_argument("--metrics", type=Path, default=paths.metrics_dir / "bm25.json")
    args = ap.parse_args(argv)
    retrieval = settings.retrieval
    tokenizer = (
        model_tokenizer(settings.embedding)
        if retrieval.bm25_tokenizer == "model_subwords"
        else None
    )
    with stage_run(
        "bm25", input_hash=sha256_file(args.chunks), settings=settings
    ) as run:
        run.log_params(run_params(retrieval))
        metrics = run_bm25(
            args.chunks,
            args.out,
            args.metrics,
            embedding=settings.embedding,
            retrieval=retrieval,
            tokenizer=tokenizer,
        )
        run.log_metrics(
            {k: v for k, v in metrics.items() if isinstance(v, int | float)}
        )


if __name__ == "__main__":
    main()
