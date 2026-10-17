"""The ``embed`` stage: chunks become the dense half of the search index.

Each chunk's document text (``retrieval.document_text``, D14) is embedded with
the pinned model into a Qdrant collection (``raglaw.retrieval.dense``). The BM25
half is its own stage (``bm25``), so a tokenizer change never re-embeds.

Before embedding, every document text is counted with the model's own tokenizer.
A text longer than the model's input would be truncated silently, so one such
chunk stops the stage (C17). After building, the collection must hold exactly
one point per chunk.

Run as ``python -m raglaw.ingest.embed``.
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

from langchain_core.embeddings import Embeddings

from raglaw.config import Embedding, Retrieval, Settings
from raglaw.logging_setup import log_anomaly
from raglaw.records import read_records
from raglaw.retrieval.dense import build_dense, count_points
from raglaw.retrieval.document_text import document_text
from raglaw.retrieval.embeddings import (
    ModelTokenizer,
    huggingface_embeddings,
    model_tokenizer,
)
from raglaw.schema import Chunk, LogEvent
from raglaw.tracking import sha256_file, stage_run

logger = logging.getLogger(__name__)


class EmbedError(RuntimeError):
    """A chunk would be truncated, or the collection doesn't hold every chunk once."""


def _check_lengths(
    chunks: list[Chunk], texts: list[str], tokenizer: ModelTokenizer
) -> list[int]:
    """Count each text's tokens; log and fail on any the model would truncate."""
    counts = [len(tokenizer(t)["input_ids"]) for t in texts]
    limit = tokenizer.model_max_length
    over = [(c, n) for c, n in zip(chunks, counts, strict=True) if n > limit]
    for c, n in over:
        log_anomaly(
            logger,
            "Document text longer than the model's input",
            page=c.source_pages[0],
            row_index=-1,
            row_class="article",
            article_number=c.article_number,
            anomaly_type="truncated_chunk",
            chunk_id=c.chunk_id,
            tokens=n,
            max_tokens=limit,
        )
    if over:
        raise EmbedError(
            f"{len(over)} chunks are longer than {limit} tokens; the model would "
            "truncate them"
        )
    return counts


def run_embed(
    chunks_in: Path,
    dense_dir: Path,
    metrics_out: Path,
    *,
    embedding: Embedding,
    retrieval: Retrieval,
    embeddings: Embeddings,
    tokenizer: ModelTokenizer,
    device: str = "cpu",
) -> dict[str, Any]:
    """
    Embed ``chunks_in`` into ``dense_dir``, with metrics.

    ``device`` is where ``embeddings`` runs, recorded with the metrics and in
    the manifest.

    returns:
    - metrics (dict[str, Any]): what was written to ``metrics_out``

    exceptions:
    - EmbedError: a chunk would be truncated (nothing is built), or the
      collection doesn't hold one point per chunk
    """
    chunks = read_records(chunks_in, Chunk)
    texts = [
        document_text(c, retrieval.document_text, retrieval.repealed_text)
        for c in chunks
    ]
    counts = _check_lengths(chunks, texts, tokenizer)
    started = time.perf_counter()
    manifest = build_dense(
        chunks,
        dense_dir,
        embeddings=embeddings,
        embedding=embedding,
        document_text=retrieval.document_text,
        repealed_text=retrieval.repealed_text,
        chunks_sha256=sha256_file(chunks_in),
        device=device,
    )
    seconds = time.perf_counter() - started
    indexed = count_points(dense_dir)
    if indexed != len(chunks):
        raise EmbedError(f"{indexed} points indexed for {len(chunks)} chunks")
    metrics: dict[str, Any] = {
        "chunks_indexed": indexed,
        "embedding_dim": manifest.embedding_dim,
        "max_chunk_tokens": max(counts),
        "max_tokens": tokenizer.model_max_length,
        "truncated_chunks": 0,
        "device": device,
        "seconds": round(seconds, 1),
        "document_text": retrieval.document_text,
        "repealed_text": retrieval.repealed_text,
    }
    metrics_out.parent.mkdir(parents=True, exist_ok=True)
    metrics_out.write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    logger.info(
        "Embedded %d chunks in %.1f s",
        indexed,
        seconds,
        extra={"event_type": LogEvent.STAGE_COMPLETED, **metrics},
    )
    return metrics


def run_params(
    embedding: Embedding, retrieval: Retrieval, device: str
) -> dict[str, object]:
    """
    The embedding config a run is compared by, under the names used across runs.

    returns:
    - params (dict[str, object]): the model, its revision, the document text
      choices and the device
    """
    return {
        "embedding_model": embedding.model,
        "embedding_revision": embedding.revision,
        "document_text": retrieval.document_text,
        "repealed_text": retrieval.repealed_text,
        "device": device,
    }


def main(argv: list[str] | None = None) -> None:
    settings = Settings()
    paths = settings.paths
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--chunks", type=Path, default=paths.corpus_dir / "chunks.json")
    ap.add_argument("--out", type=Path, default=paths.index_dir / "dense")
    ap.add_argument("--metrics", type=Path, default=paths.metrics_dir / "embed.json")
    args = ap.parse_args(argv)
    with stage_run(
        "embed", input_hash=sha256_file(args.chunks), settings=settings
    ) as run:
        run.log_params(
            run_params(settings.embedding, settings.retrieval, settings.device)
        )
        metrics = run_embed(
            args.chunks,
            args.out,
            args.metrics,
            embedding=settings.embedding,
            retrieval=settings.retrieval,
            embeddings=huggingface_embeddings(settings.embedding, settings.device),
            tokenizer=model_tokenizer(settings.embedding),
            device=settings.device,
        )
        run.log_metrics(
            {k: v for k, v in metrics.items() if isinstance(v, int | float)}
        )


if __name__ == "__main__":
    main()
