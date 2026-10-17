"""Batch re-indexing (Phase 12): add new articles to the index without the pipeline.

When an amendment adds articles, the corpus pipeline needs the source PDF
again. This adds them directly: it chunks the new articles the way ``chunk``
does, appends them to the existing chunks, and rebuilds both index halves
(``embed`` and ``bm25``; BM25 has no incremental add) into a new directory,
with manifests that match the combined chunks. The tracked index is never
touched; serving a re-indexed directory means pointing ``paths.index_dir`` at
it and exporting a retriever registered on it.

Run with ``scripts/reindex.py``.
"""

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from langchain_core.embeddings import Embeddings

from raglaw.config import Chunking, Embedding, Retrieval
from raglaw.ingest.chunk import chunk_documents
from raglaw.ingest.loader import CivilCodeArticleLoader
from raglaw.records import read_records, write_records
from raglaw.retrieval.dense import build_dense
from raglaw.retrieval.lexical import build_bm25
from raglaw.retrieval.tokenize import bm25_tokenizer
from raglaw.schema import Article, Chunk
from raglaw.tracking import sha256_file

logger = logging.getLogger(__name__)


class ReindexError(ValueError):
    """A new article's number is already indexed, or repeats among the new ones."""


@dataclass(frozen=True)
class Reindexed:
    """What a re-index changed."""

    documents_before: int
    documents_after: int
    added_articles: list[int]


def reindex(
    chunks_path: Path,
    new_articles: Sequence[Article],
    out_dir: Path,
    *,
    chunking: Chunking,
    embedding: Embedding,
    retrieval: Retrieval,
    embeddings: Embeddings,
    device: str | None = None,
) -> Reindexed:
    """
    Chunk ``new_articles``, append them to ``chunks_path``'s chunks, and build
    both index halves over all of them in ``out_dir`` (``dense/``, ``bm25/``,
    with ``chunks.json`` beside them).

    returns:
    - reindexed (Reindexed): the chunk counts before and after, and the articles added

    exceptions:
    - ReindexError: a new article's number is already indexed (a repealed
      range's chunk covers every number in its range), or appears twice among
      ``new_articles``
    """
    chunks = read_records(chunks_path, Chunk)
    taken = {
        n
        for c in chunks
        for n in range(c.article_number, (c.range_end or c.article_number) + 1)
    }
    numbers = [a.article_number for a in new_articles]
    repeated = sorted({n for n in numbers if numbers.count(n) > 1})
    if repeated:
        raise ReindexError(f"articles {repeated} appear more than once")
    clash = sorted(n for n in numbers if n in taken)
    if clash:
        raise ReindexError(f"articles {clash} are already in the index")
    out_dir.mkdir(parents=True, exist_ok=True)
    staged = out_dir / "new_articles.json"
    write_records(staged, Article, list(new_articles))
    # The embedder too: the structural_semantic strategy splits by meaning.
    new_chunks, _ = chunk_documents(
        list(CivilCodeArticleLoader(staged).lazy_load()), chunking, embeddings
    )
    combined = [*chunks, *new_chunks]
    combined_path = out_dir / "chunks.json"
    write_records(combined_path, Chunk, combined)
    sha = sha256_file(combined_path)
    build_dense(
        combined,
        out_dir / "dense",
        embeddings=embeddings,
        embedding=embedding,
        document_text=retrieval.document_text,
        repealed_text=retrieval.repealed_text,
        chunks_sha256=sha,
        device=device,
    )
    build_bm25(
        combined,
        out_dir / "bm25",
        tokenize=bm25_tokenizer(
            retrieval.bm25_tokenizer, stopwords=retrieval.bm25_stopwords
        ),
        retrieval=retrieval,
        embedding=embedding,
        chunks_sha256=sha,
    )
    added = sorted({c.article_number for c in new_chunks})
    logger.info(
        "Re-indexed %d chunks into %s (+%d, articles %s)",
        len(combined),
        out_dir,
        len(new_chunks),
        added,
    )
    return Reindexed(len(chunks), len(combined), added)
