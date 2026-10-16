"""Add new articles to the index in batch (Phase 12): chunk, append, rebuild both halves.

Reads a JSON list of articles (the ``Article`` fields; see
``scripts/examples/test_amendment.json``, a test document that is not part of the
Code), appends their chunks to ``data/corpus/chunks.json``'s, and rebuilds the
dense and BM25 index into ``--out``, leaving the tracked index untouched. Prints
``documents_indexed`` before and after (``raglaw.ingest.reindex``).

Run as ``uv run python scripts/reindex.py scripts/examples/test_amendment.json
--out build/reindexed`` (add the GPU group on a GPU machine).
"""

import argparse
import json
import logging
from pathlib import Path

from raglaw.config import Settings
from raglaw.ingest.reindex import reindex
from raglaw.retrieval.embeddings import huggingface_embeddings
from raglaw.schema import Article

logger = logging.getLogger("reindex")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    settings = Settings()
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("articles", type=Path, help="a JSON list of new articles")
    ap.add_argument("--out", type=Path, default=Path("build/reindexed"))
    args = ap.parse_args()
    articles = [
        Article.model_validate(a) for a in json.loads(args.articles.read_text("utf-8"))
    ]
    result = reindex(
        Path(settings.paths.corpus_dir) / "chunks.json",
        articles,
        args.out,
        chunking=settings.chunking,
        embedding=settings.embedding,
        retrieval=settings.retrieval,
        embeddings=huggingface_embeddings(settings.embedding, settings.device),
        device=settings.device,
    )
    logger.info(
        "documents_indexed: %d -> %d (articles %s added) in %s",
        result.documents_before,
        result.documents_after,
        result.added_articles,
        args.out,
    )


if __name__ == "__main__":
    main()
