"""Download the pinned embedder and reranker into the Hugging Face cache, then exit.

Compose's ``models-pull`` runs this once before ``models`` starts, so a slow
first download never runs into the service's healthcheck. It loads both models
on CPU through the same functions the service uses (``huggingface_embeddings``,
``CrossEncoderReranker``), so it fetches exactly the files the service will
read, at the revisions ``params.yaml`` pins. A later run finds them cached and
exits in seconds.

Run as ``python -m raglaw.serving.fetch_models``; it needs the ``embed``
dependency group.
"""

import logging

from raglaw.config import Settings
from raglaw.retrieval.embeddings import huggingface_embeddings
from raglaw.retrieval.rerank import CrossEncoderReranker

logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    settings = Settings()
    for name, revision in (
        (settings.embedding.model, settings.embedding.revision),
        (settings.reranker.model, settings.reranker.revision),
    ):
        logger.info("Fetching %s at %s", name, revision)
    huggingface_embeddings(settings.embedding, "cpu")
    CrossEncoderReranker(settings.reranker, "cpu")
    logger.info("Both models are in the cache")


if __name__ == "__main__":
    main()
