"""The cross-encoder reranker for ``search.rerank``.

A bi-encoder (the embedder) compares a question's vector with each chunk's
vector, computed apart. A cross-encoder reads the question and one chunk
together and scores how well the chunk answers it: more accurate, but too slow
to run over the whole corpus, so it re-scores only the top fused chunks.
"""

from collections.abc import Sequence
from typing import Protocol

from raglaw.config import Reranker
from raglaw.retrieval.embeddings import check_device


class RerankScorer(Protocol):
    """Anything that scores (question, chunk text) pairs; higher is better."""

    def score(self, query: str, texts: Sequence[str]) -> list[float]: ...


class CrossEncoderReranker:
    """The pinned cross-encoder (``reranker`` in ``params.yaml``), on ``device``."""

    def __init__(self, config: Reranker, device: str = "cpu") -> None:
        from sentence_transformers import CrossEncoder  # embed group only

        check_device(device)
        self.model = CrossEncoder(config.model, revision=config.revision, device=device)
        self.model.max_seq_length = config.max_length
        self.batch_size = config.batch_size

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        """
        Score each chunk text against the question.

        returns:
        - scores (list[float]): the model's relevance, in 0 to 1 (its sigmoid)
        """
        scores = self.model.predict(
            [(query, t) for t in texts],
            batch_size=self.batch_size,
            show_progress_bar=False,
        )
        return [float(s) for s in scores]
