"""The ``models`` service's clients: question embeddings and reranking over HTTP.

The API process has no ``torch`` (D18). It reaches the pinned models through
the ``models`` service (``raglaw.serving.models_app``), which runs the same
implementation ``embed`` indexed with, so a question embeds exactly as the
documents did. These two clients fit where the in-process models did: a
LangChain ``Embeddings`` for ``HybridRetriever``'s dense search, and a
``RerankScorer`` for its reranking.
"""

from collections.abc import Sequence

import httpx
from langchain_core.embeddings import Embeddings

# Long enough for a CPU rerank of `rerank_depth` chunks, short enough that a
# dead service fails a request instead of hanging it.
TIMEOUT_SECONDS = 30.0


class ModelsServiceError(RuntimeError):
    """The ``models`` service couldn't be reached, or refused the request."""


def _post(client: httpx.Client, path: str, payload: dict) -> dict:
    try:
        response = client.post(path, json=payload)
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise ModelsServiceError(f"{path} on {client.base_url} failed: {exc}") from exc
    return response.json()


class RemoteEmbeddings(Embeddings):
    """The embedding model, served by the ``models`` service at ``base_url``."""

    def __init__(self, base_url: str, *, client: httpx.Client | None = None) -> None:
        self.client = client or httpx.Client(base_url=base_url, timeout=TIMEOUT_SECONDS)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        Embed chunk texts as ``embed`` did.

        exceptions:
        - ModelsServiceError: the service is down or answered an error
        """
        return _post(self.client, "/embed", {"texts": texts, "kind": "documents"})[
            "vectors"
        ]

    def embed_query(self, text: str) -> list[float]:
        """
        Embed one question.

        exceptions:
        - ModelsServiceError: the service is down or answered an error
        """
        return _post(self.client, "/embed", {"texts": [text], "kind": "query"})[
            "vectors"
        ][0]


class RemoteReranker:
    """The cross-encoder, served by the ``models`` service at ``base_url``."""

    def __init__(self, base_url: str, *, client: httpx.Client | None = None) -> None:
        self.client = client or httpx.Client(base_url=base_url, timeout=TIMEOUT_SECONDS)

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        """
        Score each chunk text against the question.

        returns:
        - scores (list[float]): the model's relevance, in 0 to 1

        exceptions:
        - ModelsServiceError: the service is down or answered an error
        """
        return _post(self.client, "/rerank", {"query": query, "texts": list(texts)})[
            "scores"
        ]


def models_health(base_url: str, *, client: httpx.Client | None = None) -> dict:
    """
    Ask the ``models`` service what it serves.

    returns:
    - health (dict): its ``/health`` body

    exceptions:
    - ModelsServiceError: the service is down or not ready
    """
    client = client or httpx.Client(base_url=base_url, timeout=5.0)
    try:
        response = client.get("/health")
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise ModelsServiceError(f"{client.base_url} is not answering: {exc}") from exc
    return response.json()
