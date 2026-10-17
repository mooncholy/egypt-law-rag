"""The ``models`` service (D18): the pinned embedder and reranker over HTTP.

The API process has no ``torch``; this one holds both models, loaded once at
startup with the implementation ``embed`` indexed with
(``huggingface_embeddings``, ``CrossEncoderReranker``), so a question embeds
exactly as the documents did. It runs on ``settings.device``, GPU or CPU.

Run it with ``uv run uvicorn raglaw.serving.models_app:app --port 8001``; it
needs the ``embed`` dependency group.
"""

from contextlib import asynccontextmanager
from typing import Annotated, Literal

from fastapi import FastAPI, Request
from langchain_core.embeddings import Embeddings
from pydantic import BaseModel, Field, StringConstraints

from raglaw.config import Settings
from raglaw.logging_conf import configure_server_logging, get_logger
from raglaw.retrieval.rerank import RerankScorer
from raglaw.schema import LogEvent

configure_server_logging()
logger = get_logger(__name__)

# Above `search.rerank_depth` and a chunking batch, below a body big enough to
# stall the service.
MAX_TEXTS = 256
Text = Annotated[str, StringConstraints(min_length=1)]


class EmbedRequest(BaseModel):
    """Texts to embed: chunk texts (``documents``) or one question (``query``)."""

    texts: list[Text] = Field(min_length=1, max_length=MAX_TEXTS)
    kind: Literal["documents", "query"] = Field(
        description="`query` embeds each text as a question, `documents` as an "
        "indexed chunk; bge-m3 treats both alike, the contract keeps them apart."
    )


class EmbedResponse(BaseModel):
    vectors: list[list[float]] = Field(description="One normalized vector per text.")


class RerankRequest(BaseModel):
    """A question and the chunk texts to score against it."""

    query: Text
    texts: list[Text] = Field(min_length=1, max_length=MAX_TEXTS)


class RerankResponse(BaseModel):
    scores: list[float] = Field(description="Relevance per text, 0 to 1.")


class ModelsHealth(BaseModel):
    """What the service serves, so the API can tell it's the pinned pair."""

    status: Literal["healthy"]
    embedding: str = Field(description="`<model>@<revision>`.")
    reranker: str = Field(description="`<model>@<revision>`.")
    device: str


def _load(settings: Settings) -> tuple[Embeddings, RerankScorer]:
    from raglaw.retrieval.embeddings import huggingface_embeddings
    from raglaw.retrieval.rerank import CrossEncoderReranker

    return (
        huggingface_embeddings(settings.embedding, settings.device),
        CrossEncoderReranker(settings.reranker, settings.device),
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load both models once, unless the factory was handed them (tests)."""
    if app.state.embeddings is None or app.state.reranker is None:
        app.state.embeddings, app.state.reranker = _load(app.state.settings)
    settings = app.state.settings
    logger.info(
        "Models loaded",
        extra={
            "event_type": LogEvent.STARTUP,
            "embedding": settings.embedding.model,
            "reranker": settings.reranker.model,
            "device": settings.device,
        },
    )
    yield


def embed(request: Request, body: EmbedRequest) -> EmbedResponse:
    """
    Embed each text with the pinned embedder.

    returns:
    - vectors (EmbedResponse): one per text, in order
    """
    embeddings: Embeddings = request.app.state.embeddings
    if body.kind == "query":
        return EmbedResponse(vectors=[embeddings.embed_query(t) for t in body.texts])
    return EmbedResponse(vectors=embeddings.embed_documents(body.texts))


def rerank(request: Request, body: RerankRequest) -> RerankResponse:
    """
    Score each text against the question with the pinned cross-encoder.

    returns:
    - scores (RerankResponse): one per text, in order
    """
    reranker: RerankScorer = request.app.state.reranker
    return RerankResponse(scores=reranker.score(body.query, body.texts))


def health(request: Request) -> ModelsHealth:
    """
    Report the models served; only answers once both are loaded.

    returns:
    - health (ModelsHealth): each model with its pinned revision, and the device
    """
    settings: Settings = request.app.state.settings
    return ModelsHealth(
        status="healthy",
        embedding=f"{settings.embedding.model}@{settings.embedding.revision}",
        reranker=f"{settings.reranker.model}@{settings.reranker.revision}",
        device=settings.device,
    )


def create_models_app(
    settings: Settings | None = None,
    *,
    embeddings: Embeddings | None = None,
    reranker: RerankScorer | None = None,
) -> FastAPI:
    """
    Build the service; the models load when its lifespan starts.

    Plain ``def`` handlers: FastAPI runs them in its thread pool, so a model
    call never blocks the event loop.

    returns:
    - app (FastAPI): serving ``embeddings`` and ``reranker`` when given
      (tests), else the pinned models on ``settings.device``
    """
    app = FastAPI(
        title="Egyptian Civil Code RAG: models",
        description="The pinned embedder and reranker, for the API (D18).",
        lifespan=lifespan,
    )
    app.state.settings = settings or Settings()
    app.state.embeddings, app.state.reranker = embeddings, reranker
    app.post("/embed")(embed)
    app.post("/rerank")(rerank)
    app.get("/health")(health)
    return app


app = create_models_app()
