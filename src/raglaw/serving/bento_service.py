"""The pipeline as two BentoML services (Phase 12): ``RagLaw`` (async /ask) over ``Models``.

``Models`` serves the embedder and the reranker as batchable APIs: BentoML's
adaptive batching merges calls that arrive together into one model batch,
waiting at most ``max_latency_ms`` and capped at ``max_batch_size``, so
concurrent questions share the GPU's batches instead of queueing one by one.
``RagLaw`` loads the baked retriever (D26, ``RAGLAW_CHAMPION_DIR``) with
clients that call ``Models`` through ``bentoml.depends``, and answers with the
same ``AnswerPipeline`` as the FastAPI app, so both serve identical answers.
The LLM is the OpenAI-compatible endpoint in ``RAGLAW_LLM_*`` (vLLM, D20).

Serve both with ``bentoml serve raglaw.serving.bento_service:RagLaw`` (the
``bento`` and ``embed`` groups); ``POST /ask`` takes ``{"question": "..."}``
on port 3000. The FastAPI app (``raglaw.api.main``) stays the image compose
runs; this is the BentoML form of the same service.
"""

from collections.abc import Sequence
from typing import Any

import bentoml
from langchain_core.embeddings import Embeddings

from raglaw.api.schemas import AskRequest
from raglaw.config import Settings
from raglaw.llm import OpenAIChat
from raglaw.rag import AnswerPipeline


class BentoEmbeddings(Embeddings):
    """The embedder, through ``Models.embed`` (a batchable API)."""

    def __init__(self, models: Any) -> None:
        self.models = models

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.models.embed(texts)

    def embed_query(self, text: str) -> list[float]:
        # bge-m3 encodes a query exactly as a document (no query prompt).
        return self.models.embed([text])[0]


class BentoReranker:
    """The reranker, through ``Models.rerank`` (a batchable API over pairs)."""

    def __init__(self, models: Any) -> None:
        self.models = models

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        return self.models.rerank([[query, t] for t in texts])


@bentoml.service(traffic={"timeout": 60})
class Models:
    """The embedder and the reranker, on ``RAGLAW_DEVICE``."""

    def __init__(self) -> None:
        from raglaw.retrieval.embeddings import huggingface_embeddings
        from raglaw.retrieval.rerank import CrossEncoderReranker

        settings = Settings()
        self.embeddings = huggingface_embeddings(settings.embedding, settings.device)
        self.reranker = CrossEncoderReranker(settings.reranker, settings.device)

    @bentoml.api(batchable=True, max_batch_size=64, max_latency_ms=50)
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Each text's normalized vector, in order."""
        return self.embeddings.embed_documents(texts)

    @bentoml.api(batchable=True, max_batch_size=128, max_latency_ms=50)
    def rerank(self, pairs: list[list[str]]) -> list[float]:
        """Each (question, chunk text) pair's relevance in 0 to 1, in order."""
        return self.reranker.score_pairs([(q, t) for q, t in pairs])


@bentoml.service(traffic={"timeout": 300})
class RagLaw:
    """/ask: retrieve with the baked champion, then answer with the LLM."""

    models = bentoml.depends(Models)

    def __init__(self) -> None:
        from raglaw.retrieval.champion import load_champion

        settings = Settings()
        if not (settings.llm_base_url and settings.llm_model and settings.llm_api_key):
            raise RuntimeError("set RAGLAW_LLM_BASE_URL, _MODEL and _API_KEY")
        self.champion = load_champion(
            settings,
            embeddings=BentoEmbeddings(self.models),
            reranker=BentoReranker(self.models),
        )
        llm = OpenAIChat(
            str(settings.llm_base_url),
            settings.llm_api_key.get_secret_value(),
            settings.llm_model,
        )
        self.pipeline = AnswerPipeline(self.champion.retrieve, llm, settings.answer)

    @bentoml.api
    async def ask(self, question: str) -> dict[str, Any]:
        """The answer and the articles it cites, as the FastAPI /ask returns them."""
        asked = AskRequest(question=question)
        answer = await self.pipeline.answer(asked.question)
        return {"answer": answer.text, "sources": answer.sources}
