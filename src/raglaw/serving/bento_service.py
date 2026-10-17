"""The pipeline as two BentoML services (Phase 12): ``RagLaw`` (async /ask) over ``Models``.

``Models`` serves the embedder and the reranker as batchable APIs: BentoML's
adaptive batching merges calls that arrive together into one model batch,
waiting at most ``max_latency_ms`` and capped at ``max_batch_size``, so
concurrent questions share the GPU's batches instead of queueing one by one.
``RagLaw`` loads the baked retriever (D26, ``RAGLAW_CHAMPION_DIR``) with
clients that call ``Models`` through ``bentoml.depends``, and answers with the
same ``AnswerPipeline`` as the FastAPI app, so both serve identical answers.

The pipeline retrieves in a worker thread, where BentoML's synchronous proxy
can't run (it needs an AnyIO thread), and the proxy isn't ready during
``__init__``. So the clients submit the proxy's async API to the service's
event loop from whatever thread they're in, and the champion loads on the
first ``/ask``.
The LLM is the OpenAI-compatible endpoint in ``RAGLAW_LLM_*`` (vLLM, D20).

Serve both with ``bentoml serve raglaw.serving.bento_service:RagLaw`` (the
``bento`` and ``embed`` groups); ``POST /ask`` takes ``{"question": "..."}``
on port 3000. The FastAPI app (``raglaw.api.main``) stays the image compose
runs; this is the BentoML form of the same service.
"""

import asyncio
from collections.abc import Coroutine, Sequence
from typing import Any

import bentoml
from langchain_core.embeddings import Embeddings

from raglaw.api.schemas import AskRequest
from raglaw.config import Settings
from raglaw.llm import OpenAIChat
from raglaw.monitoring.pii import redact
from raglaw.rag import Answer, AnswerPipeline


def respond(answer: Answer) -> dict[str, Any]:
    """``/ask``'s reply, through the same PII guardrail as the FastAPI app."""
    return {"answer": redact(answer.text).text, "sources": answer.sources}


class _OnLoop:
    """Calls ``Models``' async APIs from a worker thread, on the service's loop."""

    def __init__(self, models: Any, loop: asyncio.AbstractEventLoop) -> None:
        self.models = models
        self.loop = loop

    def _call(self, call: Coroutine[Any, Any, Any]) -> Any:
        return asyncio.run_coroutine_threadsafe(call, self.loop).result()


class BentoEmbeddings(_OnLoop, Embeddings):
    """The embedder, through ``Models.embed`` (a batchable API)."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._call(self.models.to_async.embed(texts))

    def embed_query(self, text: str) -> list[float]:
        # bge-m3 encodes a query exactly as a document (no query prompt).
        return self.embed_documents([text])[0]


class BentoReranker(_OnLoop):
    """The reranker, through ``Models.rerank`` (a batchable API over pairs)."""

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        return self._call(self.models.to_async.rerank([[query, t] for t in texts]))


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
        self.settings = Settings()
        if not (
            self.settings.llm_base_url
            and self.settings.llm_model
            and self.settings.llm_api_key
        ):
            raise RuntimeError("set RAGLAW_LLM_BASE_URL, _MODEL and _API_KEY")
        self.pipeline: AnswerPipeline | None = None
        self._loading = asyncio.Lock()

    async def _pipeline(self) -> AnswerPipeline:
        """The pipeline, built on the first call: loading the champion embeds a
        probe text, which needs ``Models``, and its proxy isn't ready in
        ``__init__``."""
        async with self._loading:
            if self.pipeline is None:
                from raglaw.retrieval.champion import load_champion

                loop = asyncio.get_running_loop()
                champion = await asyncio.to_thread(
                    load_champion,
                    self.settings,
                    embeddings=BentoEmbeddings(self.models, loop),
                    reranker=BentoReranker(self.models, loop),
                )
                llm = OpenAIChat(
                    str(self.settings.llm_base_url),
                    self.settings.llm_api_key.get_secret_value(),
                    self.settings.llm_model,
                )
                self.pipeline = AnswerPipeline(
                    champion.retrieve, llm, self.settings.answer
                )
        return self.pipeline

    @bentoml.api
    async def ask(self, question: str) -> dict[str, Any]:
        """The answer and the articles it cites, as the FastAPI /ask returns them.
        The request body is ``{"question": "..."}``."""
        asked = AskRequest(question=question)
        pipeline = await self._pipeline()
        return respond(await pipeline.answer(asked.question))
