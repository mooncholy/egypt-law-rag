import asyncio

import pytest

from raglaw.rag import Answer

bento = pytest.importorskip("raglaw.serving.bento_service")

pytestmark = pytest.mark.unit


class AsyncAPIs:
    async def embed(self, texts):
        return [[float(len(t))] for t in texts]

    async def rerank(self, pairs):
        return [1.0 if q in t else 0.0 for q, t in pairs]


class StubModels:
    """``Models`` as ``bentoml.depends`` hands it: its APIs under ``to_async``."""

    to_async = AsyncAPIs()


def test_the_embedder_and_reranker_are_batchable_apis():
    """Phase 12: adaptive batching on both models, nothing batched in /ask."""
    apis = bento.Models.apis

    assert apis["embed"].batchable and apis["rerank"].batchable
    assert not bento.RagLaw.apis["ask"].batchable
    assert set(bento.RagLaw.dependencies) == {"models"}


def test_the_clients_work_from_the_retrievers_worker_thread():
    """The pipeline retrieves in ``asyncio.to_thread``; the clients hand each
    call to the service's loop, where BentoML's async API runs."""

    async def ask():
        loop = asyncio.get_running_loop()
        embeddings = bento.BentoEmbeddings(StubModels(), loop)
        reranker = bento.BentoReranker(StubModels(), loop)

        def retrieve():
            return (
                embeddings.embed_query("abc"),
                embeddings.embed_documents(["a", "bb"]),
                reranker.score("lease", ["a lease", "a sale"]),
            )

        return await asyncio.to_thread(retrieve)

    query, documents, scores = asyncio.run(ask())

    assert query == [3.0]
    assert documents == [[1.0], [2.0]]
    assert scores == [1.0, 0.0]


def test_bentos_ask_redacts_like_the_fastapi_app():
    """The PII guardrail covers every /ask, the BentoML form included."""
    answer = Answer(
        text="The holder of ID 29801011234567 is of age under Article 44.",
        sources=["Egyptian Civil Code, Article 44"],
        language="en",
        unaddressed=False,
        articles=[44],
        rank1_rerank_score=0.9,
    )

    reply = bento.respond(answer)

    assert reply == {
        "answer": "The holder of ID [national_id] is of age under Article 44.",
        "sources": ["Egyptian Civil Code, Article 44"],
    }
