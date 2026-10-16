import pytest

bento = pytest.importorskip("raglaw.serving.bento_service")

pytestmark = pytest.mark.unit


class StubModels:
    """``Models`` as ``bentoml.depends`` hands it: its APIs as plain calls."""

    def embed(self, texts):
        return [[float(len(t))] for t in texts]

    def rerank(self, pairs):
        return [1.0 if q in t else 0.0 for q, t in pairs]


def test_the_embedder_and_reranker_are_batchable_apis():
    """Phase 12: adaptive batching on both models, nothing batched in /ask."""
    apis = bento.Models.apis

    assert apis["embed"].batchable and apis["rerank"].batchable
    assert not bento.RagLaw.apis["ask"].batchable
    assert set(bento.RagLaw.dependencies) == {"models"}


def test_the_clients_fit_where_the_retriever_expects_its_models():
    embeddings = bento.BentoEmbeddings(StubModels())
    reranker = bento.BentoReranker(StubModels())

    assert embeddings.embed_query("abc") == [3.0]
    assert embeddings.embed_documents(["a", "bb"]) == [[1.0], [2.0]]
    assert reranker.score("lease", ["a lease", "a sale"]) == [1.0, 0.0]
