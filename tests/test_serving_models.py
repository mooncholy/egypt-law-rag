import math

import httpx
import pytest
from fastapi.testclient import TestClient

from raglaw.config import Settings
from raglaw.records import read_records
from raglaw.retrieval.remote import (
    ModelsServiceError,
    RemoteEmbeddings,
    RemoteReranker,
    models_health,
)
from raglaw.schema import Chunk
from raglaw.serving.models_app import create_models_app

# --- The service and its clients (stub models) --------------------------------


@pytest.mark.unit
def test_the_service_reports_the_pinned_models(models_client):
    settings = Settings(_env_file=None)

    health = models_health("http://testserver", client=models_client)

    assert health == {
        "status": "healthy",
        "embedding": f"{settings.embedding.model}@{settings.embedding.revision}",
        "reranker": f"{settings.reranker.model}@{settings.reranker.revision}",
        "device": settings.device,
    }


@pytest.mark.unit
def test_remote_embeddings_embed_questions_and_documents_apart(models_client):
    embeddings = RemoteEmbeddings("http://testserver", client=models_client)

    assert embeddings.embed_query("query") == [0.0, 1.0]
    assert embeddings.embed_documents(["query", "chunk"]) == [[0.0, 1.0], [1.0, 0.0]]


@pytest.mark.unit
def test_the_remote_reranker_scores_each_text_in_order(models_client):
    reranker = RemoteReranker("http://testserver", client=models_client)

    assert reranker.score("q", ["other", "relevant"]) == [0.1, 0.9]


@pytest.mark.unit
@pytest.mark.parametrize(
    ("path", "body"),
    [
        ("/embed", {"texts": [], "kind": "query"}),
        ("/embed", {"texts": ["x"], "kind": "passage"}),
        ("/rerank", {"query": "", "texts": ["x"]}),
        ("/rerank", {"query": "q", "texts": ["x"] * 257}),
    ],
)
def test_the_service_rejects_malformed_requests(models_client, path, body):
    assert models_client.post(path, json=body).status_code == 422


@pytest.mark.unit
def test_a_service_that_isnt_running_is_a_models_service_error():
    embeddings = RemoteEmbeddings("http://127.0.0.1:9")  # discard port: refused

    with pytest.raises(ModelsServiceError, match="/embed"):
        embeddings.embed_query("q")
    with pytest.raises(ModelsServiceError, match="not answering"):
        models_health("http://127.0.0.1:9")


@pytest.mark.unit
def test_an_error_status_is_a_models_service_error():
    failing = httpx.Client(
        base_url="http://models",
        transport=httpx.MockTransport(lambda request: httpx.Response(500)),
    )

    with pytest.raises(ModelsServiceError, match="500"):
        RemoteReranker("http://models", client=failing).score("q", ["x"])


# --- U34: the service embeds as `embed` did (the real model) ---------------------------


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


@pytest.mark.corpus
def test_u34_the_service_embeds_20_chunks_as_the_model_does_in_process():
    from raglaw.retrieval.document_text import document_text
    from raglaw.retrieval.embeddings import huggingface_embeddings

    settings = Settings()
    chunks = read_records(settings.paths.corpus_dir / "chunks.json", Chunk)[::57][:20]
    texts = [
        document_text(
            c, settings.retrieval.document_text, settings.retrieval.repealed_text
        )
        for c in chunks
    ]
    model = huggingface_embeddings(settings.embedding, settings.device)
    app = create_models_app(settings, embeddings=model, reranker=None)
    app.state.reranker = object()  # this test needs the embedder only

    with TestClient(app) as client:
        served = RemoteEmbeddings("http://testserver", client=client)
        remote = served.embed_documents(texts)
        query = served.embed_query(texts[0])

    local = model.embed_documents(texts)
    assert len(remote) == 20
    assert min(_cosine(r, l) for r, l in zip(remote, local, strict=True)) > 1 - 1e-5
    assert _cosine(query, model.embed_query(texts[0])) > 1 - 1e-5
