import json

import pytest
from fastapi.testclient import TestClient
from mlflow.tracking import MlflowClient
from pydantic import SecretStr

from raglaw.api import main
from raglaw.api.main import (
    CORRELATION_ID_HEADER,
    SERVICE_VERSION,
    VERSION_HEADER,
    indexed_chunks,
    llm_component,
    load_service,
    models_component,
    retriever_component,
)
from raglaw.config import Settings
from raglaw.llm import LLMError, OpenAIChat
from raglaw.rag import AnswerPipeline
from raglaw.retrieval.champion import export_champion
from raglaw.retrieval.remote import ModelsServiceError

pytestmark = pytest.mark.unit

QUESTION = {"question": "ما هي شروط الأهلية؟"}
LLM_SETTINGS = {
    "llm_api_key": "test-key",
    "llm_base_url": "http://localhost:8002/v1",
    "llm_model": "Qwen/Qwen2.5-3B-Instruct",
}


# --- /health ------------------------------------------------------------------


def test_health_is_degraded_with_each_components_reason(client):
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "degraded"
    assert body["documents_indexed"] == 0
    assert body["components"]["llm"] == {
        "ready": False,
        "detail": "RAGLAW_LLM_API_KEY, RAGLAW_LLM_BASE_URL, RAGLAW_LLM_MODEL are not set.",
    }
    assert set(body["components"]) == {"llm", "models", "retriever"}


def test_u33_health_is_healthy_with_the_index_size_when_every_component_is_ready(
    ask_client, retrieved, make_chunk
):
    with ask_client(retrieved([make_chunk(1)]), documents_indexed=1150) as client:
        body = client.get("/health").json()

    assert (body["status"], body["documents_indexed"]) == ("healthy", 1150)


# --- /ask ---------------------------------------------------------------------


def test_ask_answers_503_with_each_reason(client):
    response = client.post("/ask", json=QUESTION)

    assert response.status_code == 503
    assert set(response.json()["reasons"]) == {"llm", "models", "retriever"}


def test_u30_ask_returns_the_answer_and_the_retrieved_articles_it_cites(
    ask_client, retrieved, make_chunk
):
    reply = "Article 44 sets majority at twenty-one; Article 999 does not exist."
    documents = retrieved([make_chunk(44), make_chunk(45)])

    with ask_client(documents, reply=reply) as client:
        response = client.post("/ask", json={"question": "What is majority?"})

    assert response.status_code == 200
    assert response.json() == {
        "answer": reply,
        "sources": ["Egyptian Civil Code, Article 44"],
    }


def sse(text: str) -> list[tuple[str | None, dict]]:
    """Parse server-sent events into (event, data) pairs."""
    events = []
    for block in text.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.split("\n"))
        events.append((lines.get("event"), json.loads(lines["data"])))
    return events


def test_u38_a_streamed_answer_arrives_in_pieces_then_its_sources(
    ask_client, retrieved, make_chunk
):
    # Longer than the PII guard's hold-back (6 words), as any real answer is;
    # a shorter one leaves in one piece, at the end.
    reply = (
        "Article 44 sets majority at twenty-one completed Gregorian years, "
        "and a person of age is fully capable."
    )
    documents = retrieved([make_chunk(44), make_chunk(45)])

    with ask_client(documents, reply=reply, streaming=True) as client:
        response = client.post("/ask/stream", json={"question": "What is majority?"})

    events = sse(response.text)
    assert response.headers["content-type"].startswith("text/event-stream")
    tokens = [data["token"] for event, data in events if event is None]
    assert len(tokens) > 1  # more than one piece before the end
    assert "".join(tokens).strip() == reply
    assert events[-1] == ("sources", {"sources": ["Egyptian Civil Code, Article 44"]})


def test_a_reply_made_without_the_llm_streams_as_one_piece(
    ask_client, retrieved, make_chunk
):
    with ask_client(retrieved([make_chunk(1)], rerank_score=0.01), streaming=True) as c:
        events = sse(c.post("/ask/stream", json=QUESTION).text)

    assert len(events) == 2 and events[0][1]["token"].startswith("لم أجد")
    assert events[1] == ("sources", {"sources": []})


def test_streaming_is_refused_with_a_503_when_a_component_isnt_ready(client):
    response = client.post("/ask/stream", json=QUESTION)

    assert response.status_code == 503


def test_the_no_answer_reply_has_empty_sources(ask_client, retrieved, make_chunk):
    with ask_client(retrieved([make_chunk(1)], rerank_score=0.01)) as client:
        body = client.post("/ask", json=QUESTION).json()

    assert body["sources"] == []
    assert body["answer"].startswith("لم أجد")


def test_a_models_service_failure_mid_request_is_a_503_naming_it(
    make_app, make_service, stub_llm, answer_config
):
    def retrieve(question):
        raise ModelsServiceError("/embed on http://localhost:8001 failed")

    pipeline = AnswerPipeline(retrieve, stub_llm(), answer_config)
    with TestClient(make_app(make_service(pipeline))) as client:
        response = client.post("/ask", json=QUESTION)

    assert response.status_code == 503
    assert response.json()["reasons"] == {
        "models": "/embed on http://localhost:8001 failed"
    }


def test_an_llm_failure_is_a_503_naming_it(
    make_app, make_service, retrieved, make_chunk, answer_config
):
    class DownLLM:
        async def complete(self, messages, *, temperature, max_tokens):
            raise LLMError("connection refused")

    pipeline = AnswerPipeline(
        lambda _: retrieved([make_chunk(1)]), DownLLM(), answer_config
    )
    with TestClient(make_app(make_service(pipeline))) as client:
        response = client.post("/ask", json=QUESTION)

    assert response.status_code == 503
    assert response.json()["reasons"] == {"llm": "connection refused"}


@pytest.mark.parametrize("question", ["", "   ", "x" * 2001])
def test_ask_rejects_empty_or_oversized_question(client, question):
    response = client.post("/ask", json={"question": question})

    assert response.status_code == 422
    assert [e["field"] for e in response.json()["errors"]] == ["question"]


def test_ask_rejects_invalid_json_without_field(client):
    response = client.post(
        "/ask", content="{", headers={"Content-Type": "application/json"}
    )

    assert response.status_code == 422
    error = response.json()["errors"][0]
    assert error["field"] is None
    assert error["message"].startswith("Request body is not valid JSON")


def test_correlation_id_is_echoed_or_generated(client):
    echoed = client.get("/health", headers={CORRELATION_ID_HEADER: "abc-123"})
    generated = client.get("/health")

    assert echoed.headers[CORRELATION_ID_HEADER] == "abc-123"
    assert echoed.headers[VERSION_HEADER] == SERVICE_VERSION
    assert generated.headers[CORRELATION_ID_HEADER]


def test_unhandled_error_becomes_500_with_correlation_id(make_app):
    app = make_app()

    @app.get("/boom")
    async def boom():
        raise RuntimeError("boom")

    with TestClient(app) as test_client:
        response = test_client.get("/boom")

    assert response.status_code == 500
    assert response.json() == {
        "detail": "An unexpected internal server error occurred."
    }
    assert response.headers[CORRELATION_ID_HEADER]


def test_root_redirects_to_docs(client):
    response = client.get("/", follow_redirects=False)

    assert response.status_code == 307
    assert response.headers["location"] == "/docs"


# --- Loading the components at startup ------------------------------------------


def test_the_llm_needs_its_endpoint_model_and_key(clean_env):
    status, llm = llm_component(Settings(_env_file=None, llm_api_key="k"))

    assert (status.ready, llm) == (False, None)
    assert status.detail == "RAGLAW_LLM_BASE_URL, RAGLAW_LLM_MODEL are not set."


@pytest.mark.parametrize("api_key", ["", "   "])
def test_a_blank_key_counts_as_missing(clean_env, api_key):
    status, _ = llm_component(
        Settings(_env_file=None, **(LLM_SETTINGS | {"llm_api_key": api_key}))
    )

    assert status.detail == "RAGLAW_LLM_API_KEY is not set."


def test_a_configured_llm_is_ready_without_a_call(clean_env):
    status, llm = llm_component(Settings(_env_file=None, **LLM_SETTINGS))

    assert status.ready
    assert isinstance(llm, OpenAIChat)
    assert llm.model == "Qwen/Qwen2.5-3B-Instruct"


def test_the_models_service_must_answer(clean_env, monkeypatch):
    def down(url):
        raise ModelsServiceError(f"{url} is not answering")

    monkeypatch.setattr(main, "models_health", down)

    status = models_component(Settings(_env_file=None))

    assert not status.ready
    assert "is not answering" in status.detail


def test_the_models_service_must_serve_the_pinned_models(
    clean_env, monkeypatch, models_client
):
    settings = Settings(_env_file=None)
    pinned = models_client.get("/health").json()
    monkeypatch.setattr(main, "models_health", lambda url: pinned)
    assert models_component(settings).ready

    other = pinned | {"reranker": "other/model@" + "f" * 40}
    monkeypatch.setattr(main, "models_health", lambda url: other)
    status = models_component(settings)

    assert not status.ready
    assert "reranker other/model" in status.detail


def test_without_a_registered_champion_the_retriever_isnt_ready(
    champion_settings, tmp_path
):
    empty = f"sqlite:///{tmp_path / 'empty.db'}"
    settings = champion_settings.model_copy(update={"mlflow_tracking_uri": empty})

    status, champion = retriever_component(settings)

    assert (status.ready, champion) == (False, None)
    assert status.detail.startswith("champion not loaded: MlflowException")


def test_without_the_models_service_the_retriever_isnt_ready(
    champion_settings, closed_url, in_production
):
    in_production(champion_settings)
    closed = {"models_url": closed_url}
    status, _ = retriever_component(champion_settings.model_copy(update=closed))

    assert not status.ready
    assert "ModelsServiceError" in status.detail


def test_a_models_service_embedding_to_another_size_leaves_it_unready(
    champion_settings, stub_embeddings, in_production
):
    in_production(champion_settings)
    four_dims = stub_embeddings({}, default=[1.0, 0.0, 0.0, 0.0])

    status, _ = retriever_component(champion_settings, embeddings=four_dims)

    assert not status.ready
    assert "QdrantVectorStoreError" in status.detail


def test_the_retriever_is_the_production_version_and_names_it(
    champion_settings, search_index, in_production
):
    version = in_production(champion_settings)

    status, champion = retriever_component(
        champion_settings, embeddings=search_index.embeddings
    )
    try:
        assert status.ready
        assert status.detail.startswith(
            f"models:/civil-code-retriever@production (version {version.version}, "
            "registry)"
        )
        assert champion.documents_indexed == 5
    finally:
        champion.close()


def test_a_baked_copy_is_named_as_such_on_health(
    champion_settings, search_index, in_production, tmp_path
):
    """D26: the API image loads an exported copy; /health says so."""
    version = in_production(champion_settings)
    export_champion(champion_settings, tmp_path / "champion")
    baked = champion_settings.model_copy(update={"champion_dir": tmp_path / "champion"})

    status, champion = retriever_component(baked, embeddings=search_index.embeddings)
    try:
        assert status.ready
        assert f"(version {version.version}, baked copy at {tmp_path}" in status.detail
    finally:
        champion.close()


def test_the_index_size_is_read_from_its_manifest(champion_settings, tmp_path):
    assert indexed_chunks(champion_settings) == 5
    elsewhere = champion_settings.paths.model_copy(update={"index_dir": tmp_path})
    assert (
        indexed_chunks(champion_settings.model_copy(update={"paths": elsewhere})) == 0
    )


def test_the_pipeline_is_built_only_when_every_component_is_ready(
    champion_settings, search_index, monkeypatch, models_client, in_production
):
    in_production(champion_settings)
    pinned = models_client.get("/health").json()
    monkeypatch.setattr(main, "models_health", lambda url: pinned)
    # model_copy skips validation, so the key goes in as the SecretStr it becomes
    llm = LLM_SETTINGS | {"llm_api_key": SecretStr(LLM_SETTINGS["llm_api_key"])}
    settings = champion_settings.model_copy(update=llm)

    service = load_service(settings, embeddings=search_index.embeddings)
    try:
        assert all(c.ready for c in service.components.values())
        assert service.pipeline is not None
        assert service.documents_indexed == 5
    finally:
        service.close()

    MlflowClient(settings.mlflow_tracking_uri).delete_registered_model_alias(
        "civil-code-retriever", "production"
    )
    unready = load_service(settings, embeddings=search_index.embeddings)
    assert unready.pipeline is None
    assert not unready.components["retriever"].ready


def test_unreachable_artifacts_leave_the_retriever_unready(
    champion_settings, monkeypatch, in_production
):
    in_production(champion_settings)
    from botocore.exceptions import NoCredentialsError

    def no_credentials(*args, **kwargs):
        raise NoCredentialsError()

    monkeypatch.setattr("mlflow.pyfunc.load_model", no_credentials)

    status, _ = retriever_component(champion_settings)

    assert not status.ready
    assert "NoCredentialsError" in status.detail


def test_startup_waits_for_the_models_service_then_stops_waiting(monkeypatch):
    """A restart brings api back before models; it waits instead of degrading."""
    settings = Settings(_env_file=None, models_wait_seconds=30)
    answers = iter([False, False, True])

    def health(url):
        if not next(answers):
            raise ModelsServiceError("not yet")
        return {}

    monkeypatch.setattr(main, "models_health", health)
    now = [0.0]
    sleep = lambda s: now.__setitem__(0, now[0] + s)

    assert main.wait_for_models(settings, sleep=sleep, clock=lambda: now[0])
    assert now[0] == 10  # two waits of 5 s

    monkeypatch.setattr(
        main,
        "models_health",
        lambda url: (_ for _ in ()).throw(ModelsServiceError("down")),
    )
    now[0] = 0.0
    assert not main.wait_for_models(settings, sleep=sleep, clock=lambda: now[0])
    assert now[0] >= 30


def test_u42_ask_redacts_personal_data_and_metrics_count_it(
    ask_client, retrieved, make_chunk
):
    """Phase 14: the guardrail runs on every answer, and each decision is a metric."""
    reply = "Under Article 44, the holder of ID 29801011234567 is of age."
    with ask_client(retrieved([make_chunk(44)]), reply=reply) as client:
        before = _sample(client.get("/metrics").text, "redacted")
        answer = client.post("/ask", json=QUESTION).json()["answer"]
        exposed = client.get("/metrics").text

    assert answer == "Under Article 44, the holder of ID [national_id] is of age."
    assert _sample(exposed, "redacted") == before + 1
    assert 'raglaw_requests_total{path="/ask",status="200"}' in exposed


def test_u42_a_streamed_answer_is_redacted_too(ask_client, retrieved, make_chunk):
    reply = (
        "Call 01012345678 about Article 44, which sets majority at twenty-one years."
    )
    with ask_client(retrieved([make_chunk(44)]), reply=reply, streaming=True) as c:
        events = sse(c.post("/ask/stream", json=QUESTION).text)

    text = "".join(data["token"] for event, data in events if event is None)
    assert "01012345678" not in text and "[phone]" in text


def _sample(exposition: str, decision: str) -> float:
    prefix = f'raglaw_pii_decisions_total{{decision="{decision}"}} '
    lines = [ln for ln in exposition.splitlines() if ln.startswith(prefix)]
    return float(lines[0].removeprefix(prefix)) if lines else 0.0
