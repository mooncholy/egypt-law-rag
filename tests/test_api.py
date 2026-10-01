import pytest
from fastapi.testclient import TestClient

from raglaw.api.main import CORRELATION_ID_HEADER

pytestmark = pytest.mark.unit

QUESTION = {"question": "ما هي شروط الأهلية؟"}


def test_health_is_degraded_without_key(client):
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "degraded"
    assert body["components"]["llm"] == {
        "ready": False,
        "detail": "RAGLAW_LLM_API_KEY is not set.",
    }


@pytest.mark.parametrize("api_key", ["", "   "])
def test_blank_key_counts_as_missing(make_app, api_key):
    with TestClient(make_app(api_key)) as test_client:
        llm = test_client.get("/health").json()["components"]["llm"]

    assert llm["ready"] is False


def test_key_makes_llm_ready(make_app):
    with TestClient(make_app("test-key")) as test_client:
        llm = test_client.get("/health").json()["components"]["llm"]

    assert llm["ready"] is True


def test_health_is_ok_when_every_component_is_ready(ready_client):
    assert ready_client.get("/health").json()["status"] == "ok"


def test_ask_answers_503_with_each_reason(client):
    response = client.post("/ask", json=QUESTION)

    assert response.status_code == 503
    assert response.json()["reasons"] == {
        "llm": "RAGLAW_LLM_API_KEY is not set.",
        "retriever": "The retrieval index is not loaded.",
    }


def test_ask_is_501_when_ready_until_phase_2(ready_client):
    assert ready_client.post("/ask", json=QUESTION).status_code == 501


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
