from collections.abc import Callable, Iterator

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from raglaw.api.main import create_app
from raglaw.api.schemas import ComponentStatus
from raglaw.config import Settings


@pytest.fixture
def make_app() -> Callable[..., FastAPI]:
    """Build an API app with the given LLM key, ignoring the developer's real .env."""

    def _make_app(api_key: str | None = None) -> FastAPI:
        return create_app(Settings(_env_file=None, llm_api_key=api_key))

    return _make_app


@pytest.fixture
def client(make_app: Callable[..., FastAPI]) -> Iterator[TestClient]:
    """A started API with no LLM key and no retriever, as a fresh checkout runs."""
    with TestClient(make_app()) as test_client:
        yield test_client


@pytest.fixture
def ready_client(client: TestClient) -> TestClient:
    """``client`` with every component marked ready, as a fully configured service."""
    client.app.state.components = {
        "llm": ComponentStatus(ready=True, detail="ok"),
        "retriever": ComponentStatus(ready=True, detail="ok"),
    }
    return client
