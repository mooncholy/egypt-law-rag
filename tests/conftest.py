import logging
import os
import re
import subprocess
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from mlflow.tracking import MlflowClient

from raglaw.api.main import create_app
from raglaw.api.schemas import ComponentStatus
from raglaw.config import Settings
from raglaw.logging_conf import extra_fields
from raglaw.logging_setup import close_logging, setup_logging

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURES_DIR = REPO_ROOT / "tests" / "fixtures"
# Settings reads params.yaml from the working directory, so tests run from the
# repo root, as `dvc repro` does.
FULL_PDF = REPO_ROOT / Settings(_env_file=None).paths.raw_pdf
FIXTURE_PDFS = {
    "page_001": FIXTURES_DIR / "page_001.pdf",
    "pages_007_009": FIXTURES_DIR / "pages_007_009.pdf",
    "pages_046_047": FIXTURES_DIR / "pages_046_047.pdf",
    "page_081": FIXTURES_DIR / "page_081.pdf",
}
NEEDS_FULL_PDF = ("profile", "corpus")


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Skip ``profile`` and ``corpus`` tests when the DVC-tracked PDF isn't pulled.

    CI has no S3 access, so these run only where ``dvc pull`` has fetched it.
    """
    if FULL_PDF.exists():
        return
    skip = pytest.mark.skip(reason="run dvc pull")
    for item in items:
        if any(item.get_closest_marker(marker) for marker in NEEDS_FULL_PDF):
            item.add_marker(skip)


# --- Paths -----------------------------------------------------------------


@pytest.fixture
def repo_root() -> Path:
    """The repository root, for files tests read in place (e.g. ``.env.example``)."""
    return REPO_ROOT


@pytest.fixture
def full_pdf() -> Path:
    """The full DVC-tracked source PDF; only ``profile``/``corpus`` tests use it."""
    return FULL_PDF


@pytest.fixture
def fixture_pdfs() -> dict[str, Path]:
    """The four committed excerpts of the source PDF, keyed by page range."""
    return FIXTURE_PDFS


@pytest.fixture
def stub_path() -> Path:
    """The committed 10-article stub (created from the gold set in Phase 4)."""
    return FIXTURES_DIR / "articles_stub.json"


# --- Configuration ---------------------------------------------------------


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove every ``RAGLAW_*`` variable, so the developer's shell can't leak in."""
    for key in [k for k in os.environ if k.startswith("RAGLAW_")]:
        monkeypatch.delenv(key)


# --- Logging ---------------------------------------------------------------


@pytest.fixture
def pipeline_log(tmp_path: Path) -> Iterator[Path]:
    """Pipeline logging set up for a stage named ``test-stage``, writing to ``tmp_path``."""
    yield setup_logging("test-stage", logs_dir=tmp_path)
    close_logging()


@pytest.fixture
def log_records(
    caplog: pytest.LogCaptureFixture,
) -> Callable[..., list[dict[str, Any]]]:
    """Captured log records as dicts: level, message and the ``extra`` fields.

    Sees loggers that propagate to the root, which every pipeline logger does.
    The API's ``logging_conf.get_logger`` loggers don't propagate, so their
    records aren't captured here.
    """
    caplog.set_level(logging.DEBUG)

    def _records(
        level: int = logging.DEBUG, event_type: str | None = None
    ) -> list[dict[str, Any]]:
        return [
            {"level": r.levelname, "message": r.getMessage(), **extra_fields(r)}
            for r in caplog.records
            if r.levelno >= level
            and (event_type is None or getattr(r, "event_type", None) == event_type)
        ]

    return _records


# --- Tracking --------------------------------------------------------------


@pytest.fixture(scope="session")
def tracking_db(tmp_path_factory: pytest.TempPathFactory) -> str:
    """One MLflow sqlite database for the session; creating one takes ~1.5 s."""
    return f"sqlite:///{tmp_path_factory.mktemp('mlflow')}/mlflow.db"


@pytest.fixture
def tracking_settings(
    tracking_db: str, tmp_path: Path, request: pytest.FixtureRequest
) -> Settings:
    """Settings for a tracked stage, isolated per test.

    The session database, but an experiment named after this test, with its
    artifacts and logs under ``tmp_path``. No AWS profile, so nothing reaches S3.
    """
    settings = Settings(
        _env_file=None,
        mlflow_tracking_uri=tracking_db,
        mlflow_artifact_root=f"file://{tmp_path}/artifacts",
        aws_profile=None,
    )
    return settings.model_copy(
        update={
            "paths": settings.paths.model_copy(update={"logs_dir": tmp_path / "logs"}),
            # model_copy skips validation, so make the name valid here:
            # test_x[None] -> test_x-none-
            "tracking": settings.tracking.model_copy(
                update={
                    "experiments": settings.tracking.experiments.model_copy(
                        update={
                            "corpus": re.sub(
                                r"[^a-z0-9._-]+", "-", request.node.name.lower()
                            )
                        }
                    )
                }
            ),
        }
    )


@pytest.fixture
def mlflow_client(tracking_db: str) -> MlflowClient:
    """A client on the session database, for reading back what a stage recorded."""
    return MlflowClient(tracking_uri=tracking_db)


@pytest.fixture
def git_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A throwaway git repo as the working directory, committed clean.

    It holds a tracked module under ``src/``, a ``dvc.lock`` outside the code
    paths, and a ``.gitignore`` for ``__pycache__``, so a test can change one
    kind of file and see what ``git_state`` makes of it.
    """
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "stage.py").write_text("ROWS = 1\n")
    (tmp_path / "dvc.lock").write_text("md5: 1\n")
    (tmp_path / ".gitignore").write_text("__pycache__/\n")
    # Signing and hooks off, so a developer's global git config can't interfere.
    git = [
        "git",
        "-c",
        "user.name=t",
        "-c",
        "user.email=t@t",
        "-c",
        "commit.gpgsign=false",
    ]
    for args in (
        ["init", "-q"],
        ["add", "."],
        ["commit", "-q", "--no-verify", "-m", "init"],
    ):
        subprocess.run([*git, *args], cwd=tmp_path, check=True, capture_output=True)
    monkeypatch.chdir(tmp_path)
    return tmp_path


# --- API -------------------------------------------------------------------


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
