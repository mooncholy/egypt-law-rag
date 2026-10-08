import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pymupdf
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from langchain_core.embeddings import Embeddings
from mlflow.tracking import MlflowClient

from raglaw.api.main import create_app
from raglaw.api.schemas import ComponentStatus
from raglaw.config import Embedding, Retrieval, Settings
from raglaw.logging_conf import extra_fields
from raglaw.logging_setup import close_logging, setup_logging
from raglaw.records import write_records
from raglaw.retrieval.dense import build_dense
from raglaw.retrieval.document_text import document_text
from raglaw.retrieval.lexical import build_bm25
from raglaw.retrieval.tokenize import bm25_tokenizer
from raglaw.schema import Chunk

REPO_ROOT = Path(__file__).resolve().parents[1]
# Settings reads params.yaml from the working directory, so tests run from the
# repo root, as `dvc repro` does.
FULL_PDF = REPO_ROOT / Settings(_env_file=None).paths.raw_pdf
METRICS_DIR = REPO_ROOT / Settings(_env_file=None).paths.metrics_dir
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
def stage_metrics() -> Callable[[str], dict[str, Any]]:
    """Load a stage's metrics file from ``docs/metrics``, written by ``dvc repro``."""

    def _load(name: str) -> dict[str, Any]:
        path = METRICS_DIR / f"{name}.json"
        if not path.exists():
            pytest.fail(f"{path} is missing: run `dvc repro` first")
        return json.loads(path.read_text("utf-8"))

    return _load


@pytest.fixture
def source_profile(stage_metrics: Callable[[str], dict[str, Any]]) -> dict[str, Any]:
    """The ``profile`` stage's metrics on the full PDF."""
    return stage_metrics("source_profile")


# --- Synthetic PDF ----------------------------------------------------------

# Printed above page 1's table, like the source's promulgation law (P7).
OUTSIDE_TABLE_TEXT = "PROMULGATION LAW"
# Highlighted on page 2, like the source's untranslated passages (P32).
HIGHLIGHTED_TEXT = "As provided in Article 2."
# Rows of a synthetic page: (English cell, Arabic-side cell, English is bold).
# The right column holds Latin text: the built-in font has no Arabic glyphs,
# and the Arabic-specific measures are covered by the `profile` tests instead.
SYNTHETIC_PAGES = [
    [
        ("SECTION I\nGeneral Provisions", "AL-FASL 1", True),
        ("Article 1\nLegislative provisions govern.", "MADA 1\nBody one.", False),
        ("Article 2 No provision may be repealed.", "MADA 2\nBody two.", False),
        ("SECTION II", "AL-FASL 2", True),
    ],
    [
        ("except by a later law.", "Body two, continued.", False),
        ("Articles 3-5 repealed", "Repealed.", False),
        ("Article 6\nAs provided in Article 2.", "MADA 6\nBody six.", False),
    ],
]


def _draw_ruled_table(page: pymupdf.Page, rows: list[tuple[str, str, bool]]) -> None:
    """Draw ``rows`` as a two-column table ruled with lines, as the source is (P3)."""
    x0, mid, x1, row_height = 20, 200, 380, 60
    tops = [30 + i * row_height for i in range(len(rows) + 1)]
    for top in tops:
        page.draw_line((x0, top), (x1, top))
    for x in (x0, mid, x1):
        page.draw_line((x, tops[0]), (x, tops[-1]))
    for top, (en, ar, bold) in zip(tops, rows, strict=False):
        page.insert_text((x0 + 5, top + 15), en, fontname="hebo" if bold else "helv")
        page.insert_text((mid + 5, top + 15), ar, fontname="helv")


@pytest.fixture(scope="session")
def synthetic_pdf(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A two-page PDF laid out like the source: one ruled two-column table per page.

    It holds a bold heading, articles (one with body text on its header line),
    a repeal row, a continuation opening page 2, a keyword-only heading ending
    page 1, a line printed above page 1's table, and a yellow fill under one
    English line on page 2. CI exercises the
    PDF-reading code on it without the source.
    """
    path = tmp_path_factory.mktemp("pdf") / "synthetic.pdf"
    with pymupdf.open() as doc:
        for rows in SYNTHETIC_PAGES:
            _draw_ruled_table(doc.new_page(width=400, height=300), rows)
        doc[0].insert_text((20, 18), OUTSIDE_TABLE_TEXT, fontname="helv")
        page = doc[1]
        [where] = page.search_for(HIGHLIGHTED_TEXT)
        page.draw_rect(where, color=None, fill=(1, 1, 0), overlay=False)
        doc.save(path)
    return path


@pytest.fixture
def highlighted_text() -> str:
    """The English line under a yellow fill on page 2 of ``synthetic_pdf``."""
    return HIGHLIGHTED_TEXT


@pytest.fixture
def outside_table_text() -> str:
    """The line printed above page 1's table in ``synthetic_pdf``."""
    return OUTSIDE_TABLE_TEXT


@pytest.fixture
def tracked_pdf(synthetic_pdf: Path, tmp_path: Path) -> Path:
    """The synthetic PDF next to a `.dvc` file recording its md5, as DVC writes one."""
    pdf = tmp_path / "source.pdf"
    shutil.copy(synthetic_pdf, pdf)
    md5 = hashlib.md5(pdf.read_bytes()).hexdigest()
    pdf.with_name("source.pdf.dvc").write_text(
        f"outs:\n- md5: {md5}\n  path: source.pdf\n", encoding="utf-8"
    )
    return pdf


# --- Embeddings ----------------------------------------------------------------


class StubEmbeddings(Embeddings):
    """Fixed vectors per text, so the semantic splitter is tested without a model.

    A text not in ``vectors`` gets the ``default`` vector.
    """

    def __init__(self, vectors: dict[str, list[float]], default: list[float]) -> None:
        self.vectors, self.default = vectors, default
        self.calls = 0

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        return [self.vectors.get(t.strip(), self.default) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self.vectors.get(text.strip(), self.default)


@pytest.fixture
def stub_embeddings() -> Callable[..., StubEmbeddings]:
    """Build a ``StubEmbeddings`` from a {text: vector} map and a default vector."""
    return StubEmbeddings


class StubTokenizer:
    """A model tokenizer without a model: whitespace pieces, plus the two special
    tokens (``<s>``, ``</s>``) a real one adds to every input."""

    def __init__(self, model_max_length: int = 8192) -> None:
        self.model_max_length = model_max_length

    def tokenize(self, text: str) -> list[str]:
        return text.split()

    def __call__(self, text: str) -> dict[str, list[int]]:
        return {"input_ids": [0, *range(3, 3 + len(text.split())), 2]}


@pytest.fixture
def stub_tokenizer() -> Callable[..., StubTokenizer]:
    """Build a ``StubTokenizer``, optionally with a small ``model_max_length``."""
    return StubTokenizer


# --- Chunks ----------------------------------------------------------------


@pytest.fixture
def make_chunk() -> Callable[..., Chunk]:
    """Build a one-part ``Chunk`` for article ``number``; any field can be overridden."""

    def _make(number: int, text_ar: str = "نص المادة", **overrides: Any) -> Chunk:
        fields = {
            "chunk_id": f"art-{number}-p1",
            "article_number": number,
            "citation": f"Article {number}",
            "heading_path": ["Root"],
            "heading_path_ar": ["الجذر"],
            "source_pages": [1],
            "is_repealed": False,
            "part_index": 1,
            "part_count": 1,
            "paragraphs": [],
            "text_ar": text_ar,
            "text_en": f"English text of article {number}.",
            "strategy": "structural",
        }
        return Chunk(**(fields | overrides))

    return _make


@pytest.fixture
def embed_chunks_file(tmp_path: Path, make_chunk: Callable[..., Chunk]) -> Path:
    """A ``chunks.json`` of two chunks; article 2's document text is the longer."""
    path = tmp_path / "chunks.json"
    write_records(
        path, Chunk, [make_chunk(1, "نص قصير"), make_chunk(2, "نص أطول قليلا من الأول")]
    )
    return path


@pytest.fixture
def lexical_chunks(make_chunk: Callable[..., Chunk]) -> list[Chunk]:
    """Two chunks sharing one Arabic word (سنة), each with one English word."""
    return [
        make_chunk(1, "التقادم خمس عشرة سنة", text_en="Prescription."),
        make_chunk(2, "الأهلية إحدى وعشرون سنة", text_en="Majority."),
    ]


# --- Search index ------------------------------------------------------------

# The pinned model, as params.yaml names it; tests never load it.
EMBEDDING = Embedding(
    model="BAAI/bge-m3",
    revision="5617a9f61b028005a4858fdac845db406aefb181",
    batch_size=2,
)
DENSE_VARIANT = "both_without_headings"


@pytest.fixture
def embedding_config() -> Embedding:
    """The pinned embedding model's config, for building and checking stub indexes."""
    return EMBEDDING


@pytest.fixture
def dense_chunks(make_chunk: Callable[..., Chunk]) -> list[Chunk]:
    """Three chunks; the third has an English-only passage (R28)."""
    return [
        make_chunk(1, "التقادم خمس عشرة سنة"),
        make_chunk(2, "الأهلية إحدى وعشرون سنة"),
        make_chunk(3, "الفوائد سبعة في المائة", only_in_en=["Only English."]),
    ]


@pytest.fixture
def built_dense(
    tmp_path: Path,
    dense_chunks: list[Chunk],
    stub_embeddings: Callable[..., StubEmbeddings],
) -> tuple[Path, Any, StubEmbeddings, list[str]]:
    """``dense_chunks`` embedded into ``tmp_path/dense``.

    Chunk 2's text has its own direction in the stub space; the rest share one.
    Returns the directory, the manifest, the embeddings and the document texts.
    """
    texts = [document_text(c, DENSE_VARIANT) for c in dense_chunks]
    embeddings = stub_embeddings({texts[1]: [0.0, 1.0, 0.0]}, default=[1.0, 0.0, 0.0])
    manifest = build_dense(
        dense_chunks,
        tmp_path / "dense",
        embeddings=embeddings,
        embedding=EMBEDDING,
        document_text=DENSE_VARIANT,
        chunks_sha256="0" * 64,
    )
    return tmp_path / "dense", manifest, embeddings, texts


# Every query embeds to this vector, so any wording ranks the same by cosine:
# art-3 (0.995), art-1 (0.100), art-4-p2 (0.080), art-4-p1 (0.060), art-2 (0).
QUERY_VECTOR = [0.1, 0.0, 1.0]
SEARCH_VECTORS = {
    "art-1-p1": [1.0, 0.0, 0.0],
    "art-2-p1": [0.0, 1.0, 0.0],
    "art-3-p1": [0.0, 0.0, 1.0],
    "art-4-p1": [0.6, 0.8, 0.0],
    "art-4-p2": [0.8, 0.6, 0.0],
}


@dataclass(frozen=True)
class SearchIndex:
    """A built two-half index over ``chunks``, as ``embed`` and ``bm25`` write it."""

    dense_dir: Path
    bm25_dir: Path
    chunks: list[Chunk]
    embeddings: StubEmbeddings
    retrieval: Retrieval


@pytest.fixture
def search_index(
    tmp_path: Path,
    make_chunk: Callable[..., Chunk],
    stub_embeddings: Callable[..., StubEmbeddings],
) -> SearchIndex:
    """Five chunks indexed in both halves; article 4 has two parts.

    Dense order is fixed by ``SEARCH_VECTORS`` (any query: 3, 1, 4-p2, 4-p1, 2).
    Only article 2's English holds "majority", so BM25 for it finds article 2
    alone. Nothing holds "4" as a term.
    """
    chunks = [
        make_chunk(1, "التقادم خمس عشرة سنة", text_en="Prescription is fifteen years."),
        make_chunk(
            2, "الأهلية إحدى وعشرون سنة", text_en="Majority is twenty-one years."
        ),
        make_chunk(3, "الفوائد سبعة في المائة", text_en="Interest is seven percent."),
        make_chunk(4, "(١( الوديعة عقد", text_en="Deposit.", part_count=2),
        make_chunk(
            4,
            "(٢( يلتزم المودع لديه",
            text_en="Deposit.",
            chunk_id="art-4-p2",
            part_index=2,
            part_count=2,
        ),
    ]
    retrieval = Retrieval(document_text=DENSE_VARIANT, bm25_tokenizer="words")
    texts = [document_text(c, DENSE_VARIANT) for c in chunks]
    embeddings = stub_embeddings(
        {t: SEARCH_VECTORS[c.chunk_id] for c, t in zip(chunks, texts, strict=True)},
        default=QUERY_VECTOR,
    )
    common = {"retrieval": retrieval, "chunks_sha256": "0" * 64}
    build_dense(
        chunks,
        tmp_path / "dense",
        embeddings=embeddings,
        embedding=EMBEDDING,
        document_text=DENSE_VARIANT,
        chunks_sha256="0" * 64,
    )
    build_bm25(
        chunks,
        tmp_path / "bm25",
        tokenize=bm25_tokenizer("words"),
        embedding=EMBEDDING,
        **common,
    )
    return SearchIndex(
        tmp_path / "dense", tmp_path / "bm25", chunks, embeddings, retrieval
    )


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
