import hashlib
import json
import logging
import os
import re
import shutil
import socket
import subprocess
import sys
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import pymupdf
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from mlflow.entities.model_registry import ModelVersion
from mlflow.tracking import MlflowClient

from raglaw.api.main import Service, create_app
from raglaw.api.schemas import ComponentStatus
from raglaw.config import Answer as AnswerConfig
from raglaw.config import Embedding, Retrieval, Search, Settings
from raglaw.ingest.evaluate_retrieval import evaluate, run_params
from raglaw.logging_conf import extra_fields
from raglaw.logging_setup import close_logging, setup_logging
from raglaw.rag import AnswerPipeline
from raglaw.records import write_records
from raglaw.retrieval.champion import MODEL_NAME, promote, register_champion
from raglaw.retrieval.dense import build_dense
from raglaw.retrieval.document_text import document_text
from raglaw.retrieval.lexical import build_bm25
from raglaw.retrieval.retriever import HybridRetriever
from raglaw.retrieval.scoring import EvalQuestion, QuestionScore
from raglaw.retrieval.tokenize import bm25_tokenizer
from raglaw.schema import Chunk
from raglaw.serving.models_app import create_models_app
from raglaw.tracking import stage_run

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
    """Load a metrics file from ``docs/metrics``, written by ``dvc repro`` or a script."""

    def _load(name: str) -> dict[str, Any]:
        path = METRICS_DIR / f"{name}.json"
        if not path.exists():
            pytest.fail(f"{path} is missing: run `dvc repro` (or its script) first")
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
    texts: dict[str, str]  # chunk_id -> its document text


@pytest.fixture
def build_search_index(
    tmp_path: Path, stub_embeddings: Callable[..., StubEmbeddings]
) -> Callable[..., SearchIndex]:
    """Index ``chunks`` in both halves under ``tmp_path/<name>``.

    Each chunk embeds to ``vectors[chunk_id]``; every query embeds to
    ``query_vector``, so the dense order is fixed whatever the wording.
    """

    def _build(
        chunks: list[Chunk],
        vectors: dict[str, list[float]],
        query_vector: list[float],
        retrieval: Retrieval | None = None,
        name: str = "index",
    ) -> SearchIndex:
        retrieval = retrieval or Retrieval(
            document_text=DENSE_VARIANT, bm25_tokenizer="words"
        )
        texts = {
            c.chunk_id: document_text(
                c, retrieval.document_text, repealed_text=retrieval.repealed_text
            )
            for c in chunks
        }
        embeddings = stub_embeddings(
            {texts[c.chunk_id]: vectors[c.chunk_id] for c in chunks},
            default=query_vector,
        )
        root = tmp_path / name
        build_dense(
            chunks,
            root / "dense",
            embeddings=embeddings,
            embedding=EMBEDDING,
            document_text=retrieval.document_text,
            repealed_text=retrieval.repealed_text,
            chunks_sha256="0" * 64,
        )
        build_bm25(
            chunks,
            root / "bm25",
            tokenize=bm25_tokenizer("words"),
            retrieval=retrieval,
            embedding=EMBEDDING,
            chunks_sha256="0" * 64,
        )
        return SearchIndex(
            root / "dense", root / "bm25", chunks, embeddings, retrieval, texts
        )

    return _build


@pytest.fixture
def search_index(
    make_chunk: Callable[..., Chunk], build_search_index: Callable[..., SearchIndex]
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
    return build_search_index(chunks, SEARCH_VECTORS, QUERY_VECTOR)


@pytest.fixture
def citing_index(
    make_chunk: Callable[..., Chunk], build_search_index: Callable[..., SearchIndex]
) -> SearchIndex:
    """Article 1 cites 3 in Arabic (dual form) and 4 in English; dense: 1, 2, 3, 4, 5.

    Article 5 is a repealed note citing its own range and article 2.
    """
    chunks = [
        make_chunk(1, "تسري أحكام المادتين ٣ ، ٢", text_en="Articles 4 and 2 apply."),
        make_chunk(2, "نص ثان", text_en="Second."),
        make_chunk(3, "نص ثالث", text_en="Third."),
        make_chunk(4, "نص رابع", text_en="Fourth."),
        make_chunk(
            5,
            "المواد من ٥ إلى ٦ ملغاة وفقا للمادة ٢",
            is_repealed=True,
            text_en="Articles 5-6 repealed",
        ),
    ]
    vectors = {
        "art-1-p1": [1.0, 0.0],
        "art-2-p1": [0.9, 0.44],
        "art-3-p1": [0.7, 0.71],
        "art-4-p1": [0.5, 0.87],
        "art-5-p1": [0.1, 0.99],
    }
    return build_search_index(chunks, vectors, [1.0, 0.0], name="citing")


class StubReranker:
    """Fixed scores per document text, so reranking is tested without a model.

    A text not in ``scores`` gets ``default``; ``calls`` counts the questions scored.
    """

    def __init__(self, scores: dict[str, float], default: float = 0.0) -> None:
        self.scores, self.default = scores, default
        self.calls = 0

    def score(self, query: str, texts: list[str]) -> list[float]:
        self.calls += 1
        return [self.scores.get(t, self.default) for t in texts]


@pytest.fixture
def stub_reranker() -> Callable[..., StubReranker]:
    """Build a ``StubReranker`` from a {document text: score} map and a default."""
    return StubReranker


@pytest.fixture
def champion_settings(
    tracking_settings: Settings,
    search_index: SearchIndex,
    embedding_config: Embedding,
) -> Settings:
    """``tracking_settings`` pointed at ``search_index``, with its config.

    The index lives at ``<tmp>/index/{dense,bm25}``, as a deployment's
    ``paths.index_dir`` would. Reranking is off, so opening the champion
    never loads the real cross-encoder.
    """
    return tracking_settings.model_copy(
        update={
            "embedding": embedding_config,
            "retrieval": search_index.retrieval,
            "search": tracking_settings.search.model_copy(update={"rerank": False}),
            "paths": tracking_settings.paths.model_copy(
                update={"index_dir": search_index.dense_dir.parent}
            ),
        }
    )


@pytest.fixture
def evaluation_run() -> Callable[..., str]:
    """Log a finished retrieval run with ``settings``' config, as
    ``evaluate_retrieval`` logs it (any param can be overridden); returns its id."""

    def _run(settings: Settings, **param_overrides: Any) -> str:
        with stage_run(
            "evaluate_retrieval",
            input_hash="x",
            settings=settings,
            experiment="retrieval",
            run_name="champion",
        ) as run:
            run.log_params(
                run_params(settings, chunks_sha256="0" * 64) | param_overrides
            )
            run.log_metrics({"overall.recall_at_5": 0.7})
        return run.run_id

    return _run


@pytest.fixture
def in_production(evaluation_run: Callable[..., str]) -> Callable[..., ModelVersion]:
    """Register a run with ``settings``' config, scored ``heldout_recall`` on the
    held-out half, and promote it to ``production``.

    Every caller uses the same default score, so on the shared session
    registry each promotion is "not lower" than the last.
    """

    def _promote(settings: Settings, heldout_recall: float = 0.8) -> ModelVersion:
        register_champion(
            settings,
            evaluation_run(settings),
            heldout={"recall_at_5": heldout_recall},
        )
        return promote(settings)

    return _promote


# --- Evaluation --------------------------------------------------------------


@pytest.fixture
def make_question() -> Callable[..., EvalQuestion]:
    """Build an in-scope English ``rule`` question expecting ``articles``; any
    field can be overridden (an out-of-scope one needs ``expected_articles=[]``)."""

    def _make(qid: str, articles: list[int] | None = None, **overrides: Any):
        fields = {
            "id": qid,
            "question": f"Question {qid}?",
            "language": "en",
            "register": "english",
            "kind": "rule",
            "expected_articles": [1] if articles is None else articles,
            "supporting_articles": [],
            "match": "all",
            "legal_basis": "A test question.",
            "difficulty": "easy",
        }
        return EvalQuestion(**(fields | overrides))

    return _make


@pytest.fixture
def search_questions(make_question: Callable[..., EvalQuestion]) -> list[EvalQuestion]:
    """Questions over ``search_index``, with outcomes worked out by hand.

    q001 and its Arabic twin q005 find article 2 first (BM25: majority,
    الاهليه); q002 finds article 3 first (interest); q003 expects article 99,
    which is never returned; q004 is out of scope.
    """
    return [
        make_question("q001", [2], question="majority", pair_id="p1"),
        make_question("q002", [3], question="interest"),
        make_question("q003", [99], question="majority"),
        make_question("q004", [], question="theft penalty?", kind="out_of_scope"),
        make_question(
            "q005",
            [2],
            question="الأهلية",
            language="ar",
            register="msa",
            pair_id="p1",
            translated_from="q001",
        ),
    ]


@pytest.fixture
def search_scores(
    search_index: SearchIndex,
    search_questions: list[EvalQuestion],
) -> list[QuestionScore]:
    """``search_questions`` scored by a hybrid retriever over ``search_index``."""
    search = Search(
        mode="hybrid", article_lookup=True, candidates=50, rrf_k=60, top_k=10
    )
    with HybridRetriever.from_index(
        search_index.dense_dir,
        search_index.bm25_dir,
        embedding=EMBEDDING,
        retrieval=search_index.retrieval,
        search=search,
        embeddings=search_index.embeddings,
    ) as retriever:
        return evaluate(search_questions, retriever)


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

    The session database, but experiments named after this test (``<test>``
    for corpus runs, ``<test>-retrieval`` for retrieval runs), with artifacts
    and logs under ``tmp_path``. No AWS profile, so nothing reaches S3.
    """
    name = re.sub(r"[^a-z0-9._-]+", "-", request.node.name.lower())
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
                        update={"corpus": name, "retrieval": f"{name}-retrieval"}
                    )
                }
            ),
        }
    )


@pytest.fixture(scope="session")
def mlflow_server(tmp_path_factory: pytest.TempPathFactory) -> Iterator[str]:
    """A real MLflow tracking server, as Phase 8 runs in compose: a sqlite
    backend instead of Postgres, and artifacts proxied to a local directory
    instead of S3. Yields its URL."""
    root = tmp_path_factory.mktemp("mlflow-server")
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    process = subprocess.Popen(
        [
            sys.executable, "-m", "mlflow", "server",
            "--host", "127.0.0.1",
            "--port", str(port),
            "--backend-store-uri", f"sqlite:///{root / 'mlflow.db'}",
            "--serve-artifacts",
            "--artifacts-destination", (root / "artifacts").as_uri(),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )  # fmt: skip
    url = f"http://127.0.0.1:{port}"
    deadline = time.monotonic() + 90
    while True:
        try:
            if httpx.get(f"{url}/health", timeout=1).status_code == 200:
                break
        except httpx.HTTPError:
            pass
        if process.poll() is not None or time.monotonic() > deadline:
            process.kill()
            pytest.fail(f"mlflow server didn't start on {url}")
        time.sleep(0.5)
    yield url
    process.terminate()
    process.wait(timeout=20)


@pytest.fixture
def server_settings(
    champion_settings: Settings, mlflow_server: str
) -> Iterator[Settings]:
    """``champion_settings`` tracking to ``mlflow_server``, with no artifact
    root, so the server proxies artifacts as in compose. The registered model
    is deleted first, so each test starts with an empty registry."""
    settings = champion_settings.model_copy(
        update={"mlflow_tracking_uri": mlflow_server, "mlflow_artifact_root": None}
    )
    client = MlflowClient(mlflow_server)
    if client.search_registered_models(f"name = '{MODEL_NAME}'"):
        client.delete_registered_model(MODEL_NAME)
    yield settings


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


# --- Answering ---------------------------------------------------------------


class StubLLM:
    """Replies ``reply`` to every chat, and keeps each chat it was sent."""

    def __init__(self, reply: str = "") -> None:
        self.reply = reply
        self.calls: list[list[dict[str, str]]] = []

    async def complete(
        self, messages: list[dict[str, str]], *, temperature: float
    ) -> str:
        self.calls.append(messages)
        return self.reply


@pytest.fixture
def stub_llm() -> Callable[..., StubLLM]:
    """Build a ``StubLLM`` with a fixed reply."""
    return StubLLM


@pytest.fixture
def retrieved() -> Callable[..., list[Document]]:
    """Chunks as ``HybridRetriever`` returns them, in the given rank order.

    Rank 1 carries ``rerank_score`` and ``lookup`` (fetched by its article
    number); the other ranks score just below it.
    """

    def _retrieved(
        chunks: list[Chunk], rerank_score: float | None = 0.9, lookup: bool = False
    ) -> list[Document]:
        documents = []
        for rank, chunk in enumerate(chunks, start=1):
            score = rerank_score if rank == 1 or rerank_score is None else 0.1
            found = {
                "rank": rank,
                "rerank_score": score,
                "lookup": lookup and rank == 1,
            }
            documents.append(
                Document(
                    id=chunk.chunk_id,
                    page_content=chunk.text_ar,
                    metadata={**chunk.model_dump(mode="json"), "retrieval": found},
                )
            )
        return documents

    return _retrieved


@pytest.fixture
def answer_config() -> AnswerConfig:
    """``params.yaml``'s answer block, with the no-answer threshold at 0.3."""
    return Settings(_env_file=None).answer.model_copy(
        update={"no_answer_threshold": 0.3, "max_sources": 5}
    )


@pytest.fixture
def make_pipeline(
    answer_config: AnswerConfig,
) -> Callable[..., tuple[AnswerPipeline, StubLLM]]:
    """An ``AnswerPipeline`` retrieving ``documents`` for any question, and its LLM."""

    def _make(
        documents: list[Document], reply: str = ""
    ) -> tuple[AnswerPipeline, StubLLM]:
        llm = StubLLM(reply)
        return AnswerPipeline(lambda _: documents, llm, answer_config), llm

    return _make


class _Capture(logging.Handler):
    def __init__(self) -> None:
        super().__init__(logging.DEBUG)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


@pytest.fixture
def rag_log_records() -> Iterator[Callable[..., list[dict[str, Any]]]]:
    """``raglaw.rag``'s records as dicts. Its logger (``get_logger``) doesn't
    propagate, so ``log_records`` can't see it."""
    logger = logging.getLogger("raglaw.rag")
    handler = _Capture()
    logger.addHandler(handler)

    def _records(event_type: str | None = None) -> list[dict[str, Any]]:
        return [
            {"level": r.levelname, "message": r.getMessage(), **extra_fields(r)}
            for r in handler.records
            if event_type is None or getattr(r, "event_type", None) == event_type
        ]

    yield _records
    logger.removeHandler(handler)


# --- Models service --------------------------------------------------------


@pytest.fixture
def closed_url() -> str:
    """A local URL nothing listens on: a port just freed, so a connection is
    refused at once. (A fixed port such as 9 can hang instead, e.g. under WSL's
    mirrored networking.)"""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    return f"http://127.0.0.1:{port}"


@pytest.fixture
def models_client(
    stub_embeddings: Callable[..., StubEmbeddings],
    stub_reranker: Callable[..., StubReranker],
) -> Iterator[TestClient]:
    """A started ``models`` service on stub models: every text embeds to
    [1, 0], and "relevant" scores 0.9 against any question, anything else 0.1."""
    app = create_models_app(
        Settings(_env_file=None),
        embeddings=stub_embeddings({"query": [0.0, 1.0]}, default=[1.0, 0.0]),
        reranker=stub_reranker({"relevant": 0.9}, default=0.1),
    )
    with TestClient(app) as test_client:
        yield test_client


# --- API -------------------------------------------------------------------

NOT_CONFIGURED = {
    "llm": "RAGLAW_LLM_API_KEY, RAGLAW_LLM_BASE_URL, RAGLAW_LLM_MODEL are not set.",
    "models": "http://localhost:8001 is not answering.",
    "retriever": "champion not loaded.",
}


@pytest.fixture
def make_service() -> Callable[..., Service]:
    """A loaded ``Service``: every component ready with ``pipeline``, else none
    ready, as a fresh checkout starts."""

    def _make(
        pipeline: AnswerPipeline | None = None, documents_indexed: int = 0
    ) -> Service:
        if pipeline is None:
            components = {
                name: ComponentStatus(ready=False, detail=detail)
                for name, detail in NOT_CONFIGURED.items()
            }
        else:
            components = {
                name: ComponentStatus(ready=True, detail="ok")
                for name in NOT_CONFIGURED
            }
        return Service(components, documents_indexed, pipeline)

    return _make


@pytest.fixture
def make_app(make_service: Callable[..., Service]) -> Callable[..., FastAPI]:
    """Build an API app whose startup loads ``service``, never the machine's
    registry, models service or .env."""

    def _make_app(service: Service | None = None) -> FastAPI:
        loaded = service or make_service()
        return create_app(Settings(_env_file=None), load=lambda _: loaded)

    return _make_app


@pytest.fixture
def client(make_app: Callable[..., FastAPI]) -> Iterator[TestClient]:
    """A started API with no component ready, as a fresh checkout runs."""
    with TestClient(make_app()) as test_client:
        yield test_client


@pytest.fixture
def ask_client(
    make_app: Callable[..., FastAPI],
    make_service: Callable[..., Service],
    make_pipeline: Callable[..., tuple[AnswerPipeline, StubLLM]],
) -> Callable[..., TestClient]:
    """A started, fully ready API answering through a stub pipeline; enter it
    with ``with``."""

    def _make(
        documents: list[Document], reply: str = "", documents_indexed: int = 1150
    ) -> TestClient:
        pipeline, _ = make_pipeline(documents, reply)
        return TestClient(make_app(make_service(pipeline, documents_indexed)))

    return _make
