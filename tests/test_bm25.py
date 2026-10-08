import json

import pytest

from raglaw.config import Embedding, Retrieval
from raglaw.ingest.bm25 import run_bm25, run_params
from raglaw.records import write_records
from raglaw.retrieval.lexical import Bm25Manifest
from raglaw.retrieval.manifest import read_manifest
from raglaw.schema import Chunk, LogEvent
from raglaw.tracking import sha256_file

EMBEDDING = Embedding(
    model="BAAI/bge-m3",
    revision="5617a9f61b028005a4858fdac845db406aefb181",
    batch_size=2,
)
WORDS = Retrieval(document_text="both_without_headings", bm25_tokenizer="words")


def write_chunks(tmp_path, *chunks):
    path = tmp_path / "chunks.json"
    write_records(path, Chunk, chunks)
    return path


def run(tmp_path, chunks_file, retrieval=WORDS, tokenizer=None):
    return run_bm25(
        chunks_file,
        tmp_path / "bm25",
        tmp_path / "bm25.json",
        embedding=EMBEDDING,
        retrieval=retrieval,
        tokenizer=tokenizer,
    )


@pytest.mark.unit
def test_the_stage_indexes_every_chunk_and_writes_its_metrics(tmp_path, make_chunk):
    chunks_file = write_chunks(
        tmp_path,
        make_chunk(1, "التقادم خمس عشرة سنة", text_en="Prescription."),
        make_chunk(2, "الأهلية سنة", text_en="Majority."),
    )

    metrics = run(tmp_path, chunks_file)

    assert json.loads((tmp_path / "bm25.json").read_text("utf-8")) == metrics
    assert metrics["chunks_indexed"] == 2
    assert metrics["vocabulary_size"] == 7
    assert metrics["max_chunk_terms"] == 5
    assert metrics["chunks_without_terms"] == 0
    assert metrics["bm25_tokenizer"] == "words"
    manifest = read_manifest(tmp_path / "bm25", Bm25Manifest)
    assert manifest.chunks_sha256 == sha256_file(chunks_file)


@pytest.mark.unit
def test_a_chunk_with_no_terms_is_logged_and_counted(tmp_path, make_chunk, log_records):
    """BM25 can never find it, so it is surfaced, not absorbed."""
    chunks_file = write_chunks(
        tmp_path, make_chunk(1, "نص"), make_chunk(2, "...", text_en="—")
    )

    metrics = run(tmp_path, chunks_file)

    assert metrics["chunks_without_terms"] == 1
    [anomaly] = log_records(event_type=LogEvent.ANOMALY)
    assert (anomaly["anomaly_type"], anomaly["article_number"]) == (
        "chunk_without_terms",
        2,
    )


@pytest.mark.unit
def test_model_subwords_uses_the_given_model_tokenizer(
    tmp_path, make_chunk, stub_tokenizer
):
    chunks_file = write_chunks(tmp_path, make_chunk(1, "نص"))
    subwords = Retrieval(
        document_text="both_without_headings", bm25_tokenizer="model_subwords"
    )

    run(tmp_path, chunks_file, subwords, stub_tokenizer())

    manifest = read_manifest(tmp_path / "bm25", Bm25Manifest)
    assert manifest.tokenizer_model == EMBEDDING.model


@pytest.mark.unit
def test_runs_are_compared_by_tokenizer_and_document_text():
    assert run_params(WORDS) == {
        "document_text": "both_without_headings",
        "bm25_tokenizer": "words",
    }


# --- Corpus: the real index -------------------------------------------------------


@pytest.mark.corpus
def test_every_chunk_is_in_the_bm25_index(stage_metrics):
    bm25, chunks = stage_metrics("bm25"), stage_metrics("chunks")

    assert bm25["chunks_indexed"] == chunks["chunks"]
