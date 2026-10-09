import json

import pytest

from raglaw.config import Embedding, Retrieval
from raglaw.ingest.embed import EmbedError, run_embed, run_params
from raglaw.retrieval.dense import DenseManifest
from raglaw.retrieval.manifest import read_manifest
from raglaw.schema import LogEvent
from raglaw.tracking import sha256_file

EMBEDDING = Embedding(
    model="BAAI/bge-m3",
    revision="5617a9f61b028005a4858fdac845db406aefb181",
    batch_size=2,
)
RETRIEVAL = Retrieval(document_text="both_with_headings", bm25_tokenizer="words")


def run(tmp_path, chunks_file, embeddings, tokenizer):
    return run_embed(
        chunks_file,
        tmp_path / "dense",
        tmp_path / "embed.json",
        embedding=EMBEDDING,
        retrieval=RETRIEVAL,
        embeddings=embeddings,
        tokenizer=tokenizer,
        device="cpu",
    )


@pytest.mark.unit
def test_the_stage_embeds_every_chunk_and_writes_its_metrics(
    tmp_path, embed_chunks_file, stub_embeddings, stub_tokenizer
):
    metrics = run(
        tmp_path, embed_chunks_file, stub_embeddings({}, [1.0, 0.0]), stub_tokenizer()
    )

    assert json.loads((tmp_path / "embed.json").read_text("utf-8")) == metrics
    assert metrics["chunks_indexed"] == 2
    assert metrics["embedding_dim"] == 2
    assert metrics["truncated_chunks"] == 0
    assert metrics["max_tokens"] == 8192
    assert metrics["device"] == "cpu"
    assert metrics["document_text"] == "both_with_headings"
    # The longest document text: 12 whitespace pieces, plus <s> and </s>
    assert metrics["max_chunk_tokens"] == 14
    manifest = read_manifest(tmp_path / "dense", DenseManifest)
    assert manifest.chunks_sha256 == sha256_file(embed_chunks_file)


@pytest.mark.unit
def test_a_chunk_the_model_would_truncate_stops_the_stage_before_embedding(
    tmp_path, embed_chunks_file, stub_embeddings, stub_tokenizer, log_records
):
    embeddings = stub_embeddings({}, [1.0, 0.0])

    with pytest.raises(EmbedError, match="truncate"):
        run(
            tmp_path, embed_chunks_file, embeddings, stub_tokenizer(model_max_length=12)
        )  # 11 and 14 tokens

    assert embeddings.calls == 0
    assert not (tmp_path / "dense").exists()
    [anomaly] = log_records(event_type=LogEvent.ANOMALY)
    assert (anomaly["anomaly_type"], anomaly["article_number"]) == (
        "truncated_chunk",
        2,
    )


@pytest.mark.unit
def test_runs_are_compared_by_model_and_document_text():
    assert run_params(EMBEDDING, RETRIEVAL, "cpu") == {
        "embedding_model": "BAAI/bge-m3",
        "embedding_revision": EMBEDDING.revision,
        "document_text": "both_with_headings",
        "repealed_text": "note",
        "device": "cpu",
    }


# --- Corpus: the real index (C17) -------------------------------------------------


@pytest.mark.corpus
def test_every_chunk_is_embedded_and_none_truncated(stage_metrics):
    embed, chunks = stage_metrics("embed"), stage_metrics("chunks")

    assert embed["chunks_indexed"] == chunks["chunks"]
    assert embed["truncated_chunks"] == 0
