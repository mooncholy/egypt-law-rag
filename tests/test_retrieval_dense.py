import json
import uuid

import pytest
from langchain_qdrant import QdrantVectorStore

from raglaw.retrieval.dense import (
    COLLECTION,
    DenseManifest,
    build_dense,
    count_points,
    point_id,
    qdrant_client,
)
from raglaw.retrieval.manifest import read_manifest

pytestmark = pytest.mark.unit

VARIANT = "both_without_headings"  # what built_dense indexes


def points(dense_dir):
    client = qdrant_client(dense_dir)
    try:
        records, _ = client.scroll(COLLECTION, limit=100, with_payload=True)
        return sorted(records, key=lambda r: r.payload["metadata"]["chunk_id"])
    finally:
        client.close()


# --- U29: every chunk indexed once, with its full payload ------------------------


def test_every_chunk_is_one_point_with_its_full_payload(built_dense, dense_chunks):
    dense_dir, _, _, texts = built_dense

    records = points(dense_dir)

    assert [r.payload["metadata"] for r in records] == [
        c.model_dump(mode="json") for c in dense_chunks
    ]
    assert [r.payload["page_content"] for r in records] == texts
    assert [str(r.id) for r in records] == [point_id(c.chunk_id) for c in dense_chunks]
    assert count_points(dense_dir) == 3


def test_point_ids_are_stable_uuids_of_the_chunk_id():
    assert point_id("art-1-p1") == str(uuid.uuid5(uuid.NAMESPACE_URL, "art-1-p1"))
    assert point_id("art-1-p1") != point_id("art-1-p2")


def test_rebuilding_replaces_the_collection_instead_of_adding_to_it(
    built_dense, dense_chunks, stub_embeddings, embedding_config
):
    dense_dir, *_ = built_dense

    build_dense(
        dense_chunks[:2],
        dense_dir,
        embeddings=stub_embeddings({}, default=[1.0, 0.0, 0.0]),
        embedding=embedding_config,
        document_text=VARIANT,
        chunks_sha256="1" * 64,
    )

    assert count_points(dense_dir) == 2
    assert read_manifest(dense_dir, DenseManifest).chunk_ids == ["art-1-p1", "art-2-p1"]


def test_dense_search_finds_the_chunk_nearest_the_query(built_dense):
    dense_dir, _, embeddings, texts = built_dense
    client = qdrant_client(dense_dir)
    try:
        store = QdrantVectorStore(client, COLLECTION, embedding=embeddings)
        [(document, score)] = store.similarity_search_with_score(texts[1], k=1)
    finally:
        client.close()

    assert document.metadata["chunk_id"] == "art-2-p1"
    assert score == pytest.approx(1.0)


# --- The manifest ---------------------------------------------------------------


def test_the_manifest_records_what_the_vectors_were_built_from(
    built_dense, embedding_config
):
    dense_dir, manifest, *_ = built_dense

    assert read_manifest(dense_dir, DenseManifest) == manifest
    assert manifest == DenseManifest(
        chunks_sha256="0" * 64,
        chunk_ids=["art-1-p1", "art-2-p1", "art-3-p1"],
        document_text=VARIANT,
        collection=COLLECTION,
        embedding_model=embedding_config.model,
        embedding_revision=embedding_config.revision,
        embedding_dim=3,
        repealed_text="note",
    )


def test_the_manifest_is_written_deterministically(built_dense):
    dense_dir, *_ = built_dense
    text = (dense_dir / "manifest.json").read_text("utf-8")

    assert text == json.dumps(json.loads(text), indent=2, sort_keys=True) + "\n"
