import json
import uuid

import pytest
from langchain_qdrant import QdrantVectorStore

from raglaw.config import Embedding
from raglaw.retrieval.dense import (
    COLLECTION,
    DenseManifest,
    build_dense,
    count_points,
    point_id,
    qdrant_client,
)
from raglaw.retrieval.document_text import document_text
from raglaw.retrieval.manifest import read_manifest

pytestmark = pytest.mark.unit

EMBEDDING = Embedding(
    model="BAAI/bge-m3",
    revision="5617a9f61b028005a4858fdac845db406aefb181",
    batch_size=2,
)
VARIANT = "both_without_headings"


@pytest.fixture
def chunks(make_chunk):
    return [
        make_chunk(1, "التقادم خمس عشرة سنة"),
        make_chunk(2, "الأهلية إحدى وعشرون سنة"),
        make_chunk(3, "الفوائد سبعة في المائة", only_in_en=["Only English."]),
    ]


@pytest.fixture
def built(tmp_path, chunks, stub_embeddings):
    """The chunks embedded; chunk 2's text has its own direction in the stub space."""
    texts = [document_text(c, VARIANT) for c in chunks]
    embeddings = stub_embeddings({texts[1]: [0.0, 1.0, 0.0]}, default=[1.0, 0.0, 0.0])
    manifest = build_dense(
        chunks,
        tmp_path / "dense",
        embeddings=embeddings,
        embedding=EMBEDDING,
        document_text=VARIANT,
        chunks_sha256="0" * 64,
    )
    return tmp_path / "dense", manifest, embeddings, texts


def points(dense_dir):
    client = qdrant_client(dense_dir)
    try:
        records, _ = client.scroll(COLLECTION, limit=100, with_payload=True)
        return sorted(records, key=lambda r: r.payload["metadata"]["chunk_id"])
    finally:
        client.close()


# --- U29: every chunk indexed once, with its full payload ------------------------


def test_every_chunk_is_one_point_with_its_full_payload(built, chunks):
    dense_dir, _, _, texts = built

    records = points(dense_dir)

    assert [r.payload["metadata"] for r in records] == [
        c.model_dump(mode="json") for c in chunks
    ]
    assert [r.payload["page_content"] for r in records] == texts
    assert [str(r.id) for r in records] == [point_id(c.chunk_id) for c in chunks]
    assert count_points(dense_dir) == 3


def test_point_ids_are_stable_uuids_of_the_chunk_id():
    assert point_id("art-1-p1") == str(uuid.uuid5(uuid.NAMESPACE_URL, "art-1-p1"))
    assert point_id("art-1-p1") != point_id("art-1-p2")


def test_rebuilding_replaces_the_collection_instead_of_adding_to_it(
    built, chunks, stub_embeddings
):
    dense_dir, *_ = built

    build_dense(
        chunks[:2],
        dense_dir,
        embeddings=stub_embeddings({}, default=[1.0, 0.0, 0.0]),
        embedding=EMBEDDING,
        document_text=VARIANT,
        chunks_sha256="1" * 64,
    )

    assert count_points(dense_dir) == 2
    assert read_manifest(dense_dir, DenseManifest).chunk_ids == ["art-1-p1", "art-2-p1"]


def test_dense_search_finds_the_chunk_nearest_the_query(built):
    dense_dir, _, embeddings, texts = built
    client = qdrant_client(dense_dir)
    try:
        store = QdrantVectorStore(client, COLLECTION, embedding=embeddings)
        [(document, score)] = store.similarity_search_with_score(texts[1], k=1)
    finally:
        client.close()

    assert document.metadata["chunk_id"] == "art-2-p1"
    assert score == pytest.approx(1.0)


# --- The manifest ---------------------------------------------------------------


def test_the_manifest_records_what_the_vectors_were_built_from(built):
    dense_dir, manifest, *_ = built

    assert read_manifest(dense_dir, DenseManifest) == manifest
    assert manifest == DenseManifest(
        chunks_sha256="0" * 64,
        chunk_ids=["art-1-p1", "art-2-p1", "art-3-p1"],
        document_text=VARIANT,
        collection=COLLECTION,
        embedding_model=EMBEDDING.model,
        embedding_revision=EMBEDDING.revision,
        embedding_dim=3,
    )


def test_the_manifest_is_written_deterministically(built):
    dense_dir, *_ = built
    text = (dense_dir / "manifest.json").read_text("utf-8")

    assert text == json.dumps(json.loads(text), indent=2, sort_keys=True) + "\n"
