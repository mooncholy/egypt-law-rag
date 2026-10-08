"""The dense half of the index: chunk embeddings in a Qdrant collection.

``dense/qdrant/`` is a Qdrant collection in local mode (D13), one point per
chunk, keyed by a UUID derived from its ``chunk_id``. Points use LangChain's
layout: the document text as ``page_content`` and the whole ``Chunk`` record as
``metadata``, so a reader never needs ``chunks.json``. Serving it from a Qdrant
container later changes only the client.
"""

import uuid
from collections.abc import Sequence
from pathlib import Path

from langchain_core.embeddings import Embeddings
from langchain_qdrant import QdrantVectorStore
from pydantic import Field
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from raglaw.config import Embedding
from raglaw.retrieval.document_text import document_text as build_text
from raglaw.retrieval.manifest import IndexManifest, fresh_dir, write_manifest
from raglaw.schema import Chunk

COLLECTION = "civil_code"
QDRANT_DIR = "qdrant"


class DenseManifest(IndexManifest):
    """What the dense index was built from."""

    collection: str = Field(description="The Qdrant collection's name.")
    embedding_model: str = Field(
        description="Hugging Face id of the model that embedded the chunks; "
        "queries must use the same one."
    )
    embedding_revision: str = Field(description="The model's pinned commit (D12).")
    embedding_dim: int = Field(gt=0, description="Length of each dense vector.")


def point_id(chunk_id: str) -> str:
    """The Qdrant point id of a chunk: a UUID derived from ``chunk_id``, stable across builds."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


def qdrant_client(dense_dir: Path) -> QdrantClient:
    """
    Open the dense index's Qdrant collection in local mode.

    Local mode locks the directory, so close the client when done.

    returns:
    - client (QdrantClient): on ``dense_dir/qdrant``
    """
    return QdrantClient(path=str(dense_dir / QDRANT_DIR))


def build_dense(
    chunks: Sequence[Chunk],
    dense_dir: Path,
    *,
    embeddings: Embeddings,
    embedding: Embedding,
    document_text: str,
    chunks_sha256: str,
) -> DenseManifest:
    """
    Embed every chunk's document text into a new collection in ``dense_dir``.

    returns:
    - manifest (DenseManifest): what was built, also written to
      ``dense_dir/manifest.json``
    """
    fresh_dir(dense_dir)
    texts = [build_text(c, document_text) for c in chunks]
    dim = len(embeddings.embed_documents(texts[:1])[0])
    client = qdrant_client(dense_dir)
    try:
        client.create_collection(
            COLLECTION, vectors_config=VectorParams(size=dim, distance=Distance.COSINE)
        )
        QdrantVectorStore(client, COLLECTION, embedding=embeddings).add_texts(
            texts,
            metadatas=[c.model_dump(mode="json") for c in chunks],
            ids=[point_id(c.chunk_id) for c in chunks],
            batch_size=embedding.batch_size,
        )
    finally:
        client.close()
    manifest = DenseManifest(
        chunks_sha256=chunks_sha256,
        chunk_ids=[c.chunk_id for c in chunks],
        document_text=document_text,
        collection=COLLECTION,
        embedding_model=embedding.model,
        embedding_revision=embedding.revision,
        embedding_dim=dim,
    )
    write_manifest(dense_dir, manifest)
    return manifest


def count_points(dense_dir: Path) -> int:
    """
    Count the points in the dense collection.

    returns:
    - count (int): exact, not estimated
    """
    client = qdrant_client(dense_dir)
    try:
        return client.count(COLLECTION, exact=True).count
    finally:
        client.close()
