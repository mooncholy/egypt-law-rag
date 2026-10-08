"""Hybrid retrieval: a named article first, then dense and BM25 fused by RRF.

For one question, ``HybridRetriever``:

1. fetches every part of each article the question names by number
   (``article_references``), when ``search.article_lookup`` is on;
2. ranks chunks by the dense index (cosine to the query's embedding), by BM25
   (the query tokenized as the index was), or both fused by reciprocal rank
   fusion, as ``search.mode`` says;
3. returns the looked-up parts, then the ranked chunks not already listed, cut
   to ``search.top_k``.

Opening an index checks both manifests against each other and against the
config, so a query is never embedded or tokenized differently from the
documents. Every payload is read into memory once (the index is small), so a
BM25 or lookup hit needs no further Qdrant call.
"""

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any, Self

import bm25s
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever
from langchain_qdrant import QdrantVectorStore
from pydantic import ConfigDict, PrivateAttr
from qdrant_client import QdrantClient

from raglaw.config import Embedding, Retrieval, Search
from raglaw.retrieval.dense import COLLECTION, DenseManifest, qdrant_client
from raglaw.retrieval.embeddings import ModelTokenizer
from raglaw.retrieval.fusion import reciprocal_rank_fusion
from raglaw.retrieval.lexical import Bm25Manifest, load_bm25
from raglaw.retrieval.lookup import article_references
from raglaw.retrieval.manifest import read_manifest
from raglaw.retrieval.tokenize import bm25_tokenizer

logger = logging.getLogger(__name__)

REBUILD = "rebuild it with `dvc repro embed bm25`"


class IndexMismatchError(ValueError):
    """The index was built from other chunks, or differently from the config."""


def check_manifests(
    dense: DenseManifest,
    lexical: Bm25Manifest,
    embedding: Embedding,
    retrieval: Retrieval,
) -> None:
    """
    Check that both index halves cover the same chunks and match the config.

    exceptions:
    - IndexMismatchError: naming every difference found
    """
    problems = []
    if (dense.chunks_sha256, dense.chunk_ids) != (
        lexical.chunks_sha256,
        lexical.chunk_ids,
    ):
        problems.append("the dense and BM25 indexes were built from different chunks")
    built_model = (dense.embedding_model, dense.embedding_revision)
    if built_model != (embedding.model, embedding.revision):
        problems.append(
            f"embedded with {built_model[0]} at revision {built_model[1]}, but the "
            f"config names {embedding.model} at {embedding.revision}"
        )
    for name, built in (("dense", dense), ("BM25", lexical)):
        if built.document_text != retrieval.document_text:
            problems.append(
                f"the {name} index holds document_text={built.document_text!r}, "
                f"the config {retrieval.document_text!r}"
            )
    if lexical.bm25_tokenizer != retrieval.bm25_tokenizer:
        problems.append(
            f"BM25 was built with bm25_tokenizer={lexical.bm25_tokenizer!r}, the "
            f"config names {retrieval.bm25_tokenizer!r}"
        )
    elif lexical.bm25_tokenizer == "model_subwords" and (
        lexical.tokenizer_model,
        lexical.tokenizer_revision,
    ) != (embedding.model, embedding.revision):
        problems.append("BM25's subword terms came from another model's tokenizer")
    if problems:
        raise IndexMismatchError(f"{'; '.join(problems)}; {REBUILD}")


class HybridRetriever(BaseRetriever):
    """Search both index halves for a question; a LangChain retriever.

    Open it with ``from_index`` and close it when done (or use ``with``): Qdrant
    local mode locks the index directory while a client holds it.

    Each returned ``Document`` is a chunk: its document text as
    ``page_content``, its ``Chunk`` record as metadata, plus
    ``metadata["retrieval"]``: ``rank`` (1-based), ``dense_score`` (cosine),
    ``bm25_score`` and ``rrf_score`` (None where that list didn't hold it, or
    the mode doesn't use it), and ``lookup`` (fetched by article number).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    search: Search

    _client: QdrantClient = PrivateAttr()
    _store: QdrantVectorStore = PrivateAttr()
    _bm25: bm25s.BM25 = PrivateAttr()
    _tokenize: Callable[[str], list[str]] = PrivateAttr()
    _chunk_ids: list[str] = PrivateAttr()
    _payloads: dict[str, tuple[str, dict[str, Any]]] = PrivateAttr()
    _parts: dict[int, list[str]] = PrivateAttr()

    @classmethod
    def from_index(
        cls,
        dense_dir: Path,
        bm25_dir: Path,
        *,
        embedding: Embedding,
        retrieval: Retrieval,
        search: Search,
        embeddings: Embeddings,
        tokenizer: ModelTokenizer | None = None,
    ) -> Self:
        """
        Open both index halves after checking them against the config.

        ``embeddings`` must be the model the index was built with;
        ``tokenizer`` (the model's own) is needed only for ``model_subwords``.

        returns:
        - retriever (HybridRetriever): holding the Qdrant lock until ``close``

        exceptions:
        - IndexMismatchError: see ``check_manifests``; also when the
          collection's points differ from the manifest's chunks
        """
        dense = read_manifest(dense_dir, DenseManifest)
        lexical = read_manifest(bm25_dir, Bm25Manifest)
        check_manifests(dense, lexical, embedding, retrieval)
        retriever = cls(search=search)
        retriever._tokenize = bm25_tokenizer(retrieval.bm25_tokenizer, tokenizer)
        retriever._bm25 = load_bm25(bm25_dir)
        retriever._chunk_ids = dense.chunk_ids
        client = qdrant_client(dense_dir)
        try:
            retriever._load_payloads(client, dense.chunk_ids)
        except Exception:
            client.close()
            raise
        retriever._client = client
        retriever._store = QdrantVectorStore(client, COLLECTION, embedding=embeddings)
        return retriever

    def _load_payloads(self, client: QdrantClient, chunk_ids: list[str]) -> None:
        records, _ = client.scroll(
            COLLECTION, limit=len(chunk_ids) + 1, with_payload=True, with_vectors=False
        )
        payloads = {
            r.payload["metadata"]["chunk_id"]: (
                r.payload["page_content"],
                r.payload["metadata"],
            )
            for r in records
        }
        if sorted(payloads) != sorted(chunk_ids):
            raise IndexMismatchError(
                f"the collection holds {len(payloads)} chunks, the manifest "
                f"{len(chunk_ids)}; {REBUILD}"
            )
        parts: dict[int, list[tuple[int, str]]] = {}
        for chunk_id, (_, meta) in payloads.items():
            parts.setdefault(meta["article_number"], []).append(
                (meta["part_index"], chunk_id)
            )
        self._payloads = payloads
        self._parts = {n: [c for _, c in sorted(ps)] for n, ps in parts.items()}

    def close(self) -> None:
        """Release the Qdrant client and its directory lock."""
        self._client.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _dense(self, query: str) -> list[tuple[str, float]]:
        hits = self._store.similarity_search_with_score(query, k=self.search.candidates)
        return [(d.metadata["chunk_id"], float(score)) for d, score in hits]

    def _lexical(self, query: str) -> list[tuple[str, float]]:
        terms = self._tokenize(query)
        if not terms:
            return []
        k = min(self.search.candidates, len(self._chunk_ids))
        positions, scores = self._bm25.retrieve([terms], k=k, show_progress=False)
        # bm25s always returns k hits; those that share no term score 0.
        return [
            (self._chunk_ids[i], float(s))
            for i, s in zip(positions[0], scores[0], strict=True)
            if s > 0
        ]

    def _looked_up(self, query: str) -> list[str]:
        if not self.search.article_lookup:
            return []
        found = []
        for number in article_references(query):
            if number in self._parts:
                found += self._parts[number]
            else:
                logger.debug("Article %d is named but not in the index", number)
        return found

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        mode = self.search.mode
        dense = self._dense(query) if mode in ("dense", "hybrid") else []
        lexical = self._lexical(query) if mode in ("bm25", "hybrid") else []
        rrf: dict[str, float] = {}
        if mode == "hybrid":
            fused = reciprocal_rank_fusion(
                [[c for c, _ in dense], [c for c, _ in lexical]], k=self.search.rrf_k
            )
            rrf = dict(fused)
            ranked = [c for c, _ in fused]
        else:
            ranked = [c for c, _ in (dense if mode == "dense" else lexical)]
        looked_up = self._looked_up(query)
        order = list(dict.fromkeys(looked_up + ranked))[: self.search.top_k]
        dense_scores, bm25_scores = dict(dense), dict(lexical)
        documents = []
        for rank, chunk_id in enumerate(order, start=1):
            text, meta = self._payloads[chunk_id]
            found = {
                "rank": rank,
                "dense_score": dense_scores.get(chunk_id),
                "bm25_score": bm25_scores.get(chunk_id),
                "rrf_score": rrf.get(chunk_id),
                "lookup": chunk_id in looked_up,
            }
            documents.append(
                Document(
                    id=chunk_id,
                    page_content=text,
                    metadata={**meta, "retrieval": found},
                )
            )
        return documents
