"""The lexical half of the index: BM25 over the chunks' document texts.

``bm25/`` holds a ``bm25s`` index next to its manifest. bm25s knows chunks only
by position, which ``manifest.chunk_ids`` maps back to ids. Terms come from the
``retrieval.bm25_tokenizer`` choice (``raglaw.retrieval.tokenize``).
"""

from collections.abc import Callable, Sequence
from pathlib import Path

import bm25s
from pydantic import Field

from raglaw.config import Embedding, Retrieval
from raglaw.retrieval.document_text import document_text as build_text
from raglaw.retrieval.manifest import IndexManifest, fresh_dir, write_manifest
from raglaw.schema import Chunk

# bm25s defaults, written to the manifest so a change is visible.
BM25_K1, BM25_B, BM25_METHOD = 1.5, 0.75, "lucene"
# bm25s adds this placeholder term to every vocabulary, so an empty query works.
EMPTY_TERM = ""


class Bm25Manifest(IndexManifest):
    """What the BM25 index was built from."""

    bm25_tokenizer: str = Field(
        description="The `retrieval.bm25_tokenizer` the terms came from; queries "
        "must be tokenized the same way."
    )
    tokenizer_model: str | None = Field(
        description="For `model_subwords`, the model whose tokenizer made the "
        "terms; None for the word tokenizers."
    )
    tokenizer_revision: str | None = Field(
        description="That model's pinned commit; None for the word tokenizers."
    )
    vocabulary_size: int = Field(ge=0, description="Distinct terms in the index.")
    bm25_k1: float = Field(
        description="Term-frequency saturation: how quickly repeats of a term "
        "stop adding to the score."
    )
    bm25_b: float = Field(
        description="Length normalization: 0 ignores document length, 1 fully "
        "normalizes by it."
    )
    bm25_method: str = Field(description="The bm25s scoring variant (e.g., `lucene`).")


def build_bm25(
    chunks: Sequence[Chunk],
    bm25_dir: Path,
    *,
    tokenize: Callable[[str], list[str]],
    retrieval: Retrieval,
    embedding: Embedding,
    chunks_sha256: str,
) -> Bm25Manifest:
    """
    BM25-index every chunk's document text into ``bm25_dir``, replacing any index there.

    ``embedding`` names the model whose tokenizer ``model_subwords`` uses.

    returns:
    - manifest (Bm25Manifest): what was built, also written to
      ``bm25_dir/manifest.json``
    """
    fresh_dir(bm25_dir)
    terms = [tokenize(build_text(c, retrieval.document_text)) for c in chunks]
    bm25 = bm25s.BM25(k1=BM25_K1, b=BM25_B, method=BM25_METHOD)
    bm25.index(terms, show_progress=False)
    bm25.save(str(bm25_dir), show_progress=False)
    subwords = retrieval.bm25_tokenizer == "model_subwords"
    manifest = Bm25Manifest(
        chunks_sha256=chunks_sha256,
        chunk_ids=[c.chunk_id for c in chunks],
        document_text=retrieval.document_text,
        bm25_tokenizer=retrieval.bm25_tokenizer,
        tokenizer_model=embedding.model if subwords else None,
        tokenizer_revision=embedding.revision if subwords else None,
        vocabulary_size=len(set(bm25.vocab_dict) - {EMPTY_TERM}),
        bm25_k1=BM25_K1,
        bm25_b=BM25_B,
        bm25_method=BM25_METHOD,
    )
    write_manifest(bm25_dir, manifest)
    return manifest


def load_bm25(bm25_dir: Path) -> bm25s.BM25:
    """
    Load the BM25 index; its results are positions in ``manifest.chunk_ids``.

    returns:
    - bm25 (bm25s.BM25): ready to ``retrieve``
    """
    return bm25s.BM25.load(str(bm25_dir), show_progress=False)
