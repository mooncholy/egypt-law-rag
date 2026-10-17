import pytest

from raglaw.config import Embedding, Retrieval
from raglaw.retrieval.lexical import Bm25Manifest, build_bm25, load_bm25
from raglaw.retrieval.manifest import read_manifest
from raglaw.retrieval.tokenize import bm25_tokenizer

pytestmark = pytest.mark.unit

EMBEDDING = Embedding(
    model="BAAI/bge-m3",
    revision="5617a9f61b028005a4858fdac845db406aefb181",
    batch_size=2,
)
WORDS = Retrieval(document_text="both_without_headings", bm25_tokenizer="words")


def build(tmp_path, chunks, retrieval=WORDS, tokenize=None):
    return build_bm25(
        chunks,
        tmp_path / "bm25",
        tokenize=tokenize or bm25_tokenizer(retrieval.bm25_tokenizer),
        retrieval=retrieval,
        embedding=EMBEDDING,
        chunks_sha256="0" * 64,
    )


def test_bm25_finds_the_chunk_holding_a_folded_term(tmp_path, lexical_chunks):
    """`الاهليه` (no hamza, ha for ta marbuta) matches `الأهلية` once folded."""
    manifest = build(tmp_path, lexical_chunks)

    ids, scores = load_bm25(tmp_path / "bm25").retrieve(
        [bm25_tokenizer("words")("الاهليه")], k=1, show_progress=False
    )

    assert manifest.chunk_ids[ids[0][0]] == "art-2-p1"
    assert scores[0][0] > 0


def test_the_manifest_records_the_tokenizer_and_vocabulary(tmp_path, lexical_chunks):
    manifest = build(tmp_path, lexical_chunks)

    assert read_manifest(tmp_path / "bm25", Bm25Manifest) == manifest
    assert manifest.bm25_tokenizer == "words"
    assert (manifest.tokenizer_model, manifest.tokenizer_revision) == (None, None)
    # 4 Arabic words + 1 English word per chunk; سنة is shared: 9 distinct terms
    assert manifest.vocabulary_size == 9
    assert (manifest.bm25_k1, manifest.bm25_b, manifest.bm25_method) == (
        1.5,
        0.75,
        "lucene",
    )


def test_model_subwords_records_the_model_whose_tokenizer_made_the_terms(
    tmp_path, lexical_chunks, stub_tokenizer
):
    subwords = Retrieval(
        document_text="both_without_headings", bm25_tokenizer="model_subwords"
    )

    manifest = build(
        tmp_path,
        lexical_chunks,
        subwords,
        bm25_tokenizer("model_subwords", stub_tokenizer()),
    )

    assert (manifest.tokenizer_model, manifest.tokenizer_revision) == (
        EMBEDDING.model,
        EMBEDDING.revision,
    )


def test_rebuilding_replaces_the_index(tmp_path, lexical_chunks):
    build(tmp_path, lexical_chunks)

    manifest = build(tmp_path, lexical_chunks[:1])

    assert read_manifest(tmp_path / "bm25", Bm25Manifest) == manifest
    assert manifest.chunk_ids == ["art-1-p1"]


def test_the_manifest_records_the_stopword_list(tmp_path, lexical_chunks):
    nltk = Retrieval(
        document_text="both_without_headings",
        bm25_tokenizer="words",
        bm25_stopwords="nltk",
    )

    manifest = build(
        tmp_path, lexical_chunks, nltk, bm25_tokenizer("words", stopwords="nltk")
    )

    assert manifest.bm25_stopwords == "nltk"
