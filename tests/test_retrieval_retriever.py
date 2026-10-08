import pytest

from raglaw.config import Embedding, Retrieval, Search
from raglaw.retrieval.dense import build_dense, qdrant_client
from raglaw.retrieval.lexical import build_bm25
from raglaw.retrieval.retriever import HybridRetriever, IndexMismatchError
from raglaw.retrieval.tokenize import bm25_tokenizer

pytestmark = pytest.mark.unit

# Worked out by hand from conftest.SEARCH_VECTORS and the chunk texts.
DENSE_ORDER = ["art-3-p1", "art-1-p1", "art-4-p2", "art-4-p1", "art-2-p1"]
# RRF, k = 60: art-2 is last in dense (1/65) but first in BM25 (1/61).
HYBRID_ORDER = ["art-2-p1", "art-3-p1", "art-1-p1", "art-4-p2", "art-4-p1"]


def search(**overrides) -> Search:
    fields = {
        "mode": "hybrid",
        "article_lookup": True,
        "candidates": 50,
        "rrf_k": 60,
        "top_k": 10,
    }
    return Search(**(fields | overrides))


def open_retriever(index, embedding, retrieval=None, **overrides):
    return HybridRetriever.from_index(
        index.dense_dir,
        index.bm25_dir,
        embedding=embedding,
        retrieval=retrieval or index.retrieval,
        search=search(**overrides),
        embeddings=index.embeddings,
    )


def ids(documents):
    return [d.metadata["chunk_id"] for d in documents]


# --- Modes -------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        ("dense", DENSE_ORDER),
        ("bm25", ["art-2-p1"]),  # zero-score BM25 hits are dropped, not padded
        ("hybrid", HYBRID_ORDER),
    ],
)
def test_each_mode_ranks_by_its_own_retrievers(
    search_index, embedding_config, mode, expected
):
    with open_retriever(search_index, embedding_config, mode=mode) as retriever:
        assert ids(retriever.invoke("majority")) == expected


def test_top_k_cuts_the_fused_ranking(search_index, embedding_config):
    with open_retriever(search_index, embedding_config, top_k=2) as retriever:
        assert ids(retriever.invoke("majority")) == HYBRID_ORDER[:2]


def test_a_question_with_no_bm25_terms_falls_back_to_dense(
    search_index, embedding_config
):
    with open_retriever(search_index, embedding_config) as retriever:
        assert ids(retriever.invoke("؟")) == DENSE_ORDER


def test_each_hit_carries_its_chunk_and_how_it_was_found(
    search_index, embedding_config
):
    with open_retriever(search_index, embedding_config) as retriever:
        [first, second, *_] = retriever.invoke("majority")

    assert (
        first.page_content == "الأهلية إحدى وعشرون سنة\nMajority is twenty-one years."
    )
    assert first.metadata["citation"] == "Article 2"
    found = first.metadata["retrieval"]
    assert found["rank"] == 1
    assert found["dense_score"] == pytest.approx(0.0, abs=1e-6)
    assert found["bm25_score"] > 0
    assert found["rrf_score"] == pytest.approx(1 / 65 + 1 / 61)
    assert found["lookup"] is False
    assert second.metadata["retrieval"]["bm25_score"] is None  # BM25 didn't find it


def test_a_single_retriever_mode_records_no_fused_score(search_index, embedding_config):
    with open_retriever(search_index, embedding_config, mode="dense") as retriever:
        found = retriever.invoke("majority")[0].metadata["retrieval"]

    assert found["dense_score"] == pytest.approx(0.995, abs=1e-3)
    assert (found["bm25_score"], found["rrf_score"]) == (None, None)


def test_the_same_question_always_gives_the_same_ranking(
    search_index, embedding_config
):
    with open_retriever(search_index, embedding_config) as retriever:
        assert retriever.invoke("majority") == retriever.invoke("majority")


# --- Article lookup ----------------------------------------------------------------


def test_a_named_article_comes_first_with_all_its_parts(search_index, embedding_config):
    with open_retriever(search_index, embedding_config, top_k=3) as retriever:
        documents = retriever.invoke("What does Article 4 say about majority?")

    assert ids(documents) == ["art-4-p1", "art-4-p2", "art-2-p1"]
    assert [d.metadata["retrieval"]["lookup"] for d in documents] == [
        True,
        True,
        False,
    ]


def test_a_looked_up_article_is_never_listed_twice(search_index, embedding_config):
    with open_retriever(search_index, embedding_config) as retriever:
        found = ids(retriever.invoke("المادة ٤ majority"))

    assert found == ["art-4-p1", "art-4-p2", "art-2-p1", "art-3-p1", "art-1-p1"]


def test_an_article_not_in_the_index_is_ignored(search_index, embedding_config):
    with open_retriever(search_index, embedding_config) as retriever:
        assert ids(retriever.invoke("Article 999 majority")) == HYBRID_ORDER


def test_lookup_can_be_switched_off(search_index, embedding_config):
    with open_retriever(
        search_index, embedding_config, article_lookup=False
    ) as retriever:
        assert ids(retriever.invoke("Article 4 majority")) == HYBRID_ORDER


# --- Refusing an index built differently ---------------------------------------------


def test_a_different_embedding_model_is_refused(search_index, embedding_config):
    other = Embedding(model=embedding_config.model, revision="f" * 40, batch_size=2)

    with pytest.raises(IndexMismatchError, match="revision"):
        open_retriever(search_index, other)


def test_a_different_document_text_is_refused(search_index, embedding_config):
    configured = Retrieval(document_text="ar_only", bm25_tokenizer="words")

    with pytest.raises(IndexMismatchError, match="document_text"):
        open_retriever(search_index, embedding_config, configured)


def test_a_different_bm25_tokenizer_is_refused(search_index, embedding_config):
    configured = Retrieval(
        document_text=search_index.retrieval.document_text,
        bm25_tokenizer="words_light_stem",
    )

    with pytest.raises(IndexMismatchError, match="bm25_tokenizer"):
        open_retriever(search_index, embedding_config, configured)


def test_a_different_stopword_list_is_refused(search_index, embedding_config):
    configured = Retrieval(
        document_text=search_index.retrieval.document_text,
        bm25_tokenizer="words",
        bm25_stopwords="lucene",
    )

    with pytest.raises(IndexMismatchError, match="bm25_stopwords"):
        open_retriever(search_index, embedding_config, configured)


def test_halves_built_from_different_chunks_are_refused(search_index, embedding_config):
    build_bm25(
        search_index.chunks[:2],
        search_index.bm25_dir,
        tokenize=bm25_tokenizer("words"),
        retrieval=search_index.retrieval,
        embedding=embedding_config,
        chunks_sha256="1" * 64,
    )

    with pytest.raises(IndexMismatchError, match="different chunks"):
        open_retriever(search_index, embedding_config)


def test_closing_releases_the_qdrant_lock(search_index, embedding_config):
    open_retriever(search_index, embedding_config).close()

    qdrant_client(search_index.dense_dir).close()  # would raise if still locked


def test_a_refused_index_leaves_no_lock_behind(search_index, embedding_config):
    other = Embedding(model="other/model", revision="f" * 40, batch_size=2)
    with pytest.raises(IndexMismatchError):
        open_retriever(search_index, other)

    qdrant_client(search_index.dense_dir).close()


def test_lookup_of_an_article_inside_a_range_finds_the_range_chunk(
    tmp_path, make_chunk, stub_embeddings, embedding_config
):
    chunks = [
        make_chunk(1, "نص"),
        make_chunk(54, "ملغاة", chunk_id="art-54-56", range_end=56, is_repealed=True),
    ]
    retrieval = Retrieval(document_text="both_without_headings", bm25_tokenizer="words")
    embeddings = stub_embeddings({}, default=[1.0, 0.0])
    build_dense(
        chunks,
        tmp_path / "dense",
        embeddings=embeddings,
        embedding=embedding_config,
        document_text="both_without_headings",
        chunks_sha256="0" * 64,
    )
    build_bm25(
        chunks,
        tmp_path / "bm25",
        tokenize=bm25_tokenizer("words"),
        retrieval=retrieval,
        embedding=embedding_config,
        chunks_sha256="0" * 64,
    )

    with HybridRetriever.from_index(
        tmp_path / "dense",
        tmp_path / "bm25",
        embedding=embedding_config,
        retrieval=retrieval,
        search=search(),
        embeddings=embeddings,
    ) as retriever:
        [first, *_] = retriever.invoke("What does Article 55 say?")

    assert first.metadata["chunk_id"] == "art-54-56"
    assert first.metadata["retrieval"]["lookup"] is True
