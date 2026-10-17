import pytest

from raglaw.config import Retrieval, Search, Settings
from raglaw.ingest.reindex import ReindexError, reindex
from raglaw.records import write_records
from raglaw.retrieval.retriever import HybridRetriever
from raglaw.schema import Chunk

pytestmark = pytest.mark.unit

RETRIEVAL = Retrieval(document_text="both_without_headings", bm25_tokenizer="words")


def index_of(make_chunk, tmp_path):
    path = tmp_path / "chunks.json"
    write_records(path, Chunk, [make_chunk(n) for n in (1, 2, 3)])
    return path


def test_u39_a_new_article_is_indexed_and_found_by_its_number(
    make_chunk, make_article, stub_embeddings, embedding_config, tmp_path
):
    """A two-paragraph article over max_chars becomes two chunks."""
    long = "(١) " + "أ" * 900 + "\n(٢) " + "ب" * 900
    embeddings = stub_embeddings({}, default=[1.0, 0.0])

    result = reindex(
        index_of(make_chunk, tmp_path),
        [make_article(1150, long)],
        tmp_path / "reindexed",
        chunking=Settings(_env_file=None).chunking,
        embedding=embedding_config,
        retrieval=RETRIEVAL,
        embeddings=embeddings,
    )

    assert (result.documents_before, result.documents_after) == (3, 5)
    assert result.added_articles == [1150]
    search = Search(
        mode="hybrid", article_lookup=True, candidates=10, rrf_k=60, top_k=5
    )
    with HybridRetriever.from_index(
        tmp_path / "reindexed" / "dense",
        tmp_path / "reindexed" / "bm25",
        embedding=embedding_config,
        retrieval=RETRIEVAL,
        search=search,
        embeddings=embeddings,
    ) as retriever:
        found = retriever.invoke("What does Article 1150 say?")
    assert found[0].metadata["article_number"] == 1150


def test_an_article_already_indexed_is_refused(
    make_chunk, make_article, stub_embeddings, embedding_config, tmp_path
):
    with pytest.raises(ReindexError, match=r"\[2\]"):
        reindex(
            index_of(make_chunk, tmp_path),
            [make_article(2)],
            tmp_path / "reindexed",
            chunking=Settings(_env_file=None).chunking,
            embedding=embedding_config,
            retrieval=RETRIEVAL,
            embeddings=stub_embeddings({}, default=[1.0, 0.0]),
        )


@pytest.mark.parametrize(
    ("indexed", "new", "why"),
    [
        ([1, (54, 80)], [60], r"\[60\] are already"),  # inside a repealed range
        ([1, 2], [1150, 1150], r"\[1150\] appear more than once"),
    ],
)
def test_a_number_covered_by_a_range_or_repeated_is_refused(
    make_chunk,
    make_article,
    stub_embeddings,
    embedding_config,
    tmp_path,
    indexed,
    new,
    why,
):
    chunks = [
        make_chunk(n) if isinstance(n, int) else make_chunk(n[0], range_end=n[1])
        for n in indexed
    ]
    write_records(tmp_path / "chunks.json", Chunk, chunks)

    with pytest.raises(ReindexError, match=why):
        reindex(
            tmp_path / "chunks.json",
            [make_article(n) for n in new],
            tmp_path / "reindexed",
            chunking=Settings(_env_file=None).chunking,
            embedding=embedding_config,
            retrieval=RETRIEVAL,
            embeddings=stub_embeddings({}, default=[1.0, 0.0]),
        )


def test_the_semantic_strategy_gets_the_embedder(
    make_chunk, make_article, stub_embeddings, embedding_config, tmp_path
):
    """structural_semantic splits by meaning, so it needs the embedder."""
    embeddings = stub_embeddings({}, default=[1.0, 0.0])
    chunking = Settings(_env_file=None).chunking.model_copy(
        update={"strategy": "structural_semantic"}
    )

    result = reindex(
        index_of(make_chunk, tmp_path),
        [make_article(1150)],
        tmp_path / "reindexed",
        chunking=chunking,
        embedding=embedding_config,
        retrieval=RETRIEVAL,
        embeddings=embeddings,
    )

    assert result.added_articles == [1150]
