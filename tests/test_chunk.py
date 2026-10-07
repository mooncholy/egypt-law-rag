import sys
from pathlib import Path

import pytest

from raglaw.config import Chunking, Semantic
from raglaw.ingest.chunk import (
    ChunkError,
    ParagraphSemanticSplitter,
    chunk_documents,
    opening_connective,
    paragraph_numbers,
    run_chunk,
    split_articles_report,
    split_paragraphs,
)
from raglaw.ingest.loader import CivilCodeArticleLoader
from raglaw.records import read_records, write_records
from raglaw.schema import Article, Chunk, LogEvent

SEMANTIC = Semantic(
    model="BAAI/bge-m3",
    revision="5617a9f61b028005a4858fdac845db406aefb181",
    breakpoint_percentile=90,
    batch_size=8,
)


def chunking(strategy: str = "structural", max_chars: int = 1000) -> Chunking:
    return Chunking(strategy=strategy, max_chars=max_chars, semantic=SEMANTIC)


def article(n: int, text_ar: str, **overrides) -> Article:
    fields = {
        "article_number": n,
        "text_en": f"English text of article {n}.",
        "text_ar": text_ar,
        "heading_path": ["Root", "SECTION I"],
        "heading_path_ar": ["الجذر", "الفصل الأول"],
        "source_pages": [3],
        "source_rows": [(3, 1)],
    }
    return Article(**(fields | overrides))


def documents(tmp_path: Path, *articles: Article):
    path = tmp_path / "articles.json"
    write_records(path, Article, articles)
    return CivilCodeArticleLoader(path).load()


# --- Paragraph markers (U17) ----------------------------------------------------


@pytest.mark.unit
def test_paragraphs_split_before_each_marker_shape():
    """Marker shapes after the right-to-left repairs: )١(  (٢(  ( ٣(  (٤.)"""
    text = ")١( الأول\nتتمة\n(٢( الثاني\n( ٣( الثالث\n(٤.) الرابع"

    assert split_paragraphs(text) == [
        ")١( الأول\nتتمة",
        "(٢( الثاني",
        "( ٣( الثالث",
        "(٤.) الرابع",
    ]
    assert paragraph_numbers(text) == [1, 2, 3, 4]


@pytest.mark.unit
def test_a_marker_inside_a_line_does_not_split():
    assert split_paragraphs("كما في الفقرة (٢) من المادة") == [
        "كما في الفقرة (٢) من المادة"
    ]


# --- Structural packing (U18, U19) -------------------------------------------------


@pytest.mark.unit
def test_a_short_article_is_one_chunk(tmp_path):
    chunks, metrics = chunk_documents(
        documents(tmp_path, article(1, "نص قصير.")), chunking()
    )

    assert [c.chunk_id for c in chunks] == ["art-1-p1"]
    assert metrics["split_articles"] == 0


@pytest.mark.unit
def test_a_long_article_splits_only_at_paragraph_boundaries(tmp_path):
    paragraphs = [f"({d}( " + "كلمة " * 30 for d in "١٢٣٤"]  # about 155 chars each
    text = "\n".join(p.strip() for p in paragraphs)

    chunks, _ = chunk_documents(
        documents(tmp_path, article(7, text)), chunking(max_chars=350)
    )

    assert [c.paragraphs for c in chunks] == [[1, 2], [3, 4]]
    assert all(len(c.text_ar) <= 350 for c in chunks)
    assert "".join(c.text_ar for c in chunks).replace("\n", "") == text.replace(
        "\n", ""
    )
    assert [(c.part_index, c.part_count) for c in chunks] == [(1, 2), (2, 2)]


@pytest.mark.unit
def test_an_oversize_paragraph_stays_whole_and_is_logged(tmp_path, log_records):
    """R22: a paragraph longer than max_chars is never cut."""
    text = "(١( " + "كلمة " * 100

    chunks, metrics = chunk_documents(
        documents(tmp_path, article(9, text)), chunking(max_chars=200)
    )

    assert len(chunks) == 1 and len(chunks[0].text_ar) > 200
    assert metrics["oversize_paragraph_anomalies"] == 1
    [anomaly] = log_records(event_type=LogEvent.ANOMALY)
    assert (anomaly["anomaly_type"], anomaly["article_number"]) == (
        "oversize_paragraph",
        9,
    )


# --- What every part carries (U20, U21, R28) -----------------------------------------


@pytest.mark.unit
def test_every_part_carries_the_article_context(tmp_path):
    text = "\n".join(f"({d}( " + "كلمة " * 30 for d in "١٢")
    source = article(
        1021,
        text,
        only_in_ar=["(٢( " + "كلمة " * 29 + "كلمة"],
        only_in_en=["Only English."],
    )

    chunks, _ = chunk_documents(documents(tmp_path, source), chunking(max_chars=200))

    assert len(chunks) == 2
    for c in chunks:
        assert c.article_number == 1021
        assert c.citation == "Article 1021 | المادة ١٠٢١"
        assert c.text_en == "English text of article 1021."  # D4: the full English
        assert c.heading_path_ar == ["الجذر", "الفصل الأول"]
        assert c.only_in_en == ["Only English."]
    assert [bool(c.only_in_ar) for c in chunks] == [
        False,
        True,
    ]  # only the part holding it


@pytest.mark.unit
def test_a_repealed_article_is_one_chunk(tmp_path):
    note = "المواد من ٥٤ إلى ٨٠ ملغاة\n(١( " + "كلمة " * 300
    chunks, _ = chunk_documents(
        documents(tmp_path, article(54, note, is_repealed=True)),
        chunking(max_chars=100),
    )

    assert [(c.chunk_id, c.is_repealed) for c in chunks] == [("art-54-p1", True)]


@pytest.mark.unit
def test_lost_text_stops_the_stage(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "langchain_text_splitters.RecursiveCharacterTextSplitter.split_text",
        lambda self, text: [text[:5]],
    )

    with pytest.raises(ChunkError, match="rebuild"):
        chunk_documents(
            documents(tmp_path, article(1, "نص أطول من خمسة أحرف")), chunking()
        )


# --- The semantic variant (U22, U23) ---------------------------------------------------


@pytest.mark.unit
def test_the_threshold_is_set_over_the_whole_corpus(stub_embeddings):
    """A per-article percentile would split every 3-paragraph article at its largest gap."""
    near, far = [1.0, 0.0], [0.0, 1.0]
    texts = [
        "(١( أ\n(٢( ب\n(٣( ج",  # three paragraphs, all close in meaning
        "(١( د\n(٢( هـ",  # two paragraphs, far apart
    ]
    embeddings = stub_embeddings({"(٢( هـ": far}, default=near)
    splitter = ParagraphSemanticSplitter(
        embeddings, breakpoint_percentile=50, max_chars=1000
    )

    threshold = splitter.fit(texts)

    assert threshold == 0.0  # the median of the corpus's distances [0, 0, 1]
    assert splitter.split_text(texts[0]) == [texts[0]]  # ordinary distances: no split
    assert splitter.split_text(texts[1]) == ["(١( د", "(٢( هـ"]  # meaning shifts: split
    assert splitter.breakpoints == 1


@pytest.mark.unit
def test_semantic_chunking_is_deterministic(tmp_path, stub_embeddings):
    docs = documents(tmp_path, article(1, "(١( أ\n(٢( ب"), article(2, "(١( ج\n(٢( د"))
    embeddings = stub_embeddings({"(٢( د": [0.0, 1.0]}, default=[1.0, 0.0])

    first, _ = chunk_documents(docs, chunking("structural_semantic"), embeddings)
    second, _ = chunk_documents(docs, chunking("structural_semantic"), embeddings)

    assert first == second


@pytest.mark.unit
def test_the_structural_strategy_never_imports_the_model(tmp_path):
    chunk_documents(documents(tmp_path, article(1, "نص.")), chunking())

    assert "langchain_huggingface" not in sys.modules


# --- Quality, recorded per run --------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize(
    ("paragraph", "connective"),
    [
        ("(٢( ومع ذلك فحقوق الحمل المستكن يعينها القانون.", "ومع ذلك"),
        (") ٢( على أنه إذا اشترط الدفع مقدماً", "على أن"),
        (")٣( فإذا كان الحائز شخصا معنويا", "فإذا"),
        ("(١( تبدأ شخصية الإنسان بتمام ولادته حيا", None),
    ],
)
def test_a_paragraph_tied_to_the_one_before_is_recognized(paragraph, connective):
    assert opening_connective(paragraph) == connective


@pytest.mark.unit
def test_quality_metrics_measure_split_parts_and_their_ties(tmp_path, stub_embeddings):
    """Article 29: its second paragraph is an exception to the first."""
    rule = "(١( تبدأ شخصية الإنسان بتمام ولادته حيا، وتنتهي بموته."
    exception = "(٢( ومع ذلك فحقوق الحمل المستكن يعينها القانون."
    # Article 31's two paragraphs are close in meaning, so the corpus threshold
    # sits below the distance between Article 29's rule and its exception.
    docs = documents(
        tmp_path, article(29, f"{rule}\n{exception}"), article(31, "(١( أ\n(٢( ب")
    )
    embeddings = stub_embeddings({exception: [0.0, 1.0]}, default=[1.0, 0.0])

    _, metrics = chunk_documents(docs, chunking("structural_semantic"), embeddings)

    assert metrics["split_parts"] == 2
    assert metrics["split_parts_with_connective"] == 1
    assert metrics["split_parts_with_connective_share"] == 1.0
    assert metrics["split_parts_under_100_chars"] == 2
    assert (
        metrics["paragraphs_with_connective_share"] == 0.5
    )  # 1 of the 2 later paragraphs


@pytest.mark.unit
def test_structural_chunks_record_no_split_parts(tmp_path):
    _, metrics = chunk_documents(
        documents(tmp_path, article(1, "نص قصير.")), chunking()
    )

    assert metrics["split_parts"] == 0
    assert metrics["split_parts_with_connective_share"] == 0.0
    assert metrics["median_chunk_chars"] == len("نص قصير.")


@pytest.mark.unit
def test_split_articles_report_lists_each_part(tmp_path, stub_embeddings):
    exception = "(٢( ومع ذلك فحقوق الحمل المستكن يعينها القانون."
    docs = documents(
        tmp_path,
        article(29, f"(١( تبدأ شخصية الإنسان.\n{exception}"),
        article(31, "(١( أ\n(٢( ب"),
    )
    embeddings = stub_embeddings({exception: [0.0, 1.0]}, default=[1.0, 0.0])
    chunks, _ = chunk_documents(docs, chunking("structural_semantic"), embeddings)

    report = split_articles_report(chunks)

    assert "# Split articles (structural_semantic): 1" in report
    assert "## Article 29" in report
    assert "opens with ومع ذلك" in report


# --- The stage ---------------------------------------------------------------------------


@pytest.mark.unit
def test_the_stage_writes_chunks_and_metrics(tmp_path):
    articles = tmp_path / "articles.json"
    write_records(articles, Article, [article(1, "نص."), article(2, "نص آخر.")])

    metrics = run_chunk(
        articles,
        tmp_path / "chunks.json",
        tmp_path / "m.json",
        chunking(),
        report_out=tmp_path / "split_articles.md",
    )

    assert [c.chunk_id for c in read_records(tmp_path / "chunks.json", Chunk)] == [
        "art-1-p1",
        "art-2-p1",
    ]
    assert metrics["articles_covered"] == 2
    assert len(metrics["chunks_sha256"]) == 64
    assert (tmp_path / "split_articles.md").read_text().startswith("# Split articles")


# --- Corpus: the real articles (C15) -------------------------------------------------------


@pytest.mark.corpus
def test_every_article_is_chunked_within_max_chars(stage_metrics):
    metrics = stage_metrics("chunks")

    assert metrics["articles_covered"] == 1_149
    assert metrics["chunks_with_article_number"] == metrics["chunks"]
    assert (
        metrics["max_chunk_chars"] <= 1_000
        or metrics["oversize_paragraph_anomalies"] > 0
    )
    assert metrics["chunks_with_untranslated_text"] == 7
