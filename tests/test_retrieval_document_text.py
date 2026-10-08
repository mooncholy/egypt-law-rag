import pytest

from raglaw.retrieval.document_text import document_text
from raglaw.schema import Chunk

pytestmark = pytest.mark.unit


def chunk(**overrides) -> Chunk:
    fields = {
        "chunk_id": "art-601-p1",
        "article_number": 601,
        "citation": "Article 601 | المادة ٦٠١",
        "heading_path": ["Section I Leases", ""],
        "heading_path_ar": ["الفصل الأول الإيجار", "موت المستأجر أو إعساره"],
        "source_pages": [81],
        "is_repealed": False,
        "part_index": 1,
        "part_count": 1,
        "paragraphs": [],
        "text_ar": "لا ينتهي الإيجار بموت المؤجر.",
        "text_en": "A lease does not end on the death of the lessor.",
        "strategy": "structural",
    }
    return Chunk(**(fields | overrides))


def test_the_default_text_holds_both_heading_paths_and_both_languages():
    """D14: Arabic headings, Arabic text, English headings, English text, in that order."""
    assert document_text(chunk(), "both_with_headings") == (
        "الفصل الأول الإيجار > موت المستأجر أو إعساره\n"
        "لا ينتهي الإيجار بموت المؤجر.\n"
        "Section I Leases\n"
        "A lease does not end on the death of the lessor."
    )


def test_an_empty_heading_label_is_left_out():
    """Articles 601-609 sit under an Arabic-only heading; English gets no label there."""
    assert " > \n" not in document_text(chunk(), "both_with_headings")
    assert document_text(
        chunk(heading_path=["", ""]), "both_with_headings"
    ).splitlines()[2] == ("A lease does not end on the death of the lessor.")


def test_the_arabic_only_text_is_the_arabic_half():
    assert document_text(chunk(), "ar_only") == (
        "الفصل الأول الإيجار > موت المستأجر أو إعساره\nلا ينتهي الإيجار بموت المؤجر."
    )


def test_the_text_without_headings_holds_both_languages_only():
    assert document_text(chunk(), "both_without_headings") == (
        "لا ينتهي الإيجار بموت المؤجر.\n"
        "A lease does not end on the death of the lessor."
    )


def test_an_unknown_variant_is_rejected():
    with pytest.raises(ValueError, match="document text"):
        document_text(chunk(), "english_only")
