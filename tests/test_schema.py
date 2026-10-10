from typing import Any

import pytest
from pydantic import Field, ValidationError

from raglaw.records import SchemaMismatchError, read_records, write_records
from raglaw.schema import Article, Record, Row

pytestmark = pytest.mark.unit


class Sample(Record):
    MODEL_NAME = "sample"

    page: int = Field(ge=1, description="Source page.")
    text: str = Field(description="Cell text.")


def article(**overrides: Any) -> Article:
    fields = {
        "article_number": 1,
        "text_en": "Legislative provisions govern ...",
        "text_ar": "تسري النصوص التشريعية ...",
        "heading_path": ["TODO", "SECTION I Laws and their Applications"],
        "heading_path_ar": ["باب تمهيدي / أحكام عامة", "الفصل الأول"],
        "source_pages": [1],
        "source_rows": [(1, 2)],
    }
    return Article(**(fields | overrides))


# --- The fingerprint ----------------------------------------------------------


def test_rewording_or_reordering_fields_keeps_the_fingerprint():
    class Reworded(Record):
        MODEL_NAME = "sample"

        text: str = Field(description="The cell's text, lines joined.")
        page: int = Field(ge=1, description="1-based page.")

    assert Reworded.schema_fingerprint() == Sample.schema_fingerprint()


def test_adding_a_field_changes_the_fingerprint():
    class Added(Sample):
        bold: bool = False

    assert Added.schema_fingerprint() != Sample.schema_fingerprint()


def test_changing_a_type_or_constraint_changes_the_fingerprint():
    class Retyped(Record):
        MODEL_NAME = "sample"

        page: str
        text: str

    class Reconstrained(Record):
        MODEL_NAME = "sample"

        page: int = Field(ge=0)
        text: str

    fingerprints = {
        Sample.schema_fingerprint(),
        Retyped.schema_fingerprint(),
        Reconstrained.schema_fingerprint(),
    }
    assert len(fingerprints) == 3


# --- Files ------------------------------------------------------------------


@pytest.mark.parametrize("suffix", [".json", ".jsonl"])
def test_records_round_trip_under_a_header(tmp_path, suffix):
    path = tmp_path / f"articles{suffix}"
    records = [article(), article(article_number=2)]

    assert write_records(path, Article, records) == 2
    assert read_records(path, Article) == records


@pytest.mark.parametrize("suffix", [".json", ".jsonl"])
def test_writing_is_deterministic(tmp_path, suffix):
    first, second = tmp_path / f"a{suffix}", tmp_path / f"b{suffix}"
    write_records(first, Article, [article()])
    write_records(second, Article, [article()])

    assert first.read_bytes() == second.read_bytes()


def test_file_written_by_an_older_schema_fails_loudly(tmp_path):
    class ArticleWithNotes(Article):
        notes: str = ""

    path = tmp_path / "articles.json"
    write_records(path, Article, [article()])

    with pytest.raises(SchemaMismatchError, match="dvc repro"):
        read_records(path, ArticleWithNotes)


def test_file_of_another_model_fails_loudly(tmp_path):
    path = tmp_path / "rows.jsonl"
    write_records(path, Row, [])

    with pytest.raises(SchemaMismatchError, match="article@"):
        read_records(path, Article)


# --- Validation -------------------------------------------------------------


def test_unknown_field_is_rejected():
    with pytest.raises(ValidationError, match="Extra inputs"):
        Row(
            page=1,
            row_index=0,
            en_text="",
            ar_text="",
            en_all_bold=False,
            ar_all_bold=False,
            bold=True,
        )


def test_article_needs_a_source_row():
    with pytest.raises(ValidationError):
        article(source_rows=[])
