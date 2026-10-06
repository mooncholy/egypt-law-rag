import hashlib
import json
from enum import StrEnum
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field


class LogEvent(StrEnum):
    """Machine-readable event names attached to log lines.

    Every log line carries one of these as ``event_type``, so a log search can
    filter on the event rather than on wording that may change.
    """

    # System events
    STARTUP = "system_startup"
    SHUTDOWN = "system_shutdown"

    # HTTP Request events
    REQUEST_START = "request_start"
    REQUEST_COMPLETED = "request_completed"
    REQUEST_FAILED = "request_failed"

    VALIDATION_FAILED = "validation_failed"

    # Pipeline events: each DVC stage logs its input hash at start and its
    # output counts at completion; anomalies are never silently absorbed.
    STAGE_START = "stage_start"
    STAGE_COMPLETED = "stage_completed"
    STAGE_FAILED = "stage_failed"
    ANOMALY = "anomaly"


def _shape(node: Any) -> Any:
    """A JSON schema reduced to what decides compatibility.

    Descriptions, titles and examples are documentation, and ``required`` is
    sorted, so rewording a field or reordering fields keeps the fingerprint.
    """
    if isinstance(node, dict):
        return {
            k: sorted(v) if k == "required" else _shape(v)
            for k, v in node.items()
            if k not in {"description", "title", "examples"}
        }
    if isinstance(node, list):
        return [_shape(v) for v in node]
    return node


class Record(BaseModel):
    """Base for every record a pipeline stage writes.

    A model's schema version is derived, never set by hand: it is a
    fingerprint of the model's fields, types and constraints. Changing any of
    them changes the fingerprint, so a file written by older code fails to load
    (``raglaw.records``) instead of being misread. The fingerprint lives in the
    file's header, not on every record.
    """

    MODEL_NAME: ClassVar[str]

    model_config = ConfigDict(extra="forbid", frozen=True)

    @classmethod
    def schema_fingerprint(cls) -> str:
        """
        Fingerprint the model's shape.

        returns:
        - fingerprint (str): the first 12 hex digits of the sha256 of the
          model's JSON schema, without its documentation
        """
        shape = json.dumps(_shape(cls.model_json_schema()), sort_keys=True)
        return hashlib.sha256(shape.encode()).hexdigest()[:12]

    @classmethod
    def schema_id(cls) -> str:
        """
        Name the model and its fingerprint in one string, for headers and run params.

        returns:
        - schema_id (str): ``<model name>@<fingerprint>``, e.g. ``article@3f9a1c0b7e2d``
        """
        return f"{cls.MODEL_NAME}@{cls.schema_fingerprint()}"


class Row(Record):
    """One detected row of a page's table, as `extract` and `repair` write it."""

    MODEL_NAME = "row"

    page: int = Field(ge=1, description="Source page number, 1-based.")
    row_index: int = Field(
        ge=0, description="0-based index of the row in the page's `lines_strict` table."
    )
    en_text: str = Field(description="The English cell, lines joined with `\\n`.")
    ar_text: str = Field(description="The Arabic cell, lines joined with `\\n`.")
    en_all_bold: bool = Field(
        description="True when every English span is bold (the heading signal, P15)."
    )


class Article(Record):
    """One Civil Code article, as `assemble` builds it (contract section 3)."""

    MODEL_NAME = "article"

    article_number: int = Field(ge=1, description="Keyed on the English number (R20).")
    text_en: str = Field(description="English body, without the `Article N` header.")
    text_ar: str = Field(description="Arabic body, without the `مادة N` header.")
    heading_path: list[str] = Field(
        description="English headings from the root down, root first."
    )
    heading_path_ar: list[str] = Field(
        description="The same headings in Arabic, aligned with `heading_path`."
    )
    source_pages: list[int] = Field(
        min_length=1, description="Every page the article's rows sit on, ascending."
    )
    source_rows: list[tuple[int, int]] = Field(
        min_length=1,
        description="Every `(page, row_index)` merged into the article, in order.",
    )
    is_repealed: bool = Field(
        default=False, description="True for 54-80 and 389-417; the text is the note."
    )


class Chunk(Record):
    """One retrievable piece of an article, as `chunk` writes it.

    A chunk never crosses an article boundary. An article longer than
    ``max_chars`` splits only at its Arabic paragraph markers (R22), and every
    part carries the article's full English text (D4).
    """

    MODEL_NAME = "chunk"

    chunk_id: str = Field(description="`art-{article_number}-p{part_index}`.")
    article_number: int = Field(ge=1)
    citation: str = Field(description="How an answer cites the article.")
    heading_path: list[str]
    heading_path_ar: list[str]
    source_pages: list[int] = Field(min_length=1)
    is_repealed: bool
    part_index: int = Field(ge=1, description="1-based position among the parts.")
    part_count: int = Field(ge=1)
    paragraphs: list[int] = Field(
        description="The Arabic paragraph numbers this part covers; empty when the "
        "article has no markers."
    )
    text_ar: str
    text_en: str = Field(description="The article's full English text (D4).")
    strategy: str = Field(description="The chunking strategy that produced it.")
