"""Repealed ranges: their former text, for matching only, and how they were repealed.

Two ranges of the Code survive only as a repeal note: Articles 54-80
(associations and foundations) and 389-417 (proof of obligations). A question
on their subject finds nothing in the current text, since the note holds none
of its words (Phase 6). ``data/repealed/repealed.json`` holds, per range:

- **sections** of the 1948 text, each headed by its English and Arabic
  heading. The text is an OCR of the original Code and is noisy; it is used
  only to recognise that a question is about a repealed subject;
- **the repeal record**: the subject and, where the current Code's repeal
  note gives them, the date and the repealing law.

The answer to such a question comes from the record and cites the current
Code's range. The former text is never shown or used to answer.
"""

import json
from dataclasses import dataclass, replace
from datetime import date
from pathlib import Path
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from raglaw.retrieval.rerank import RerankScorer

# Characters of former text per matching passage, each prefixed with its
# section's headings; sized like the probe that set the threshold (2026-10-14).
PASSAGE_CHARS = 600

AR_DIGITS = str.maketrans("0123456789", "٠١٢٣٤٥٦٧٨٩")
AR_MONTHS = (
    "يناير",
    "فبراير",
    "مارس",
    "أبريل",
    "مايو",
    "يونيو",
    "يوليو",
    "أغسطس",
    "سبتمبر",
    "أكتوبر",
    "نوفمبر",
    "ديسمبر",
)


class RepealRecord(BaseModel):
    """How one range was repealed: what the answer to a matching question says."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    first: int = Field(gt=0, description="The range's first article.")
    last: int = Field(gt=0, description="The range's last article.")
    subject_en: str = Field(min_length=1)
    subject_ar: str = Field(min_length=1)
    repealed_on: date | None = Field(
        default=None,
        description="The repeal's date, only when the current Code prints it.",
    )
    repealed_by_en: str | None = Field(
        default=None,
        min_length=1,
        description="The repealing law, only when the current Code names it.",
    )
    repealed_by_ar: str | None = Field(default=None, min_length=1)
    source: str = Field(min_length=1, description="The Code's repeal note it rests on.")

    @model_validator(mode="after")
    def _law_in_both_languages(self) -> Self:
        if (self.repealed_by_en is None) != (self.repealed_by_ar is None):
            raise ValueError("the repealing law is named in both languages or neither")
        return self


class RepealedSection(BaseModel):
    """One section of a repealed range's former text, under its headings."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    first: int = Field(description="The first article of the range it belongs to.")
    heading_en: str = Field(min_length=1)
    heading_ar: str = Field(min_length=1)
    text: str = Field(description="The 1948 text (OCR, noisy); matching only.")


class RepealedSet(BaseModel):
    """The contents of ``repealed.json``."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source: str = Field(min_length=1, description="Where the former text comes from.")
    ranges: list[RepealRecord] = Field(min_length=1)
    sections: list[RepealedSection] = Field(min_length=1)

    @model_validator(mode="after")
    def _sections_belong_to_ranges(self) -> Self:
        firsts = {r.first for r in self.ranges}
        if len(firsts) != len(self.ranges):
            raise ValueError("two repeal records share a first article")
        stray = {s.first for s in self.sections} - firsts
        if stray:
            raise ValueError(f"sections of ranges with no repeal record: {stray}")
        bare = firsts - {s.first for s in self.sections}
        if bare:
            raise ValueError(f"repeal records with no section to match: {bare}")
        return self


def load_repealed(path: Path) -> RepealedSet:
    """
    Read ``repealed.json``.

    exceptions:
    - pydantic.ValidationError: the file doesn't fit ``RepealedSet``
    """
    return RepealedSet.model_validate(json.loads(path.read_text(encoding="utf-8")))


def passages(
    sections: list[RepealedSection], size: int = PASSAGE_CHARS
) -> list[tuple[RepealedSection, str]]:
    """
    Cut each section's text into pieces of ``size`` characters, each headed by
    the section's English and Arabic headings, so that a piece carries its
    subject even where the OCR has garbled the words.

    returns:
    - passages (list[tuple[RepealedSection, str]]): each piece's section and text
    """
    out = []
    for s in sections:
        head = f"{s.heading_en}\n{s.heading_ar}\n"
        body = s.text or " "
        out += [(s, head + body[i : i + size]) for i in range(0, len(body), size)]
    return out


@dataclass(frozen=True)
class RepealedMatch:
    """The repealed range a question is closest to, and how close."""

    record: RepealRecord
    score: float
    section: str  # the English heading of the best passage
    # The rank-1 reranker score before the repeal note was put first: the
    # low score that triggered the fallback, which the note itself lacks.
    rank1_rerank_score: float | None = None

    def triggered_by(self, rank1_rerank_score: float) -> Self:
        """The same match, with the rank-1 score that let the fallback run."""
        return replace(self, rank1_rerank_score=rank1_rerank_score)

    def to_metadata(self) -> dict:
        """The match as plain data, for ``metadata["retrieval"]``."""
        return {
            "record": self.record.model_dump(mode="json"),
            "score": self.score,
            "section": self.section,
            "rank1_rerank_score": self.rank1_rerank_score,
        }


class RepealedMatcher:
    """Score a question against every repealed passage with the reranker."""

    def __init__(self, repealed: RepealedSet, scorer: RerankScorer) -> None:
        self.records = {r.first: r for r in repealed.ranges}
        self._passages = passages(repealed.sections)
        self._scorer = scorer

    def match(self, question: str) -> RepealedMatch:
        """
        The best-scoring passage's range, whatever its score; the caller
        decides whether it is close enough.

        returns:
        - match (RepealedMatch): the range, the score and the section
        """
        scores = self._scorer.score(question, [t for _, t in self._passages])
        best = max(range(len(scores)), key=lambda i: scores[i])
        section = self._passages[best][0]
        return RepealedMatch(
            self.records[section.first], scores[best], section.heading_en
        )


def _date_en(d: date) -> str:
    return f"{d.day} {d.strftime('%B')} {d.year}"


def _date_ar(d: date) -> str:
    return f"{d.day} {AR_MONTHS[d.month - 1]} {d.year}".translate(AR_DIGITS)


def repeal_answer(record: RepealRecord, language: str) -> str:
    """
    The reply to a question on a repealed subject: the range was repealed,
    with the date and the law when the current Code gives them. Nothing of
    the former text is stated.

    returns:
    - text (str): in the question's language
    """
    if language == "ar":
        first, last = (str(n).translate(AR_DIGITS) for n in (record.first, record.last))
        when = (
            f" اعتبارًا من {_date_ar(record.repealed_on)}" if record.repealed_on else ""
        )
        by = f" {record.repealed_by_ar}" if record.repealed_by_ar else ""
        return (
            f"كانت أحكام {record.subject_ar} في القانون المدني واردة في المواد من "
            f"{first} إلى {last}، وقد ألغيت{when}{by}. ولا يتضمن القانون المدني "
            "الحالي من هذه المواد إلا ما يفيد إلغاءها."
        )
    when = f" as of {_date_en(record.repealed_on)}" if record.repealed_on else ""
    by = f" {record.repealed_by_en}" if record.repealed_by_en else ""
    return (
        f"The Civil Code's provisions on {record.subject_en} were in Articles "
        f"{record.first}-{record.last}. They were repealed{when}{by}. The current "
        "Code keeps only a note that these articles are repealed."
    )
