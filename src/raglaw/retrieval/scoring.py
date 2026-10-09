"""Score retrieval against the eval set (``data/gold/retrieval_eval.jsonl``).

A question is scored on *articles*, not chunks: a retriever's ranked chunks
collapse to the articles they belong to, in order of first appearance, so the
two parts of Article 1143 count once.

- **recall@k**: the share of questions whose expected articles are all within
  the top k (``match: all``), or at least one of them (``match: any``, used
  where any article of a repealed range answers).
- **MRR**: the mean of 1 / rank of the first expected article returned, 0 when
  none is.

Both are reported overall, per ``kind``, per language and per register. The
out-of-scope questions have no article to find, so they are left out of both;
their rank-1 scores are reported next to the in-scope ones, as the evidence for
a later "no answer" threshold. Bilingual agreement is the share of Arabic and
English question pairs (``pair_id``) whose top article is the same.

``split_questions`` divides the set into a tuning half (choosing a config) and a
held-out half (reported once, for the chosen one) (D15).
"""

import json
import statistics
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Self

from langchain_core.documents import Document
from pydantic import BaseModel, ConfigDict, Field, model_validator

KS = (1, 3, 5, 10)
SCORE_NAMES = ("dense_score", "bm25_score", "rrf_score", "rerank_score")
SPLITS = ("tuning", "heldout")

Kind = Literal[
    "rule",
    "rule_with_exception",
    "multi_article",
    "cross_reference",
    "lay_term",
    "one_language_only",
    "article_lookup",
    "repealed",
    "out_of_scope",
]


class EvalSetError(ValueError):
    """The eval set's questions don't fit together (ids, pairs, translations)."""


class EvalQuestion(BaseModel):
    """One question of the retrieval eval set, with the articles that answer it."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(pattern=r"^q\d{3}$", description="`q001`, `q002`, ...")
    question: str = Field(min_length=1)
    language: Literal["ar", "en"]
    # `register` in the file; renamed here, as BaseModel has an attribute of that name.
    speech_register: Literal["english", "msa", "colloquial"] = Field(
        alias="register",
        description="`msa` is Modern Standard Arabic; `colloquial`, Egyptian.",
    )
    kind: Kind = Field(description="What the question tests.")
    expected_articles: list[int] = Field(
        description="The articles that govern the answer; empty when out of scope."
    )
    supporting_articles: list[int] = Field(
        description="Articles that help but aren't required; not scored."
    )
    match: Literal["all", "any"] = Field(
        description="Whether every expected article must be found, or one."
    )
    legal_basis: str = Field(description="Why those articles answer it.")
    difficulty: Literal["easy", "medium", "hard"]
    pair_id: str | None = Field(
        default=None,
        description="Shared by an Arabic and an English question asking the same.",
    )
    translated_from: str | None = Field(
        default=None, description="For a translated twin, the id of its source."
    )

    @model_validator(mode="after")
    def _scope_matches_articles(self) -> Self:
        if (self.kind == "out_of_scope") == bool(self.expected_articles):
            raise ValueError(
                "an out_of_scope question has no expected articles, and every "
                "other question has at least one"
            )
        return self

    @property
    def in_scope(self) -> bool:
        return self.kind != "out_of_scope"


def _check_set(questions: Sequence[EvalQuestion]) -> None:
    ids = [q.id for q in questions]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        raise EvalSetError(f"duplicate question ids: {duplicates}")
    pairs: dict[str, list[EvalQuestion]] = {}
    for q in questions:
        if q.pair_id is not None:
            pairs.setdefault(q.pair_id, []).append(q)
    for pair_id, members in sorted(pairs.items()):
        if (
            sorted(q.language for q in members) != ["ar", "en"]
            or len({q.kind for q in members}) != 1
        ):
            raise EvalSetError(
                f"pair {pair_id} must be one Arabic and one English question of "
                f"one kind, got {[(q.id, q.language, q.kind) for q in members]}"
            )
    by_id = {q.id: q for q in questions}
    for q in questions:
        source = by_id.get(q.translated_from) if q.translated_from else None
        if q.translated_from and (
            source is None or source.pair_id != q.pair_id or q.pair_id is None
        ):
            raise EvalSetError(
                f"{q.id}: translated_from {q.translated_from!r} must name a "
                "question in the same pair"
            )


def read_questions(path: Path) -> list[EvalQuestion]:
    """
    Read and check the eval set.

    returns:
    - questions (list[EvalQuestion]): in file order

    exceptions:
    - pydantic.ValidationError: a line doesn't fit ``EvalQuestion``
    - EvalSetError: duplicate ids, a pair that isn't one Arabic and one English
      question of one kind, or a ``translated_from`` outside its pair
    """
    lines = path.read_text(encoding="utf-8").splitlines()
    questions = [EvalQuestion.model_validate_json(line) for line in lines if line]
    _check_set(questions)
    return questions


def _imbalance(
    counts: dict[str, Counter], unit: list[EvalQuestion], half: str
) -> tuple[int, int]:
    """How unbalanced the halves would be with ``unit`` in ``half``: first on
    the unit's kind, then over every register."""
    after = {name: c.copy() for name, c in counts.items()}
    for q in unit:
        after[half].update([("kind", q.kind), ("register", q.speech_register)])
    tuning, heldout = after["tuning"], after["heldout"]
    kind = abs(tuning[("kind", unit[0].kind)] - heldout[("kind", unit[0].kind)])
    register = sum(
        abs(tuning[("register", r)] - heldout[("register", r)])
        for r in ("english", "msa", "colloquial")
    )
    return kind, register


def split_questions(
    questions: Iterable[EvalQuestion],
) -> dict[str, list[EvalQuestion]]:
    """
    Divide the questions into a tuning and a held-out half (D15).

    Questions are taken in id order, and each goes to the half that leaves the
    two most balanced: first on its ``kind``, then over registers (and so
    languages), tuning on a tie. Each half then holds about half of every kind
    and of every register, so a config tuned on one half isn't tuned on mostly
    English and reported on mostly Arabic. A pair moves as one, at its first
    id's place, so both languages of a question land in the same half; it
    counts as two questions. A question's half depends only on those before
    it, so appending questions with higher ids never moves an existing one.

    returns:
    - split (dict[str, list[EvalQuestion]]): ``tuning`` and ``heldout``, each
      sorted by id
    """
    units: dict[str, list[EvalQuestion]] = {}
    for q in questions:
        units.setdefault(q.pair_id or q.id, []).append(q)
    counts: dict[str, Counter] = {name: Counter() for name in SPLITS}
    split: dict[str, list[EvalQuestion]] = {name: [] for name in SPLITS}
    for unit in sorted(units.values(), key=lambda u: min(q.id for q in u)):
        half = min(SPLITS, key=lambda name: _imbalance(counts, unit, name))
        split[half] += unit
        for q in unit:
            counts[half].update([("kind", q.kind), ("register", q.speech_register)])
    return {name: sorted(qs, key=lambda q: q.id) for name, qs in split.items()}


@dataclass(frozen=True)
class Retrieved:
    """What a retriever returned for one question, reduced to what scoring reads."""

    articles: list[int] = field(
        metadata={"doc": "Article numbers, in order of their first chunk."}
    )
    top_scores: dict[str, float | None] = field(
        metadata={"doc": "The rank-1 chunk's dense, BM25 and RRF scores."}
    )
    ranges: dict[int, int] = field(
        default_factory=dict,
        metadata={
            "doc": "For a chunk standing for a repealed range, its first "
            "article -> its last; every article between ranks with it."
        },
    )

    @classmethod
    def from_documents(cls, documents: Sequence[Document]) -> Self:
        """
        Collapse ``HybridRetriever`` hits to their articles.

        returns:
        - retrieved (Retrieved): empty ``top_scores`` when nothing came back
        """
        articles = list(dict.fromkeys(d.metadata["article_number"] for d in documents))
        ranges = {
            d.metadata["article_number"]: d.metadata["range_end"]
            for d in documents
            if d.metadata.get("range_end")
        }
        top = documents[0].metadata["retrieval"] if documents else {}
        return cls(articles, {k: top[k] for k in SCORE_NAMES if k in top}, ranges)

    def positions(self) -> dict[int, int]:
        """Each article's 1-based rank; every article of a range takes the range's."""
        ranked: dict[int, int] = {}
        for rank, first in enumerate(self.articles, start=1):
            for number in range(first, self.ranges.get(first, first) + 1):
                ranked.setdefault(number, rank)
        return ranked


@dataclass(frozen=True)
class QuestionScore:
    """How one question fared."""

    question: EvalQuestion
    retrieved: Retrieved
    ranks: dict[int, int | None]  # expected article -> its rank, None if missing
    hits: dict[int, bool]  # k -> found within the top k
    reciprocal_rank: float | None  # None for an out-of-scope question

    @property
    def top_article(self) -> int | None:
        return self.retrieved.articles[0] if self.retrieved.articles else None

    def to_record(self) -> dict[str, Any]:
        """
        Flatten the score for a per-question file, which runs are compared by.

        returns:
        - record (dict[str, Any]): the question's id and groups, its expected
          articles, the rank of the first one found (None if none), a hit per
          k (keys as strings, for JSON), what came back (a range as
          ``first-last``) and the rank-1 scores
        """
        q, found = self.question, [r for r in self.ranks.values() if r is not None]
        ranges = self.retrieved.ranges
        return {
            "id": q.id,
            "kind": q.kind,
            "language": q.language,
            "register": q.speech_register,
            "pair_id": q.pair_id,
            "in_scope": q.in_scope,
            "match": q.match,
            "expected": q.expected_articles,
            "first_rank": min(found) if found else None,
            "hits": {str(k): hit for k, hit in self.hits.items()},
            "returned": [
                f"{a}-{ranges[a]}" if a in ranges else str(a)
                for a in self.retrieved.articles
            ],
            "top_scores": self.retrieved.top_scores,
        }


def score_question(
    question: EvalQuestion, retrieved: Retrieved, ks: Sequence[int] = KS
) -> QuestionScore:
    """
    Score one question's retrieval on its expected articles.

    returns:
    - score (QuestionScore): per-article ranks, a hit per k and the
      reciprocal rank; an out-of-scope question has none of them
    """
    if not question.in_scope:
        return QuestionScore(question, retrieved, {}, {}, None)
    position = retrieved.positions()
    ranks = {a: position.get(a) for a in question.expected_articles}
    found = [r for r in ranks.values() if r is not None]
    need = all if question.match == "all" else any
    hits = {k: need(r is not None and r <= k for r in ranks.values()) for k in ks}
    return QuestionScore(
        question, retrieved, ranks, hits, 1 / min(found) if found else 0.0
    )


def _mean(values: list[float]) -> float:
    return round(statistics.fmean(values), 6) if values else 0.0


def _recall(scores: list[QuestionScore], ks: Sequence[int]) -> dict[str, Any]:
    out: dict[str, Any] = {"questions": len(scores)}
    for k in ks:
        out[f"recall_at_{k}"] = _mean([float(s.hits[k]) for s in scores])
    out["mrr"] = _mean([s.reciprocal_rank or 0.0 for s in scores])
    return out


def _by(scores: list[QuestionScore], attr: str, ks: Sequence[int]) -> dict[str, Any]:
    groups: dict[str, list[QuestionScore]] = {}
    for s in scores:
        groups.setdefault(getattr(s.question, attr), []).append(s)
    return {name: _recall(group, ks) for name, group in sorted(groups.items())}


def _stats(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    return {
        "min": round(min(values), 6),
        "median": round(statistics.median(values), 6),
        "max": round(max(values), 6),
    }


def _top1(scores: list[QuestionScore]) -> dict[str, Any]:
    return {
        name: _stats(
            [
                s.retrieved.top_scores[name]
                for s in scores
                if s.retrieved.top_scores.get(name) is not None
            ]
        )
        for name in SCORE_NAMES
    }


def summarize(
    scores: Sequence[QuestionScore], ks: Sequence[int] = KS
) -> dict[str, Any]:
    """
    Aggregate question scores into the stage's metrics.

    returns:
    - summary (dict[str, Any]): ``overall`` and ``by_kind`` / ``by_language`` /
      ``by_register`` recall@k and MRR over in-scope questions;
      ``out_of_scope_questions``; and ``top1_scores``, the min, median and
      max rank-1 score per score type for in-scope and out-of-scope questions
      (None where the mode gives no such score)
    """
    in_scope = [s for s in scores if s.question.in_scope]
    out_of_scope = [s for s in scores if not s.question.in_scope]
    return {
        "overall": _recall(in_scope, ks),
        "by_kind": _by(in_scope, "kind", ks),
        "by_language": _by(in_scope, "language", ks),
        "by_register": _by(in_scope, "speech_register", ks),
        "out_of_scope_questions": len(out_of_scope),
        "top1_scores": {
            "in_scope": _top1(in_scope),
            "out_of_scope": _top1(out_of_scope),
        },
    }


def bilingual_agreement(scores: Sequence[QuestionScore]) -> dict[str, Any]:
    """
    The share of question pairs whose two languages retrieve the same top article.

    Only pairs with both members among ``scores`` count.

    returns:
    - agreement (dict[str, Any]): ``pairs``, ``same_top1`` (None without
      pairs), and both per ``kind``
    """
    pairs: dict[str, list[QuestionScore]] = {}
    for s in scores:
        if s.question.pair_id is not None:
            pairs.setdefault(s.question.pair_id, []).append(s)
    complete = [members for members in pairs.values() if len(members) == 2]

    def agree(group: list[list[QuestionScore]]) -> dict[str, Any]:
        same = [
            a.top_article is not None and a.top_article == b.top_article
            for a, b in group
        ]
        return {
            "pairs": len(same),
            "same_top1": _mean([float(x) for x in same]) if same else None,
        }

    by_kind: dict[str, list[list[QuestionScore]]] = {}
    for members in complete:
        by_kind.setdefault(members[0].question.kind, []).append(members)
    return {
        **agree(complete),
        "by_kind": {kind: agree(group) for kind, group in sorted(by_kind.items())},
    }


def dump(summary: dict[str, Any]) -> str:
    """Serialize metrics deterministically: sorted keys, indented, a trailing newline."""
    return json.dumps(summary, indent=2, sort_keys=True) + "\n"
