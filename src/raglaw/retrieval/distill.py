"""Reranker distillation (Phase 13): a smaller cross-encoder that learns the teacher's scores.

The teacher is ``reranker`` (bge-reranker-v2-m3, 568M parameters); the student
is ``distillation.student`` (a multilingual MiniLM cross-encoder, 118M). The
training data never touches the eval set:

1. ``write_questions``: the generator writes one question per article sampled
   from those no eval question expects or cites (``held_out_articles``),
   alternating Arabic and English like the eval set.
2. ``label``: each question runs through the retriever with the teacher as its
   reranker, so the pairs are exactly the ``rerank_depth`` fused candidates the
   reranker sees in serving. Each pair keeps the teacher's score (as a logit)
   and a hard label (1 when the chunk is the question's source article).
3. ``train_student``: the student learns the teacher's logits (MSE), and, as
   the control, the hard labels alone (binary cross-entropy), from the same
   checkpoint, pairs and settings.

``scripts/distill_reranker.py`` runs the steps and scores every reranker on
the eval set.
"""

import asyncio
import math
import random
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

from raglaw.config import Distillation
from raglaw.llm import LLMError, OpenAIChat
from raglaw.retrieval.embeddings import check_device
from raglaw.retrieval.rerank import RerankScorer
from raglaw.retrieval.scoring import EvalQuestion
from raglaw.schema import Article

Target = Literal["soft", "hard"]
LOGIT_CLIP = 1e-6  # keeps a probability of exactly 0 or 1 finite as a logit

QUESTION_SYSTEM = (
    "You write one question that a member of the public might ask and that the "
    "given article of the Egyptian Civil Code answers. Ask about the situation, "
    "in plain words; never mention the article or its number. Reply with the "
    "question only."
)
LANGUAGE = {"ar": "in Arabic (Modern Standard Arabic)", "en": "in English"}
# The generator sometimes drifts into another script mid-question (Chinese,
# or English words inside Arabic ones); such a reply is asked for again.
CJK = re.compile(r"[\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af]")
LATIN = re.compile(r"[A-Za-z]")
ARABIC = re.compile(r"[\u0600-\u06ff]")
MAX_LATIN_IN_ARABIC = 3
ATTEMPTS = 3


def held_out_articles(questions: Sequence[EvalQuestion]) -> set[int]:
    """Every article an eval question expects or cites as support."""
    return {
        n for q in questions for n in (*q.expected_articles, *q.supporting_articles)
    }


def pick_articles(
    articles: Sequence[Article], exclude: set[int], n: int, seed: int
) -> list[Article]:
    """
    ``n`` live articles with text in both languages, none in ``exclude``, in
    article order.

    exceptions:
    - ValueError: fewer than ``n`` articles qualify
    """
    pool = [
        a
        for a in articles
        if not a.is_repealed
        and a.text_ar.strip()
        and a.text_en.strip()
        and a.article_number not in exclude
    ]
    if len(pool) < n:
        raise ValueError(f"only {len(pool)} articles qualify, {n} asked")
    return sorted(random.Random(seed).sample(pool, n), key=lambda a: a.article_number)


def question_messages(article: Article, language: str) -> list[dict[str, str]]:
    """The prompt asking for one question, in ``language``, that ``article`` answers."""
    return [
        {"role": "system", "content": QUESTION_SYSTEM},
        {
            "role": "user",
            "content": f"Write the question {LANGUAGE[language]}.\n\n"
            f"Article (Arabic):\n{article.text_ar}\n\n"
            f"Article (English):\n{article.text_en}",
        },
    ]


def clean_question(reply: str, language: str) -> str | None:
    """
    The reply's first line, when it is written in ``language``'s script only.

    returns:
    - question (str | None): None for a reply mixing in another script
    """
    question = reply.splitlines()[0].strip()
    if CJK.search(question):
        return None
    if language == "ar":
        latin = len(LATIN.findall(question))
        return question if latin <= MAX_LATIN_IN_ARABIC else None
    return None if ARABIC.search(question) else question


async def write_questions(
    articles: Sequence[Article], llm: OpenAIChat, *, concurrency: int = 16
) -> list[dict[str, Any]]:
    """
    One question per article, alternating Arabic and English. A reply in a
    mixed script is asked for again, up to ``ATTEMPTS`` times; an article
    whose calls fail or never give a clean question is skipped.

    returns:
    - questions (list[dict]): ``article_number``, ``language`` and ``question``
    """
    gate = asyncio.Semaphore(concurrency)

    async def one(i: int, article: Article) -> dict[str, Any] | None:
        language = "ar" if i % 2 == 0 else "en"
        for _ in range(ATTEMPTS):
            async with gate:
                try:
                    reply = await llm.complete(
                        question_messages(article, language),
                        temperature=0.7,
                        max_tokens=120,
                    )
                except LLMError:
                    return None
            if question := clean_question(reply, language):
                return {
                    "article_number": article.article_number,
                    "language": language,
                    "question": question,
                }
        return None

    found = await asyncio.gather(*(one(i, a) for i, a in enumerate(articles)))
    return [q for q in found if q is not None]


class RecordingScorer:
    """A reranker that passes scoring to ``inner`` and keeps every call."""

    def __init__(self, inner: RerankScorer) -> None:
        self.inner = inner
        self.calls: list[tuple[str, list[str], list[float]]] = []

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        scores = self.inner.score(query, texts)
        self.calls.append((query, list(texts), scores))
        return scores


def to_logit(probability: float) -> float:
    """The teacher's sigmoid score turned back into its logit."""
    p = min(max(probability, LOGIT_CLIP), 1 - LOGIT_CLIP)
    return math.log(p / (1 - p))


def pairs_for(
    query: str,
    texts: Sequence[str],
    scores: Sequence[float],
    articles: Sequence[int],
    source: int,
) -> list[dict[str, Any]]:
    """
    One training pair per candidate: the teacher's logit and the hard label.

    ``articles`` gives each text's article number; the label is 1 for the
    question's ``source`` article.
    """
    return [
        {
            "query": query,
            "text": text,
            "teacher_logit": to_logit(score),
            "label": float(article == source),
        }
        for text, score, article in zip(texts, scores, articles, strict=True)
    ]


def context_precision(ranked: Sequence[int], expected: set[int], k: int = 5) -> float:
    """
    RAGAS context precision with the gold articles as the reference: the mean
    of precision@i over the ranks i <= k that hold an expected article.

    returns:
    - precision (float): 0 when none of the top ``k`` is expected
    """
    hits, total = 0, 0.0
    for i, article in enumerate(ranked[:k], 1):
        if article in expected:
            hits += 1
            total += hits / i
    return total / hits if hits else 0.0


class LocalCrossEncoder:
    """A cross-encoder from a local directory (a trained student), as a reranker."""

    def __init__(
        self, path: Path | str, *, max_length: int, batch_size: int, device: str
    ) -> None:
        from sentence_transformers import CrossEncoder  # embed group only

        check_device(device)
        self.model = CrossEncoder(str(path), device=device, max_length=max_length)
        self.batch_size = batch_size

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        scores = self.model.predict(
            [(query, t) for t in texts],
            batch_size=self.batch_size,
            show_progress_bar=False,
        )
        return [float(s) for s in scores]


def train_student(
    pairs: Sequence[dict[str, Any]],
    out: Path,
    *,
    target: Target,
    distillation: Distillation,
    device: str,
) -> None:
    """
    Train the student on ``pairs`` and save it in ``out``: on the teacher's
    logits (``soft``) or on the hard labels alone (``hard``).
    """
    import torch
    from datasets import Dataset
    from sentence_transformers.cross_encoder import (
        CrossEncoder,
        CrossEncoderTrainer,
        CrossEncoderTrainingArguments,
    )
    from sentence_transformers.cross_encoder.losses import (
        BinaryCrossEntropyLoss,
        MSELoss,
    )

    check_device(device)
    torch.manual_seed(distillation.seed)
    student = distillation.student
    model = CrossEncoder(
        student.model,
        revision=student.revision,
        device=device,
        max_length=student.max_length,
    )
    column = "teacher_logit" if target == "soft" else "label"
    data = Dataset.from_dict(
        {
            "query": [p["query"] for p in pairs],
            "text": [p["text"] for p in pairs],
            "label": [p[column] for p in pairs],
        }
    )
    loss = MSELoss(model) if target == "soft" else BinaryCrossEntropyLoss(model)
    args = CrossEncoderTrainingArguments(
        output_dir=str(out / "checkpoints"),
        num_train_epochs=distillation.epochs,
        per_device_train_batch_size=distillation.train_batch_size,
        learning_rate=distillation.learning_rate,
        warmup_ratio=0.1,
        seed=distillation.seed,
        bf16=device.startswith("cuda"),
        save_strategy="no",
        logging_steps=50,
        report_to="none",
    )
    CrossEncoderTrainer(model=model, args=args, train_dataset=data, loss=loss).train()
    model.save_pretrained(str(out))
