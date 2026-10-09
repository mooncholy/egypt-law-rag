"""``answer``: a question in, an answer citing only retrieved articles out (Phase 7).

For one question, ``AnswerPipeline.answer``:

1. retrieves with ``champion`` (the registered retriever);
2. refuses without calling the LLM when the rank-1 reranker score is below
   ``answer.no_answer_threshold`` (D21): the Code is taken not to address the
   question. A rank-1 chunk fetched because the question names its article is
   always answered;
3. gives the LLM the top ``answer.max_sources`` articles, each with its
   repealed flag and any passage printed in one language only (R28), under the
   versioned system prompt ``prompts/answer_<version>.md``;
4. keeps as ``sources`` the given articles the answer cites, as
   ``Egyptian Civil Code, Article N``. An article the answer cites that it
   wasn't given is dropped from ``sources`` and logged as a
   ``hallucinated_citation`` anomaly: that is the failure this phase guards.
"""

import asyncio
import re
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from importlib.resources import files
from typing import Literal

from langchain_core.documents import Document

from raglaw.config import Answer as AnswerConfig
from raglaw.llm import ChatModel
from raglaw.logging_conf import get_logger
from raglaw.retrieval.lookup import cited_articles
from raglaw.schema import LogEvent

logger = get_logger(__name__)

# The event of the one log line per question. Not a LogEvent member: LogEvent
# lives in schema.py, which every DVC stage depends on, so adding one would
# rerun the whole pipeline for a name.
ANSWER_EVENT = "answer"

Language = Literal["ar", "en"]

NO_ANSWER = {
    "ar": "لم أجد في مواد القانون المدني المصري ما يتناول هذا السؤال.",
    "en": "I could not find an article of the Egyptian Civil Code that addresses "
    "this question.",
}
LANGUAGE_NAMES = {"ar": "Arabic (العربية)", "en": "English"}
ARABIC_LETTER = re.compile(r"[ء-ي]")  # letters, not Arabic-Indic digits
LATIN_LETTER = re.compile(r"[A-Za-z]")


# A prompt file holds the system prompt, then, after this line, the user
# message's template, so a version fixes both. v1 predates the template.
USER_MARKER = "<!-- user -->"
V1_USER = (
    "Question:\n{question}\n\nAnswer in {language}.\n\nArticles found:\n\n{articles}"
)


@dataclass(frozen=True)
class Prompt:
    """One prompt version: the system prompt and the user message's template
    (``{question}``, ``{language}``, ``{articles}``)."""

    system: str
    user: str


def load_prompt(version: str) -> Prompt:
    """
    The prompt ``answer.prompt_version`` names.

    exceptions:
    - FileNotFoundError: no ``prompts/answer_<version>.md``
    """
    text = (files("raglaw.prompts") / f"answer_{version}.md").read_text("utf-8")
    system, _, user = text.partition(USER_MARKER)
    return Prompt(system.strip(), user.strip() or V1_USER)


def question_language(question: str) -> Language:
    """
    The language to answer in: Arabic when the question has at least as many
    Arabic words as Latin ones, so a quoted ``Article 147`` keeps it Arabic.

    returns:
    - language (Language): ``ar`` or ``en``
    """
    words = re.findall(r"\w+", question)
    arabic = sum(bool(ARABIC_LETTER.search(w)) for w in words)
    latin = sum(bool(LATIN_LETTER.search(w)) for w in words)
    return "ar" if arabic and arabic >= latin else "en"


@dataclass(frozen=True)
class ContextArticle:
    """One article as the LLM sees it: every retrieved part of it, merged."""

    first: int
    last: int  # above ``first`` only for a chunk standing for a repealed range
    citation: str  # `Article 147 | المادة ١٤٧`, so either language finds it
    is_repealed: bool
    text_ar: str
    text_en: str
    only_in_en: list[str] = field(default_factory=list)
    only_in_ar: list[str] = field(default_factory=list)

    @property
    def numbers(self) -> range:
        return range(self.first, self.last + 1)

    def render(self) -> str:
        label = (
            self.citation
            if self.first == self.last
            else f"Articles {self.first} to {self.last}"
        )
        lines = [f"### {label}{' (repealed)' if self.is_repealed else ''}"]
        lines += [f"Arabic text:\n{self.text_ar}", f"English text:\n{self.text_en}"]
        if self.only_in_ar:
            lines.append("Only in the Arabic text:\n" + "\n".join(self.only_in_ar))
        if self.only_in_en:
            lines.append("Only in the English text:\n" + "\n".join(self.only_in_en))
        return "\n\n".join(lines)


def context_articles(
    documents: Sequence[Document], max_sources: int
) -> list[ContextArticle]:
    """
    The first ``max_sources`` articles among the retrieved chunks, in rank order.

    An article split into parts appears once: its Arabic parts in part order,
    its English text once (every part carries the whole of it, D4).

    returns:
    - articles (list[ContextArticle]): at most ``max_sources``
    """
    parts: dict[int, list[dict]] = {}
    for document in documents:
        meta = document.metadata
        number = meta["article_number"]
        if number not in parts and len(parts) == max_sources:
            continue
        parts.setdefault(number, []).append(meta)
    articles = []
    for number, metas in parts.items():
        metas = sorted(metas, key=lambda m: m["part_index"])
        articles.append(
            ContextArticle(
                first=number,
                last=metas[0].get("range_end") or number,
                citation=metas[0]["citation"],
                is_repealed=metas[0]["is_repealed"],
                text_ar="\n".join(m["text_ar"] for m in metas),
                text_en=metas[0]["text_en"],
                only_in_en=metas[0]["only_in_en"],
                only_in_ar=[p for m in metas for p in m["only_in_ar"]],
            )
        )
    return articles


def is_unaddressed(documents: Sequence[Document], threshold: float) -> bool:
    """
    Whether the Code is taken not to address the question (D21).

    returns:
    - unaddressed (bool): True when nothing came back, or the rank-1 chunk's
      reranker score is below ``threshold``; False when the rank-1 chunk was
      fetched by its article number, or carries no reranker score
    """
    if not documents:
        return True
    top = documents[0].metadata["retrieval"]
    if top["lookup"] or top["rerank_score"] is None:
        return False
    return top["rerank_score"] < threshold


def build_messages(
    question: str,
    language: Language,
    articles: Sequence[ContextArticle],
    prompt: Prompt,
) -> list[dict[str, str]]:
    """
    The chat: the system prompt, then the question with its articles.

    returns:
    - messages (list[dict[str, str]]): OpenAI chat messages
    """
    user = prompt.user.format(
        question=question,
        language=LANGUAGE_NAMES[language],
        articles="\n\n".join(a.render() for a in articles),
    )
    return [
        {"role": "system", "content": prompt.system},
        {"role": "user", "content": user},
    ]


def citation(number: int) -> str:
    """How ``sources`` names an article (the handbook's format)."""
    return f"Egyptian Civil Code, Article {number}"


def check_citations(
    text: str, articles: Sequence[ContextArticle]
) -> tuple[list[int], list[int]]:
    """
    Split the articles an answer cites into those it was given and the rest.

    returns:
    - given (list[int]): cited and given, in order of first citation
    - hallucinated (list[int]): cited but not given
    """
    allowed = {n for a in articles for n in a.numbers}
    cited = cited_articles(text)
    return [n for n in cited if n in allowed], [n for n in cited if n not in allowed]


@dataclass(frozen=True)
class Answer:
    """What ``answer`` returns; ``/ask`` sends ``text`` and ``sources``."""

    text: str
    sources: list[str]
    language: Language
    unaddressed: bool  # refused by the no-answer gate; the LLM wasn't called
    articles: list[int]  # given to the LLM, in rank order
    rank1_rerank_score: float | None
    hallucinated: list[int] = field(default_factory=list)


class AnswerPipeline:
    """Retrieve, gate, prompt, and check the answer's citations.

    ``retrieve`` is called in a worker thread under a lock: Qdrant's local mode
    isn't safe across threads, and the event loop stays free meanwhile.
    """

    def __init__(
        self,
        retrieve: Callable[[str], list[Document]],
        llm: ChatModel,
        config: AnswerConfig,
    ) -> None:
        self._retrieve = retrieve
        self._lock = threading.Lock()
        self.llm = llm
        self.config = config
        self.prompt = load_prompt(config.prompt_version)

    def _retrieve_locked(self, question: str) -> list[Document]:
        with self._lock:
            return self._retrieve(question)

    async def answer(self, question: str) -> Answer:
        """
        Answer ``question`` from the retrieved articles.

        returns:
        - answer (Answer): the reply and the article citations it rests on

        exceptions:
        - ModelsServiceError: the ``models`` service failed during retrieval
        - LLMError: the LLM call failed
        """
        language = question_language(question)
        documents = await asyncio.to_thread(self._retrieve_locked, question)
        score = (
            documents[0].metadata["retrieval"]["rerank_score"] if documents else None
        )
        if is_unaddressed(documents, self.config.no_answer_threshold):
            logger.info(
                "Question not addressed by the Code",
                extra={"event_type": ANSWER_EVENT, "rank1_rerank_score": score},
            )
            return Answer(NO_ANSWER[language], [], language, True, [], score)
        articles = context_articles(documents, self.config.max_sources)
        text = await self.llm.complete(
            build_messages(question, language, articles, self.prompt),
            temperature=self.config.temperature,
        )
        given, hallucinated = check_citations(text, articles)
        for number in hallucinated:
            logger.warning(
                "Answer cites an article it wasn't given",
                extra={
                    "event_type": LogEvent.ANOMALY,
                    "anomaly": "hallucinated_citation",
                    "article_number": number,
                    "given": [a.first for a in articles],
                },
            )
        logger.info(
            "Question answered",
            extra={
                "event_type": ANSWER_EVENT,
                "rank1_rerank_score": score,
                "articles": [a.first for a in articles],
                "cited": given,
                "prompt_version": self.config.prompt_version,
            },
        )
        return Answer(
            text=text,
            sources=[citation(n) for n in given],
            language=language,
            unaddressed=False,
            articles=[a.first for a in articles],
            rank1_rerank_score=score,
            hallucinated=hallucinated,
        )
