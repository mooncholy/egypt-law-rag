"""Langfuse traces for /ask (Phase 14): one trace per question, spans inside it.

Each /ask is a trace named ``ask`` holding three spans:

- ``retrieve`` (a retriever span): the champion's search, with ``rerank``
  inside it (the cross-encoder's call to the ``models`` service);
- ``generate`` (a generation): the LLM call, with its model and token usage.

The trace carries the question, the (guarded) answer, the article texts the
LLM was given (``contexts``, which the scheduled evaluation scores
faithfulness against: ``scripts/score_traces.py``), the prompt version, the
retriever's version and the request's correlation ID.

Retrieval runs in a worker thread; ``asyncio.to_thread`` copies the context,
so its spans still nest under the request's trace. Without the three
``RAGLAW_LANGFUSE_*`` settings, ``NO_TRACER`` stands in and nothing is sent.
"""

import json
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from typing import Any

from langchain_core.documents import Document

from raglaw.config import Settings
from raglaw.retrieval.rerank import RerankScorer


class _NoSpan:
    def update(self, **_: Any) -> None:
        pass


class Tracer:
    """Spans on a Langfuse client, or nothing when ``client`` is None."""

    def __init__(self, client: Any = None) -> None:
        self.client = client

    @contextmanager
    def span(self, name: str, *, as_type: str = "span", **fields: Any) -> Iterator[Any]:
        """An observation around the block; yields it, so ``.update()`` adds output."""
        if self.client is None:
            yield _NoSpan()
            return
        with self.client.start_as_current_observation(
            name=name, as_type=as_type, **fields
        ) as observation:
            yield observation

    @contextmanager
    def trace(self, name: str, *, metadata: dict[str, str]) -> Iterator[Any]:
        """The request's root span, with ``metadata`` on the whole trace."""
        if self.client is None:
            yield _NoSpan()
            return
        from langfuse import propagate_attributes

        with (
            propagate_attributes(trace_name=name, metadata=metadata),
            self.span(name, as_type="chain") as root,
        ):
            yield root

    def usage(self, model: str, prompt: int, completion: int) -> None:
        """Put token usage on the current generation (``OpenAIChat.on_usage``)."""
        if self.client is not None:
            self.client.update_current_generation(
                model=model, usage_details={"input": prompt, "output": completion}
            )

    def trace_id(self) -> str | None:
        return self.client.get_current_trace_id() if self.client else None

    def shutdown(self) -> None:
        if self.client is not None:
            self.client.shutdown()


NO_TRACER = Tracer()


def langfuse_tracer(settings: Settings) -> Tracer:
    """
    A tracer on the Langfuse server in ``settings``.

    returns:
    - tracer (Tracer): ``NO_TRACER`` unless the host and both keys are set
    """
    public, secret = settings.langfuse_public_key, settings.langfuse_secret_key
    if not (settings.langfuse_host and public and secret):
        return NO_TRACER
    from langfuse import Langfuse

    return Tracer(
        Langfuse(
            public_key=public.get_secret_value(),
            secret_key=secret.get_secret_value(),
            base_url=settings.langfuse_host,
        )
    )


def traced_retrieve(
    retrieve: Callable[[str], list[Document]], tracer: Tracer
) -> Callable[[str], list[Document]]:
    """``retrieve`` inside a ``retrieve`` span, its output the articles found."""

    def run(question: str) -> list[Document]:
        with tracer.span("retrieve", as_type="retriever", input=question) as span:
            documents = retrieve(question)
            span.update(
                output=[d.metadata.get("article_number") for d in documents],
                metadata={
                    "rank1_rerank_score": (
                        documents[0].metadata.get("retrieval", {}).get("rerank_score")
                        if documents
                        else None
                    )
                },
            )
            return documents

    return run


class TracedReranker:
    """``inner`` inside a ``rerank`` span."""

    def __init__(self, inner: RerankScorer, tracer: Tracer) -> None:
        self.inner = inner
        self.tracer = tracer

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        with self.tracer.span("rerank", metadata={"candidates": len(texts)}) as span:
            scores = self.inner.score(query, texts)
            span.update(output={"top_score": max(scores) if scores else None})
            return scores


class TracedLLM:
    """A ``ChatModel`` whose calls are ``generate`` generations."""

    def __init__(self, inner: Any, tracer: Tracer, *, prompt_version: str) -> None:
        self.inner = inner
        self.tracer = tracer
        self.prompt_version = prompt_version
        if hasattr(inner, "stream"):
            self.stream = self._stream

    async def complete(self, messages: list[dict[str, str]], **options: Any) -> str:
        with self.tracer.span(
            "generate",
            as_type="generation",
            input=messages,
            model=getattr(self.inner, "model", None),
            model_parameters=options,
            version=self.prompt_version,
        ) as generation:
            text = await self.inner.complete(messages, **options)
            generation.update(output=text)
            return text

    async def _stream(self, messages: list[dict[str, str]], **options: Any):
        with self.tracer.span(
            "generate",
            as_type="generation",
            input=messages,
            model=getattr(self.inner, "model", None),
            model_parameters=options,
            version=self.prompt_version,
        ) as generation:
            pieces = []
            async for piece in self.inner.stream(messages, **options):
                pieces.append(piece)
                yield piece
            generation.update(output="".join(pieces))


def _flag(value: Any) -> bool:
    return value is True or str(value).lower() == "true"


def faithfulness_sample(observation: Any) -> dict[str, Any] | None:
    """
    The RAGAS sample for one ``ask`` root span read back from Langfuse, when
    the LLM answered it. Langfuse returns metadata as strings (``"False"``, a
    JSON list), so both forms are read.

    returns:
    - sample (dict | None): ``trace_id``, ``user_input``, ``response`` and
      ``retrieved_contexts``; None for a refusal, a repeal reply or no contexts
    """
    meta = observation.metadata or {}
    contexts = meta.get("contexts") or []
    if isinstance(contexts, str):
        contexts = json.loads(contexts)
    if _flag(meta.get("unaddressed")) or _flag(meta.get("repealed")) or not contexts:
        return None
    if not observation.input or not observation.output:
        return None
    return {
        "trace_id": observation.trace_id,
        "user_input": observation.input,
        "response": observation.output,
        "retrieved_contexts": list(contexts),
    }
