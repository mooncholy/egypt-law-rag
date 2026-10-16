import asyncio
from contextlib import contextmanager

import pytest
from langchain_core.documents import Document

from raglaw.config import Settings
from raglaw.monitoring.tracing import (
    NO_TRACER,
    TracedLLM,
    TracedReranker,
    Tracer,
    faithfulness_sample,
    langfuse_tracer,
    traced_retrieve,
)

pytestmark = pytest.mark.unit


class Observation:
    def __init__(self, name, as_type, parent, fields):
        self.name, self.as_type, self.parent = name, as_type, parent
        self.fields = dict(fields)

    def update(self, **fields):
        self.fields |= fields


class FakeLangfuse:
    """The few client calls ``Tracer`` makes, recording each observation's parent."""

    def __init__(self):
        self.observations: list[Observation] = []
        self._open: list[Observation] = []

    @contextmanager
    def start_as_current_observation(self, *, name, as_type="span", **fields):
        parent = self._open[-1].name if self._open else None
        observation = Observation(name, as_type, parent, fields)
        self.observations.append(observation)
        self._open.append(observation)
        try:
            yield observation
        finally:
            self._open.pop()

    def update_current_generation(self, **fields):
        self._open[-1].update(**fields)

    def get_current_trace_id(self):
        return "trace-1"


class FakeChat:
    model = "qwen"

    def __init__(self, tracer):
        self.tracer = tracer

    async def complete(self, messages, *, temperature, max_tokens):
        self.tracer.usage(self.model, 120, 7)  # as OpenAIChat's on_usage does
        return "Article 44."


class FakeReranker:
    def score(self, query, texts):
        return [0.9 for _ in texts]


def test_an_ask_trace_holds_retrieve_rerank_and_generate():
    """Phase 14: retrieve -> rerank, then generate with its token usage."""
    client = FakeLangfuse()
    tracer = Tracer(client)
    reranker = TracedReranker(FakeReranker(), tracer)

    def retrieve(question):
        reranker.score(question, ["a", "b"])
        return [Document("text", metadata={"article_number": 44})]

    llm = TracedLLM(FakeChat(tracer), tracer, prompt_version="v2")

    async def ask():
        with tracer.span("ask", as_type="chain") as root:
            documents = await asyncio.to_thread(traced_retrieve(retrieve, tracer), "Q")
            text = await llm.complete([], temperature=0.0, max_tokens=64)
            root.update(output=text)
        return documents

    asyncio.run(ask())

    tree = [(o.name, o.as_type, o.parent) for o in client.observations]
    assert tree == [
        ("ask", "chain", None),
        ("retrieve", "retriever", "ask"),
        ("rerank", "span", "retrieve"),
        ("generate", "generation", "ask"),
    ]
    generate = client.observations[-1].fields
    assert generate["usage_details"] == {"input": 120, "output": 7}
    assert generate["version"] == "v2"
    assert client.observations[1].fields["output"] == [44]


def test_without_langfuse_settings_nothing_is_traced(clean_env):
    tracer = langfuse_tracer(Settings(_env_file=None))

    assert tracer is NO_TRACER
    with tracer.trace("ask", metadata={}) as root:
        root.update(output="ignored")  # a no-op, never an error


class Stored:
    """An ``ask`` root span as Langfuse returns it: metadata as strings."""

    def __init__(self, **metadata):
        self.trace_id, self.input, self.output = "t1", "Q?", "Under Article 44, ..."
        self.metadata = metadata


@pytest.mark.parametrize(
    ("metadata", "kept"),
    [
        (
            {"unaddressed": "False", "repealed": "False", "contexts": '["Art. 44"]'},
            True,
        ),
        (
            {"unaddressed": "True", "repealed": "False", "contexts": '["Art. 44"]'},
            False,
        ),
        (
            {"unaddressed": "False", "repealed": "True", "contexts": '["Art. 54"]'},
            False,
        ),
        ({"unaddressed": False, "repealed": False, "contexts": ["Art. 44"]}, True),
        ({"unaddressed": "False", "repealed": "False", "contexts": "[]"}, False),
    ],
)
def test_only_traces_the_llm_answered_are_scored(metadata, kept):
    """The scheduled evaluation reads metadata back as strings ("False", JSON)."""
    sample = faithfulness_sample(Stored(**metadata))

    assert (sample is not None) is kept
    if kept:
        assert sample["retrieved_contexts"] == ["Art. 44"]
