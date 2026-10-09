import asyncio
import json

import httpx
import pytest
from openai import AsyncOpenAI

from raglaw.llm import LLMError, OpenAIChat

pytestmark = pytest.mark.unit

MESSAGES = [{"role": "user", "content": "Question"}]


def chat_with(handler) -> OpenAIChat:
    """An ``OpenAIChat`` whose endpoint is ``handler`` (request -> response)."""
    llm = OpenAIChat("http://llm.test/v1", "key", "test-model")
    llm.client = AsyncOpenAI(
        base_url="http://llm.test/v1",
        api_key="key",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    return llm


def completion(content: str | None) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "id": "c1",
            "object": "chat.completion",
            "created": 0,
            "model": "test-model",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
        },
    )


def test_the_reply_is_returned_stripped_with_the_model_and_temperature_sent():
    sent = []

    def handler(request):
        sent.append(json.loads(request.content))
        return completion("  Article 44.\n")

    reply = asyncio.run(chat_with(handler).complete(MESSAGES, temperature=0.0))

    assert reply == "Article 44."
    assert (sent[0]["model"], sent[0]["temperature"]) == ("test-model", 0.0)


def test_an_empty_reply_is_an_llm_error():
    llm = chat_with(lambda request: completion(None))

    with pytest.raises(LLMError, match="empty"):
        asyncio.run(llm.complete(MESSAGES, temperature=0.0))


def test_an_endpoint_error_is_an_llm_error():
    llm = chat_with(lambda request: httpx.Response(500, json={"error": "down"}))

    with pytest.raises(LLMError, match="test-model"):
        asyncio.run(llm.complete(MESSAGES, temperature=0.0))
