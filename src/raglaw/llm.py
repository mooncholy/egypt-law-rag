"""The generative model, reached through any OpenAI-compatible endpoint (D20).

A local vLLM and a hosted API look the same to the ``openai`` client, so the
endpoint, model and key are per-machine settings, never code.
"""

import logging
from typing import Protocol

logger = logging.getLogger(__name__)

# A 3B model answering from five articles needs seconds; a hung endpoint must
# fail the request instead of holding it.
TIMEOUT_SECONDS = 60.0


class LLMError(RuntimeError):
    """The endpoint couldn't be reached, refused the request, or answered nothing."""


class ChatModel(Protocol):
    """Anything that completes a chat: the endpoint, or a stub in tests."""

    async def complete(
        self, messages: list[dict[str, str]], *, temperature: float, max_tokens: int
    ) -> str: ...


class OpenAIChat:
    """``model`` at an OpenAI-compatible ``base_url``."""

    def __init__(self, base_url: str, api_key: str, model: str) -> None:
        from openai import AsyncOpenAI

        self.client = AsyncOpenAI(
            base_url=base_url, api_key=api_key, timeout=TIMEOUT_SECONDS, max_retries=1
        )
        self.model = model

    async def complete(
        self, messages: list[dict[str, str]], *, temperature: float, max_tokens: int
    ) -> str:
        """
        One chat completion, at most ``max_tokens`` long: at temperature 0 a
        small model can repeat itself without end, and an uncapped reply then
        runs until the timeout. A reply cut at the cap is logged.

        returns:
        - text (str): the reply, stripped

        exceptions:
        - LLMError: the call failed, or the reply was empty
        """
        from openai import OpenAIError

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except OpenAIError as exc:
            raise LLMError(f"{self.model} at {self.client.base_url}: {exc}") from exc
        choice = response.choices[0]
        if choice.finish_reason == "length":
            logger.warning(
                "Reply cut at max_tokens=%d", max_tokens, extra={"model": self.model}
            )
        text = (choice.message.content or "").strip()
        if not text:
            raise LLMError(f"{self.model} returned an empty answer")
        return text
