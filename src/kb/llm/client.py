"""OpenRouter chat client built on the OpenAI SDK (spec sections 3, 5).

This is the single gateway for LLM calls (agent loop + ingestion extraction).
OpenRouter-specific fields the OpenAI SDK does not model natively —
``session_id`` (sticky routing), ``provider`` (fallbacks), ``usage.include``
(cost + cache fields), ``reasoning`` — are passed through ``extra_body``.
Prompt ``cache_control`` lives on message content blocks and is therefore the
caller's responsibility (see :mod:`kb.agent.loop`).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from kb.config import Settings, get_settings

if TYPE_CHECKING:
    from openai import AsyncOpenAI
    from openai.types.chat import ChatCompletion


class LLMClient:
    """Async wrapper over ``chat.completions`` pointed at OpenRouter."""

    def __init__(self, settings: Settings | None = None, client: AsyncOpenAI | None = None) -> None:
        self._settings = settings or get_settings()
        self._client = client

    def _default_headers(self) -> dict[str, str]:
        headers: dict[str, str] = {}
        if self._settings.or_site_url:
            headers["HTTP-Referer"] = self._settings.or_site_url
        if self._settings.or_app_name:
            headers["X-Title"] = self._settings.or_app_name
        return headers

    def _get_client(self) -> AsyncOpenAI:
        if self._client is None:
            from openai import AsyncOpenAI

            self._client = AsyncOpenAI(
                api_key=self._settings.openrouter_api_key,
                base_url=self._settings.openrouter_base_url,
                default_headers=self._default_headers(),
            )
        return self._client

    async def chat(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        max_tokens: int,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: dict[str, Any] | str | None = None,
        response_format: dict[str, Any] | None = None,
        session_id: str | None = None,
        reasoning: dict[str, Any] | None = None,
    ) -> ChatCompletion:
        """Run one chat-completions request and return the raw response.

        :param session_id: Sticky-routing key (the conversation id) so OpenRouter
            keeps the same provider across agent rounds.
        :param reasoning: Only sent for models that support adaptive thinking.
        """
        extra_body: dict[str, Any] = {
            "usage": {"include": True},
            "provider": {"allow_fallbacks": self._settings.or_allow_fallbacks},
        }
        if session_id is not None:
            extra_body["session_id"] = session_id
        if reasoning is not None:
            extra_body["reasoning"] = reasoning

        params: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "extra_body": extra_body,
        }
        if tools is not None:
            params["tools"] = tools
        if tool_choice is not None:
            params["tool_choice"] = tool_choice
        if response_format is not None:
            params["response_format"] = response_format

        client = self._get_client()
        response = await client.chat.completions.create(**params)
        return cast("ChatCompletion", response)


def response_usage(response: ChatCompletion) -> dict[str, Any]:
    """Extract the usage payload (incl. OpenRouter cost/cache fields) as a dict."""
    data = response.model_dump()
    usage = data.get("usage")
    return usage if isinstance(usage, dict) else {}
