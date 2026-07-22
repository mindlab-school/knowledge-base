"""Application configuration — the single place that reads the environment.

All settings come from ``.env`` / process environment (spec section 9). Nothing
else in the codebase reads ``os.environ`` directly; modules call
:func:`get_settings` instead.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

EmbedBackend = Literal["local", "openrouter"]
PromptCacheMode = Literal["auto", "off"]


class Settings(BaseSettings):
    """Typed view over environment configuration.

    Field names map to upper-case environment variables (``openrouter_api_key``
    ← ``OPENROUTER_API_KEY``). Values are validated once at process start.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- OpenRouter / LLM gateway ---------------------------------------
    openrouter_api_key: str = ""
    openrouter_base_url: str = "https://openrouter.ai/api/v1"

    # --- Telegram / backend ---------------------------------------------
    telegram_bot_token: str = ""
    database_url: str = "postgresql://kb:kb@localhost:5432/kb"
    backend_url: str = "http://localhost:8000"
    # Shared secret guarding the backend HTTP surface. Empty disables auth
    # (local/dev); when set, clients must send it as the ``X-KB-Secret`` header.
    backend_shared_secret: str = ""

    # --- Agent model ----------------------------------------------------
    agent_model: str = "anthropic/claude-haiku-4.5"
    # empty | ":exacto" (tool-calling accuracy) | ":nitro" — appended to the slug.
    or_model_variant: str = ""
    # Only for models with adaptive thinking (Sonnet 5); empty for Haiku 4.5.
    agent_effort: str = ""

    # --- Extraction model ----------------------------------------------
    extraction_model: str = "anthropic/claude-haiku-4.5"

    # --- Embeddings -----------------------------------------------------
    embed_backend: EmbedBackend = "local"
    embed_model: str = "voyage-4-nano"
    # Must match ``halfvec(N)`` in the schema. Changing it requires a reindex.
    embed_dim: int = 1024
    # Local ONNX backend only: directory holding model.onnx + tokenizer.json.
    embed_model_path: str = "models/voyage-4-nano"

    # --- Prompt cache ---------------------------------------------------
    prompt_cache: PromptCacheMode = "auto"
    cache_min_tokens: int = 4096
    facts_block_limit: int = 4000

    # --- OpenRouter routing / headers / budget --------------------------
    or_allow_fallbacks: bool = True
    or_site_url: str = ""
    or_app_name: str = "kb-agent"
    or_monthly_limit: float = Field(default=20.0, description="soft spend limit, USD")

    @property
    def agent_model_slug(self) -> str:
        """Effective agent model slug, including the routing variant.

        Nitro/Exacto are variants of the model slug (``…:exacto``), not a
        ``sort`` field, so the variant is concatenated onto the base slug.
        """
        return f"{self.agent_model}{self.or_model_variant}"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings singleton."""
    return Settings()
