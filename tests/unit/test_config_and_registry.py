"""Unit tests for config, the doc-type registry, and topic slugs."""

from __future__ import annotations

from kb.config import Settings
from kb.doc_types.models import GenericDoc
from kb.doc_types.registry import (
    DOC_TYPE_REGISTRY,
    attribute_schemas,
    get_model,
    is_known_type,
    type_descriptions,
)
from kb.services.facts import slugify_topic


def test_settings_defaults() -> None:
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    assert settings.embed_dim == 1024
    assert settings.cache_min_tokens == 4096
    assert settings.agent_model == "anthropic/claude-haiku-4.5"


def test_agent_model_slug_with_variant() -> None:
    settings = Settings(_env_file=None, or_model_variant=":exacto")  # type: ignore[call-arg]
    assert settings.agent_model_slug == "anthropic/claude-haiku-4.5:exacto"


def test_registry_has_five_types() -> None:
    assert set(DOC_TYPE_REGISTRY) == {
        "generic",
        "regulation",
        "instruction",
        "faq",
        "meeting_notes",
    }


def test_unknown_type_falls_back_to_generic() -> None:
    assert get_model("nope") is GenericDoc
    assert not is_known_type("nope")
    assert is_known_type("regulation")


def test_schemas_and_descriptions_present() -> None:
    schemas = attribute_schemas()
    assert "properties" in schemas["regulation"]
    assert type_descriptions()["faq"]


def test_slugify_topic_drops_stopwords() -> None:
    slug = slugify_topic("у нас три пакета цены и условия")
    assert "-" in slug
    assert "и" not in slug.split("-")
    assert slug == slugify_topic("у нас три пакета цены и условия")


def test_slugify_topic_fallback() -> None:
    assert slugify_topic("!!!") == "fakt"
