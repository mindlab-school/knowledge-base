"""Unit tests for the process-level embedder cache (spec section 3).

No network or ONNX load: the cache is keyed on config, and both backend
constructors are lazy, so identity can be asserted without embedding.
"""

from __future__ import annotations

import pytest

from kb.config import Settings
from kb.ingestion.embeddings import make_embedder, reset_embedder_cache


@pytest.fixture(autouse=True)
def _reset() -> None:
    reset_embedder_cache()


def _settings(**overrides: object) -> Settings:
    params: dict[str, object] = {"embed_backend": "openrouter", **overrides}
    return Settings(_env_file=None, **params)  # type: ignore[call-arg]


def test_same_settings_returns_same_instance() -> None:
    settings = _settings()
    assert make_embedder(settings) is make_embedder(settings)


def test_equivalent_config_across_settings_objects_shares_instance() -> None:
    assert make_embedder(_settings()) is make_embedder(_settings())


def test_reset_forces_rebuild() -> None:
    settings = _settings()
    first = make_embedder(settings)
    reset_embedder_cache()
    assert make_embedder(settings) is not first


def test_changed_key_rebuilds() -> None:
    first = make_embedder(_settings(embed_dim=1024))
    second = make_embedder(_settings(embed_dim=512))
    assert second is not first


def test_different_backend_yields_different_instance() -> None:
    cloud = make_embedder(_settings())
    local = make_embedder(_settings(embed_backend="local"))
    assert local is not cloud
    assert type(local) is not type(cloud)
