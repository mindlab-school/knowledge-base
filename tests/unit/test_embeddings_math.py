"""Unit tests for embedding math and the query prefix (spec section 3)."""

from __future__ import annotations

import math

from kb.embeddings import (
    QUERY_PREFIX,
    apply_query_prefix,
    truncate_and_renormalize,
)


def test_truncation_to_dim() -> None:
    vector = list(range(2048))
    result = truncate_and_renormalize(vector, 1024)
    assert len(result) == 1024


def test_renormalized_to_unit_length() -> None:
    result = truncate_and_renormalize([3.0, 4.0, 0.0, 0.0], 4)
    assert math.isclose(math.sqrt(sum(x * x for x in result)), 1.0, rel_tol=1e-5)


def test_zero_vector_stays_zero() -> None:
    assert truncate_and_renormalize([0.0, 0.0, 0.0], 3) == [0.0, 0.0, 0.0]


def test_query_prefix_applied() -> None:
    assert apply_query_prefix("вопрос") == f"{QUERY_PREFIX}вопрос"
    assert apply_query_prefix("вопрос").startswith("Represent the query")
