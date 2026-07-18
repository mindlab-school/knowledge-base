"""Unit tests for Reciprocal Rank Fusion (spec section 4)."""

from __future__ import annotations

from kb.search.semantic import reciprocal_rank_fusion


def test_score_formula() -> None:
    fused = dict(reciprocal_rank_fusion([["a", "b"]], k=60))
    assert fused["a"] == 1 / 61
    assert fused["b"] == 1 / 62


def test_item_in_both_lists_ranks_first() -> None:
    fused = reciprocal_rank_fusion([["a", "b", "c"], ["c", "d"]])
    assert fused[0][0] == "c"


def test_pure_vector_hit_survives() -> None:
    # 'a' appears only in the first (vector) list; it must still be present.
    fused = dict(reciprocal_rank_fusion([["a"], ["b", "c"]]))
    assert "a" in fused


def test_empty_lists() -> None:
    assert reciprocal_rank_fusion([[], []]) == []
