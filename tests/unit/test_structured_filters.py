"""Unit tests for the document filter builder (spec section 4)."""

from __future__ import annotations

from datetime import date

from kb.search.structured import build_document_where


def test_empty_filters() -> None:
    sql, params = build_document_where({})
    assert sql == "TRUE"
    assert params == []


def test_doc_type_and_title() -> None:
    sql, params = build_document_where({"doc_type": "regulation", "title_contains": "отпуск"})
    assert "d.doc_type = $1" in sql
    assert "d.title ILIKE '%' || $2 || '%'" in sql
    assert params == ["regulation", "отпуск"]


def test_attributes_containment() -> None:
    sql, params = build_document_where({"attributes": {"status": "active"}})
    assert "d.attributes @> $1::jsonb" in sql
    assert params == [{"status": "active"}]


def test_attribute_date_filter() -> None:
    sql, params = build_document_where({"review_by_before": "2026-07-18"})
    assert "(d.attributes ->> $1)::date < $2" in sql
    assert params == ["review_by", date(2026, 7, 18)]


def test_created_range() -> None:
    sql, params = build_document_where({"created_after": "2026-01-01"})
    assert "d.created_at::date > $1" in sql
    assert params == [date(2026, 1, 1)]


def test_unparseable_date_predicate_skipped() -> None:
    sql, params = build_document_where({"created_before": "not-a-date"})
    assert sql == "TRUE"
    assert params == []


def test_entity_with_role() -> None:
    sql, params = build_document_where({"entity": {"name": "Иван Петров", "role": "owner"}})
    assert "EXISTS (SELECT 1 FROM entity_mentions em" in sql
    assert "e.canonical = $1" in sql
    assert "em.role = $2" in sql
    assert params == ["иван петров", "owner"]


def test_unknown_key_ignored() -> None:
    _sql, params = build_document_where({"whatever": "x", "doc_type": "faq"})
    assert params == ["faq"]
