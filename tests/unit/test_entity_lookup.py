"""Unit tests for entity lookup robustness (spec section 4).

Covers the fuzzy-match LIKE escaping and the single canonicalization formula:
the exact-match lookup must bind a Python-canonicalized key rather than re-derive
it in SQL, and the fuzzy match must treat ``%``/``_`` in a name literally.
"""

from __future__ import annotations

from typing import Any

from kb.db.repo import entities as entities_repo
from kb.search import entities


def test_fuzzy_sql_has_escape_clause() -> None:
    assert "ESCAPE '\\'" in entities._FIND_FUZZY_SQL


def test_exact_sql_binds_precanonicalized_key() -> None:
    # The canonical key is computed once in Python (``canonicalize``); the SQL no
    # longer re-derives it with ``lower(trim(...))`` (SQL ``trim`` strips only
    # spaces and ``lower`` is collation-dependent).
    assert "lower(trim" not in entities._FIND_EXACT_SQL
    assert "canonical = $1" in entities._FIND_EXACT_SQL


def test_escape_like_escapes_wildcards() -> None:
    # Backslash first, then the LIKE wildcards ``%`` and ``_``.
    assert entities._escape_like("name_%") == "name\\_\\%"
    assert entities._escape_like("a\\b") == "a\\\\b"
    assert entities._escape_like("plain") == "plain"


class _FakeConn:
    def __init__(self) -> None:
        self.fetchrow_calls: list[tuple[str, tuple[Any, ...]]] = []

    async def fetchrow(self, sql: str, *args: Any) -> None:
        self.fetchrow_calls.append((sql, args))
        return None


class _Acquire:
    def __init__(self, conn: _FakeConn) -> None:
        self._conn = conn

    async def __aenter__(self) -> _FakeConn:
        return self._conn

    async def __aexit__(self, *exc: object) -> bool:
        return False


class _FakePool:
    def __init__(self, conn: _FakeConn) -> None:
        self._conn = conn

    def acquire(self) -> _Acquire:
        return _Acquire(self._conn)


async def test_explore_entity_routes_canonical_and_escaped_params() -> None:
    conn = _FakeConn()
    result = await entities.explore_entity("  Ivan_Petrov%  ", pool=_FakePool(conn))  # type: ignore[arg-type]

    assert result is None  # both lookups miss, so we never reach the graph step
    exact_sql, exact_args = conn.fetchrow_calls[0]
    assert exact_sql is entities._FIND_EXACT_SQL
    assert exact_args == (entities_repo.canonicalize("  Ivan_Petrov%  "),)
    fuzzy_sql, fuzzy_args = conn.fetchrow_calls[1]
    assert fuzzy_sql is entities._FIND_FUZZY_SQL
    assert fuzzy_args == (entities._escape_like("  Ivan_Petrov%  "),)
