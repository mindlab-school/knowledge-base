"""Shared test fixtures and fakes.

Unit tests need nothing external. Integration tests use a real
PostgreSQL+pgvector container (testcontainers); if Docker or the image is
unavailable, those tests skip cleanly instead of failing.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio

from kb.config import get_settings
from tests.fakes import HashingEmbedder

# The Ryuk resource reaper fails to map its port on some Docker versions; the
# per-container labels still get cleaned up on container.stop().
os.environ.setdefault("TESTCONTAINERS_RYUK_DISABLED", "true")

PG_IMAGE = "pgvector/pgvector:0.8.2-pg18"
MIGRATIONS_DIR = Path(__file__).resolve().parent.parent / "migrations"

_TABLES = [
    "messages",
    "conversations",
    "facts",
    "pending_document_links",
    "document_links",
    "entity_relations",
    "entity_mentions",
    "entities",
    "chunks",
    "documents",
    "ingest_buffer",
    "ingest_sessions",
    "users",
]

try:
    from testcontainers.postgres import PostgresContainer

    _HAS_TESTCONTAINERS = True
except ImportError:  # pragma: no cover
    _HAS_TESTCONTAINERS = False


@pytest.fixture
def fake_embedder() -> HashingEmbedder:
    return HashingEmbedder()


# --- integration infrastructure -----------------------------------------


async def _apply_migrations(dsn: str) -> None:
    import asyncpg

    conn = await asyncpg.connect(dsn)
    try:
        for path in sorted(MIGRATIONS_DIR.glob("[0-9]*.sql")):
            await conn.execute(path.read_text(encoding="utf-8"))
    finally:
        await conn.close()


@pytest.fixture(scope="session")
def pg_dsn() -> Any:
    # CI provides a ready PostgreSQL+pgvector service via KB_TEST_DSN; otherwise
    # spin one up with testcontainers. Either way, apply migrations once.
    external = os.environ.get("KB_TEST_DSN")
    if external:
        os.environ["DATABASE_URL"] = external
        get_settings.cache_clear()
        asyncio.run(_apply_migrations(external))
        yield external
        return

    if not _HAS_TESTCONTAINERS:
        pytest.skip("testcontainers not installed and KB_TEST_DSN not set")
    try:
        container = PostgresContainer(PG_IMAGE, username="kb", password="kb", dbname="kb")
        container.start()
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"cannot start postgres container: {exc}")

    host = container.get_container_host_ip()
    port = container.get_exposed_port(5432)
    dsn = f"postgresql://kb:kb@{host}:{port}/kb"
    os.environ["DATABASE_URL"] = dsn
    get_settings.cache_clear()
    try:
        asyncio.run(_apply_migrations(dsn))
    except Exception as exc:  # pragma: no cover
        container.stop()
        pytest.skip(f"cannot apply migrations: {exc}")

    yield dsn
    container.stop()


@pytest_asyncio.fixture
async def pool(pg_dsn: str) -> Any:
    from kb.db.pool import close_pool, get_pool

    await close_pool()
    pool_obj = await get_pool()
    async with pool_obj.acquire() as conn:
        await conn.execute(f"TRUNCATE {', '.join(_TABLES)} RESTART IDENTITY CASCADE")
    yield pool_obj
    await close_pool()


@pytest_asyncio.fixture
async def whitelisted_user(pool: Any) -> dict[str, Any]:
    from kb.db.repo import users as users_repo

    async with pool.acquire() as conn:
        return await users_repo.create(conn, 1001, "Test User")
