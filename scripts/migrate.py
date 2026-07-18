"""Forward-only SQL migration runner (spec section 0.5, 12).

Applies numbered ``migrations/NNNN_*.sql`` files that have not been recorded in
``schema_migrations`` yet, each in its own transaction, in filename order. There
are no down-migrations: a schema change is always a new file.

Usage::

    python scripts/migrate.py            # apply pending migrations
    python scripts/migrate.py --status   # list applied / pending without changes
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

import asyncpg

MIGRATIONS_DIR = Path(__file__).resolve().parent.parent / "migrations"

_ENSURE_TABLE = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    version    TEXT PRIMARY KEY,
    applied_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
"""


def _database_url() -> str:
    """Read the database URL from the environment via the app config."""
    # Imported lazily so the script works even before the package is importable.
    from kb.config import get_settings

    return get_settings().database_url


def discover_migrations() -> list[Path]:
    """Return migration files sorted by their numeric prefix."""
    files = sorted(MIGRATIONS_DIR.glob("[0-9]*.sql"))
    if not files:
        raise RuntimeError(f"no migration files found in {MIGRATIONS_DIR}")
    return files


async def _applied_versions(conn: asyncpg.Connection) -> set[str]:
    rows = await conn.fetch("SELECT version FROM schema_migrations")
    return {row["version"] for row in rows}


async def run_migrations(database_url: str, *, dry_run: bool = False) -> list[str]:
    """Apply every pending migration and return the versions that were applied."""
    conn = await asyncpg.connect(database_url)
    applied: list[str] = []
    try:
        await conn.execute(_ENSURE_TABLE)
        done = await _applied_versions(conn)
        for path in discover_migrations():
            version = path.name
            if version in done:
                continue
            if dry_run:
                applied.append(version)
                continue
            sql = path.read_text(encoding="utf-8")
            async with conn.transaction():
                await conn.execute(sql)
                await conn.execute("INSERT INTO schema_migrations (version) VALUES ($1)", version)
            applied.append(version)
            print(f"applied {version}")
    finally:
        await conn.close()
    return applied


async def _status(database_url: str) -> None:
    conn = await asyncpg.connect(database_url)
    try:
        await conn.execute(_ENSURE_TABLE)
        done = await _applied_versions(conn)
    finally:
        await conn.close()
    for path in discover_migrations():
        mark = "applied" if path.name in done else "pending"
        print(f"[{mark}] {path.name}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Apply forward-only SQL migrations.")
    parser.add_argument(
        "--status", action="store_true", help="show applied/pending migrations and exit"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="report what would be applied without applying"
    )
    args = parser.parse_args(argv)

    url = _database_url()
    if args.status:
        asyncio.run(_status(url))
        return 0

    applied = asyncio.run(run_migrations(url, dry_run=args.dry_run))
    if not applied:
        print("database is up to date")
    elif args.dry_run:
        print("would apply:", ", ".join(applied))
    return 0


if __name__ == "__main__":
    sys.exit(main())
