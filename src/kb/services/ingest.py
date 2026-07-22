"""Ingest use-case service (spec sections 3, 4, 6).

Application-service entry points behind the ``/ingest/*`` endpoints. Each one
composes the lower-level ingest primitives (idle-session autoclose, active-session
lookup, the ingestion pipeline, session bookkeeping) into a single reusable call
so handlers and other channels stay thin. Validation failures surface as domain
exceptions; the transport layer maps them to HTTP status codes.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import asyncpg

from kb.db.pool import get_pool
from kb.ingestion.pipeline import IngestResult, ingest_source
from kb.ingestion.session import (
    SessionToggleResult,
    append_text,
    get_active_session,
    maybe_autoclose,
    note_document,
    toggle_session,
)
from kb.ingestion.sources.files import FileSource, is_supported, telegram_file_source
from kb.ingestion.sources.urls import UrlSource

MAX_FILE_SIZE = 20 * 1024 * 1024


class UnsupportedFileType(Exception):
    """The uploaded file's extension is not one the pipeline can parse."""

    def __init__(self, name: str) -> None:
        super().__init__(f"неподдерживаемый тип файла: {name}")


class FileTooLarge(Exception):
    """The uploaded file exceeds :data:`MAX_FILE_SIZE`."""

    def __init__(self) -> None:
        super().__init__("файл превышает 20 MB")


class NoActiveSession(Exception):
    """A buffered message was sent while no ingest session is open."""

    def __init__(self) -> None:
        super().__init__("нет активной сессии набора")


async def _active_session_id(user_id: int, *, pool: asyncpg.Pool) -> int | None:
    """Autoclose any stale session, then return the active session id (or ``None``)."""
    await maybe_autoclose(user_id, pool=pool)
    active = await get_active_session(user_id, pool=pool)
    return int(active["id"]) if active else None


async def ingest_uploaded_file(
    user_id: int,
    data: bytes,
    name: str,
    *,
    origin: str = "file",
    pool: asyncpg.Pool | None = None,
) -> IngestResult:
    """Ingest an uploaded file's bytes, namespacing the source path by user.

    ``origin == "telegram"`` stores a ``tg:<user_id>:<name>`` path; any other
    origin uses ``file:<user_id>:<name>``, so re-uploads version a user's own
    document without colliding with other users. Raises :class:`UnsupportedFileType`
    or :class:`FileTooLarge` on validation failure.
    """
    if not is_supported(name):
        raise UnsupportedFileType(name)
    if len(data) > MAX_FILE_SIZE:
        raise FileTooLarge()

    pool = pool or await get_pool()
    session_id = await _active_session_id(user_id, pool=pool)

    with tempfile.NamedTemporaryFile(suffix=Path(name).suffix, delete=False) as tmp:
        tmp.write(data)
        tmp_path = tmp.name
    try:
        source = (
            telegram_file_source(tmp_path, name, user_id)
            if origin == "telegram"
            else FileSource(tmp_path, source_path=f"file:{user_id}:{name}", title=Path(name).stem)
        )
        result = await ingest_source(source, session_id=session_id, pool=pool)
    finally:
        os.unlink(tmp_path)

    if session_id is not None:
        await note_document(session_id, pool=pool)
    return result


async def ingest_web_url(
    user_id: int, url: str, *, pool: asyncpg.Pool | None = None
) -> IngestResult:
    """Fetch and ingest a web page, attaching it to the user's active session."""
    pool = pool or await get_pool()
    session_id = await _active_session_id(user_id, pool=pool)
    result = await ingest_source(UrlSource(url), session_id=session_id, pool=pool)
    if session_id is not None:
        await note_document(session_id, pool=pool)
    return result


async def toggle_capture_session(
    user_id: int, *, pool: asyncpg.Pool | None = None
) -> SessionToggleResult:
    """Open the user's ingest session if none is active, otherwise close it."""
    pool = pool or await get_pool()
    return await toggle_session(user_id, pool=pool)


async def buffer_message(user_id: int, text: str, *, pool: asyncpg.Pool | None = None) -> int:
    """Buffer a text message into the active session and return its ``seq``.

    Raises :class:`NoActiveSession` if the user has no open ingest session.
    """
    pool = pool or await get_pool()
    await maybe_autoclose(user_id, pool=pool)
    active = await get_active_session(user_id, pool=pool)
    if active is None:
        raise NoActiveSession()
    return await append_text(int(active["id"]), text, pool=pool)
