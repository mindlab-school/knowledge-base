"""Unit tests for the ingest service's validation guards and error messages.

The exact Russian detail strings are load-bearing: handlers surface them verbatim
as 400/413/409 responses, so these lock the wording in place. The validation runs
before any pool acquisition, so both guards are exercisable without a database.
"""

from __future__ import annotations

import pytest

from kb.services import ingest as ingest_service


def test_domain_error_messages_are_exact() -> None:
    assert (
        str(ingest_service.UnsupportedFileType("отчёт.zip"))
        == "неподдерживаемый тип файла: отчёт.zip"
    )
    assert str(ingest_service.FileTooLarge()) == "файл превышает 20 MB"
    assert str(ingest_service.NoActiveSession()) == "нет активной сессии набора"


async def test_unsupported_file_rejected_before_db() -> None:
    with pytest.raises(ingest_service.UnsupportedFileType):
        await ingest_service.ingest_uploaded_file(1, b"data", "malware.exe")


async def test_oversize_file_rejected_before_db() -> None:
    oversize = b"0" * (ingest_service.MAX_FILE_SIZE + 1)
    with pytest.raises(ingest_service.FileTooLarge):
        await ingest_service.ingest_uploaded_file(1, oversize, "notes.txt")
