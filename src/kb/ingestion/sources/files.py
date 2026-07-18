"""File source adapter (spec section 11).

Dispatch by extension to a plain-text representation:

* ``.txt`` / ``.md`` — read directly;
* ``.html`` — trafilatura (navigation stripped);
* ``.pdf`` / ``.docx`` / ``.pptx`` — docling -> markdown (the ``parse`` extra;
  ML models load lazily, only PDF needs them).

Telegram attachments reuse this adapter but store a ``tg:<filename>`` source path
so re-sending a file with the same name becomes a new version of one document.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from kb.ingestion.sources.base import LoadedDoc

TEXT_EXTENSIONS = {".txt", ".md", ".markdown"}
HTML_EXTENSIONS = {".html", ".htm"}
DOCLING_EXTENSIONS = {".pdf", ".docx", ".pptx"}
SUPPORTED_EXTENSIONS = TEXT_EXTENSIONS | HTML_EXTENSIONS | DOCLING_EXTENSIONS


def is_supported(filename: str) -> bool:
    """Return whether a filename's extension is supported."""
    return Path(filename).suffix.lower() in SUPPORTED_EXTENSIONS


def _parse_html(path: Path) -> str:
    import trafilatura

    html = path.read_text(encoding="utf-8", errors="ignore")
    text = trafilatura.extract(html) or ""
    if not text.strip():
        raise ValueError(f"не удалось извлечь текст из HTML: {path.name}")
    return text


def _parse_with_docling(path: Path) -> str:  # pragma: no cover - heavy optional dep
    try:
        from docling.document_converter import DocumentConverter
    except ImportError as exc:
        raise RuntimeError("Парсинг PDF/DOCX/PPTX требует extra 'parse' (docling).") from exc

    converter = DocumentConverter()
    result = converter.convert(str(path))
    return str(result.document.export_to_markdown())


def _parse_file(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in TEXT_EXTENSIONS:
        return path.read_text(encoding="utf-8")
    if suffix in HTML_EXTENSIONS:
        return _parse_html(path)
    if suffix in DOCLING_EXTENSIONS:
        return _parse_with_docling(path)
    raise ValueError(f"неподдерживаемое расширение: {suffix or path.name}")


class FileSource:
    """Load a document from a local file path."""

    def __init__(
        self,
        path: str | Path,
        *,
        source_path: str | None = None,
        title: str | None = None,
    ) -> None:
        self._path = Path(path)
        self._source_path = source_path or str(self._path)
        self._title = title or self._path.stem

    async def load(self) -> LoadedDoc:
        text = await asyncio.to_thread(_parse_file, self._path)
        return LoadedDoc(
            source_path=self._source_path,
            title=self._title,
            text=text,
            source_kind="file",
        )


def telegram_file_source(local_path: str | Path, original_name: str) -> FileSource:
    """Build a FileSource for a Telegram attachment with a ``tg:`` source path."""
    return FileSource(
        local_path,
        source_path=f"tg:{original_name.lower()}",
        title=Path(original_name).stem,
    )
