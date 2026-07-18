"""Source adapter contract (spec section 11).

Every source (file, URL, Telegram attachment) is normalised to plain text by an
adapter and handed to the single ingestion pipeline. ``raw_content`` (the text)
is the stored representation of the document.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

SourceKind = str  # 'file' | 'url'


@dataclass(slots=True)
class LoadedDoc:
    """A document loaded from a source, ready for the pipeline."""

    source_path: str
    title: str
    text: str
    source_kind: SourceKind = "file"


@runtime_checkable
class Source(Protocol):
    """Anything that can produce a :class:`LoadedDoc`."""

    async def load(self) -> LoadedDoc:
        """Fetch/parse the source and return its normalised text."""
        ...
