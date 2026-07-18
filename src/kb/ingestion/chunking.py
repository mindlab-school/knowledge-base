"""Chunking strategies by ``doc_type`` (spec section 3).

All functions here are pure (no I/O) so they are cheap to unit-test. Common
limits: ``CHUNK_SIZE`` characters with ``CHUNK_OVERLAP`` of trailing context, and
a markdown table is never split across chunks.
"""

from __future__ import annotations

import re
from typing import Any

CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200

_HEADING_RE = re.compile(r"^#{1,6}\s")
_BLANK_LINE_RE = re.compile(r"\n\s*\n")


def _is_table(block: str) -> bool:
    """A block is treated as a markdown table if any line starts with ``|``."""
    return any(line.lstrip().startswith("|") for line in block.splitlines())


def _hard_wrap(block: str, size: int) -> list[str]:
    """Split an oversized non-table block on whitespace boundaries into <=size pieces."""
    words = block.split()
    pieces: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if current and len(candidate) > size:
            pieces.append(current)
            current = word
        else:
            current = candidate
    if current:
        pieces.append(current)
    return pieces or [block]


def _atomic_units(text: str, size: int) -> list[str]:
    """Split text into atomic units: paragraphs and whole tables.

    Tables stay intact even when longer than ``size``; oversized plain paragraphs
    are hard-wrapped.
    """
    units: list[str] = []
    for raw in _BLANK_LINE_RE.split(text.strip()):
        block = raw.strip()
        if not block:
            continue
        if _is_table(block) or len(block) <= size:
            units.append(block)
        else:
            units.extend(_hard_wrap(block, size))
    return units


def _pack(units: list[str], size: int, overlap: int) -> list[str]:
    """Pack atomic units into chunks of at most ``size`` chars with unit-level overlap."""
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for unit in units:
        addition = len(unit) + (2 if current else 0)
        if current and current_len + addition > size:
            chunks.append("\n\n".join(current))
            carry: list[str] = []
            carry_len = 0
            for previous in reversed(current):
                if carry_len + len(previous) > overlap:
                    break
                carry.insert(0, previous)
                carry_len += len(previous) + 2
            current = carry
            current_len = sum(len(u) + 2 for u in current)
        current.append(unit)
        current_len += addition

    if current:
        chunks.append("\n\n".join(current))
    return chunks


def _split_headings(text: str) -> list[str]:
    """Split markdown text into sections, each starting at a heading line."""
    sections: list[str] = []
    current: list[str] = []
    for line in text.splitlines():
        if _HEADING_RE.match(line) and current:
            sections.append("\n".join(current))
            current = [line]
        else:
            current.append(line)
    if current:
        sections.append("\n".join(current))
    return [section.strip() for section in sections if section.strip()]


def _chunk_generic(text: str, size: int, overlap: int) -> list[str]:
    return _pack(_atomic_units(text, size), size, overlap)


def _chunk_by_headings(text: str, size: int, overlap: int) -> list[str]:
    """Chunk by markdown headings, falling back to paragraphs when there are none."""
    sections = _split_headings(text)
    if len(sections) <= 1:
        return _chunk_generic(text, size, overlap)
    chunks: list[str] = []
    for section in sections:
        if len(section) <= size:
            chunks.append(section)
        else:
            chunks.extend(_pack(_atomic_units(section, size), size, overlap))
    return chunks


def _chunk_faq(text: str, size: int, overlap: int) -> list[str]:
    """One question/answer pair per chunk (pairs are not size-capped)."""
    sections = _split_headings(text)
    if len(sections) > 1:
        return sections
    return [block.strip() for block in _BLANK_LINE_RE.split(text.strip()) if block.strip()]


def _meeting_header(attributes: dict[str, Any]) -> str:
    parts: list[str] = []
    meeting_date = attributes.get("meeting_date")
    if meeting_date:
        parts.append(f"Дата: {meeting_date}")
    participants = attributes.get("participants") or []
    if participants:
        parts.append("Участники: " + ", ".join(str(p) for p in participants))
    return "; ".join(parts)


def _meeting_chunk(header: str, label: str, lines: list[str]) -> str:
    prefix = f"{header}\n\n" if header else ""
    return f"{prefix}{label}:\n" + "\n".join(lines)


def _chunk_meeting_notes(
    text: str, attributes: dict[str, Any], size: int, overlap: int
) -> list[str]:
    """Chunk by decision / action-item blocks; date and participants repeat in each chunk."""
    decisions = [str(d) for d in (attributes.get("decisions") or [])]
    action_items = [str(a) for a in (attributes.get("action_items") or [])]
    if not decisions and not action_items:
        return _chunk_by_headings(text, size, overlap)

    header = _meeting_header(attributes)
    base_len = len(header) + 8
    chunks: list[str] = []
    for label, items in (("Решения", decisions), ("Поручения", action_items)):
        if not items:
            continue
        current: list[str] = []
        current_len = base_len + len(label)
        for item in items:
            line = f"- {item}"
            if current and current_len + len(line) + 1 > size:
                chunks.append(_meeting_chunk(header, label, current))
                current = []
                current_len = base_len + len(label)
            current.append(line)
            current_len += len(line) + 1
        if current:
            chunks.append(_meeting_chunk(header, label, current))
    return chunks


def chunk_document(
    text: str,
    doc_type: str,
    attributes: dict[str, Any] | None = None,
    *,
    size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """Chunk a document according to its type.

    :param text: The document's plain-text / markdown content.
    :param doc_type: Registered document type; unknown types use the generic strategy.
    :param attributes: Extracted attributes (used by the ``meeting_notes`` strategy).
    :returns: Ordered list of chunk strings (empty for blank input).
    """
    attributes = attributes or {}
    if not text.strip():
        return []

    if doc_type == "faq":
        chunks = _chunk_faq(text, size, overlap)
    elif doc_type == "meeting_notes":
        chunks = _chunk_meeting_notes(text, attributes, size, overlap)
    elif doc_type in ("regulation", "instruction"):
        chunks = _chunk_by_headings(text, size, overlap)
    else:
        chunks = _chunk_generic(text, size, overlap)

    return chunks or [text.strip()]
