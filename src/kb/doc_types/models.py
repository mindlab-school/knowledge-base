"""Pydantic models for document types (spec section 2).

One model per ``doc_type``. Every field is optional so a partial extraction is
still valid. The model docstring is the human/LLM description of the type used by
the ingestion classifier; the field schema (``model_json_schema``) is fed to the
structured-output call.
"""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel


class GenericDoc(BaseModel):
    """Произвольный документ без выраженной структуры: заметка, письмо, справка."""

    summary: str | None = None
    language: str | None = None


class RegulationDoc(BaseModel):
    """Регламент или положение: обязательные правила, владелец, срок пересмотра, статус."""

    owner: str | None = None
    department: str | None = None
    effective_date: date | None = None
    review_by: date | None = None
    status: Literal["active", "draft", "deprecated"] | None = None
    summary: str | None = None


class InstructionDoc(BaseModel):
    """Инструкция/руководство: как выполнить задачу, для кого, какие системы затрагивает."""

    topic: str | None = None
    audience: str | None = None
    systems: list[str] | None = None
    summary: str | None = None


class FaqDoc(BaseModel):
    """Подборка «вопрос — ответ» по одной теме."""

    topic: str | None = None
    questions: list[str] | None = None


class MeetingNotesDoc(BaseModel):
    """Протокол встречи: дата, участники, принятые решения, поручения, проект."""

    meeting_date: date | None = None
    participants: list[str] | None = None
    decisions: list[str] | None = None
    action_items: list[str] | None = None
    project: str | None = None
