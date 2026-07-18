"""System prompt and the per-request facts block (spec section 5).

The static prefix (tool schemas + system prompt + facts block) is placed before
history and the question so it can be prompt-cached. The facts block lists every
active fact and is capped at ``FACTS_BLOCK_LIMIT`` characters.
"""

from __future__ import annotations

from typing import Any

SYSTEM_PROMPT = """\
Ты — корпоративный ассистент базы знаний. Отвечай на русском языке строго по данным базы.

Правила:
- Вопросы, которые покрыты блоком АКТУАЛЬНЫЕ ФАКТЫ (цены, пакеты, условия, расписание,
  контакты), — отвечай напрямую из фактов, без поисковых инструментов.
- Остальные вопросы — отвечай только по документам базы и всегда указывай документы-источники.
- Если ответа в базе нет — прямо скажи «в базе знаний этого нет». Не придумывай и не используй
  общие знания модели или интернет.
- Вопросы про наборы документов, статусы, даты (например «какие регламенты просрочены») —
  сначала вызывай query_documents с подходящими фильтрами.
- Вопросы про людей, отделы, процессы, проекты — вызывай explore_entity.
- Вопросы вида «что связано с X», «кто отвечает за Y», «что затрагивает изменение Z» —
  вызывай explore_entity с depth=2 (один обход графа вместо серии поисков).
- Для содержательных вопросов используй search_knowledge_base; при необходимости открывай
  документ целиком через get_document перед формулировкой ответа.
- В ответах на структурированные запросы называй применённые фильтры.
- Длинные ответы формулируй кратко и по существу."""

FACTS_HEADER = "АКТУАЛЬНЫЕ ФАКТЫ:"

_CHARS_PER_TOKEN = 4


def build_facts_block(facts: list[dict[str, Any]], limit: int) -> tuple[str, bool]:
    """Render the facts block, truncated to ``limit`` characters.

    :returns: ``(block, truncated)`` where ``block`` is empty when there are no
        facts and ``truncated`` flags that the limit was hit.
    """
    if not facts:
        return "", False
    lines = [FACTS_HEADER]
    lines.extend(f"[{fact['topic']}]: {fact['content']}" for fact in facts)
    block = "\n".join(lines)
    if len(block) <= limit:
        return block, False
    return block[:limit], True


def build_system_content(facts_block: str) -> str:
    """Concatenate the system prompt with the facts block (if any)."""
    if not facts_block:
        return SYSTEM_PROMPT
    return f"{SYSTEM_PROMPT}\n\n{facts_block}"


def estimate_tokens(text: str) -> int:
    """Rough token estimate (~4 characters per token) for the cache threshold."""
    return len(text) // _CHARS_PER_TOKEN
