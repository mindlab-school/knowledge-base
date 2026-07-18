"""Unit tests for chunking strategies (spec section 3)."""

from __future__ import annotations

from kb.ingestion.chunking import CHUNK_SIZE, chunk_document


def test_empty_text_returns_no_chunks() -> None:
    assert chunk_document("   \n  ", "generic") == []


def test_generic_splits_paragraphs() -> None:
    text = "Первый абзац.\n\nВторой абзац.\n\nТретий абзац."
    chunks = chunk_document(text, "generic", size=40, overlap=5)
    assert len(chunks) >= 2
    assert "Первый абзац." in chunks[0]


def test_regulation_splits_by_headings() -> None:
    text = "# Заголовок один\nтекст один\n\n## Заголовок два\nтекст два"
    chunks = chunk_document(text, "regulation")
    assert len(chunks) == 2
    assert chunks[0].startswith("# Заголовок один")
    assert chunks[1].startswith("## Заголовок два")


def test_faq_one_pair_per_chunk() -> None:
    text = (
        "### Как оформить отпуск?\nНужно подать заявление.\n\n"
        "### Сколько дней отпуска?\n28 календарных дней."
    )
    chunks = chunk_document(text, "faq")
    assert len(chunks) == 2
    assert "Как оформить отпуск?" in chunks[0]
    assert "Сколько дней отпуска?" in chunks[1]


def test_meeting_notes_repeats_header_per_block() -> None:
    attributes = {
        "meeting_date": "2026-05-12",
        "participants": ["Иван", "Мария"],
        "decisions": ["Утвердили план", "Согласовали бюджет"],
        "action_items": ["Иван готовит отчёт"],
    }
    chunks = chunk_document("тело протокола", "meeting_notes", attributes)
    assert any("Решения:" in chunk for chunk in chunks)
    assert any("Поручения:" in chunk for chunk in chunks)
    for chunk in chunks:
        assert "2026-05-12" in chunk
        assert "Иван" in chunk


def test_markdown_table_not_split() -> None:
    table = "\n".join(f"| a{i} | b{i} |" for i in range(60))
    text = f"Вступление.\n\n{table}\n\nЗаключение."
    chunks = chunk_document(text, "generic", size=200, overlap=20)
    assert any(table in chunk for chunk in chunks)


def test_generic_respects_size_for_plain_text() -> None:
    paragraph = " ".join(["слово"] * 800)
    chunks = chunk_document(paragraph, "generic")
    assert all(len(chunk) <= CHUNK_SIZE + 50 for chunk in chunks)
    assert len(chunks) > 1
