"""KB Agent MCP server (spec section 15.1).

A thin stdio MCP server (FastMCP) that exposes the knowledge base to Claude Code.
It has no logic and no database access of its own: every tool is a single HTTP
call to the backend. Identity and configuration come from the environment
(``KB_BACKEND_URL``, ``KB_USER_ID``, ``KB_SHARED_SECRET``).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import httpx
from mcp.server.fastmcp import FastMCP

BACKEND_URL = os.environ.get("KB_BACKEND_URL", "http://localhost:8000")
USER_ID = int(os.environ.get("KB_USER_ID", "0"))
_SHARED_SECRET = os.environ.get("KB_SHARED_SECRET", "")
HEADERS: dict[str, str] | None = {"X-KB-Secret": _SHARED_SECRET} if _SHARED_SECRET else None
TIMEOUT = 300.0

mcp = FastMCP("kb-agent")


def _render(response: httpx.Response) -> str:
    if response.status_code >= 400:
        return f"Ошибка backend {response.status_code}: {response.text}"
    return response.text


async def _post(path: str, payload: dict[str, Any]) -> str:
    async with httpx.AsyncClient(base_url=BACKEND_URL, timeout=TIMEOUT, headers=HEADERS) as client:
        return _render(await client.post(path, json=payload))


async def _get(path: str, params: dict[str, Any]) -> str:
    async with httpx.AsyncClient(base_url=BACKEND_URL, timeout=TIMEOUT, headers=HEADERS) as client:
        return _render(await client.get(path, params=params))


async def kb_search(query: str, filters: dict[str, Any] | None = None) -> str:
    """Семантический поиск фрагментов по содержанию базы знаний. Возвращает источники."""
    return await _post("/search", {"telegram_id": USER_ID, "query": query, "filters": filters})


async def kb_query_documents(filters: dict[str, Any]) -> str:
    """Структурированный поиск документов по метаданным: тип, атрибуты, даты, сущности."""
    return await _post("/search/documents", {"telegram_id": USER_ID, "filters": filters})


async def kb_explore_entity(name: str) -> str:
    """Карточка сущности (человек/отдел/продукт/процесс/проект): документы и граф связей."""
    return await _post("/search/entity", {"telegram_id": USER_ID, "name": name})


async def kb_get_document(document_id: int) -> str:
    """Полный текст документа по его id."""
    return await _get(f"/documents/{document_id}", {"telegram_id": USER_ID})


async def kb_list_facts() -> str:
    """Актуальные факты компании: цены, расписание, условия, контакты."""
    return await _get("/facts", {"telegram_id": USER_ID})


async def kb_add_url(url: str) -> str:
    """Загрузить веб-страницу в базу знаний как документ."""
    return await _post("/ingest/url", {"telegram_id": USER_ID, "url": url})


async def kb_add_file(path: str) -> str:
    """Загрузить локальный файл (pdf/docx/pptx/md/txt/html) в базу знаний."""
    file_path = Path(path)
    if not file_path.exists():
        return f"Файл не найден: {path}"
    data = file_path.read_bytes()
    async with httpx.AsyncClient(base_url=BACKEND_URL, timeout=TIMEOUT, headers=HEADERS) as client:
        response = await client.post(
            "/ingest/file",
            data={"telegram_id": str(USER_ID), "filename": file_path.name},
            files={"file": (file_path.name, data)},
        )
    return _render(response)


for _tool in (
    kb_search,
    kb_query_documents,
    kb_explore_entity,
    kb_get_document,
    kb_list_facts,
    kb_add_url,
    kb_add_file,
):
    mcp.add_tool(_tool)


def main() -> None:
    """Run the MCP server over stdio."""
    mcp.run()


if __name__ == "__main__":
    main()
