"""LLM structure extraction for ingestion (spec section 3).

One model call classifies the document type and extracts attributes, entities,
relations and document links. Structured outputs are requested via
``response_format=json_schema``; if the model does not support that, the same
call is retried once through a forced ``emit_extraction`` tool, and the chosen
path is cached for the process. Failures degrade gracefully: at most one retry,
then ``generic``/``failed`` so the pipeline never dies (invariant 4).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import openai
from pydantic import ValidationError

from kb.config import Settings, get_settings
from kb.doc_types.registry import GENERIC_TYPE, get_model, is_known_type
from kb.llm.client import LLMClient

CONFIDENCE_MIN = 0.6
_MAX_INPUT_CHARS = 15_000
_INPUT_HEAD = 12_000
_INPUT_TAIL = 3_000
_MAX_OUTPUT_TOKENS = 1500

ENTITY_TYPES = ("person", "department", "product", "process", "project")
RELATION_TYPES = ("owns", "part_of", "reports_to", "responsible_for", "related_to")
ROLE_TYPES = ("owner", "participant", "mentioned", "approver")

# Process-wide cache of whether the extraction model supports structured outputs.
# None = unknown (try structured first); True/False = decided.
_structured_supported: bool | None = None


def reset_structured_support() -> None:
    """Reset the cached structured-output capability (used by tests)."""
    global _structured_supported
    _structured_supported = None


@dataclass(slots=True)
class ExtractedEntity:
    """An entity mention produced by extraction."""

    entity_type: str
    name: str
    role: str | None = None


@dataclass(slots=True)
class ExtractedRelation:
    """A directed relation between two entity names produced by extraction."""

    source: str
    target: str
    relation: str


@dataclass(slots=True)
class ExtractionResult:
    """Outcome of extracting one document."""

    doc_type: str
    confidence: float
    attributes: dict[str, Any]
    entities: list[ExtractedEntity] = field(default_factory=list)
    relations: list[ExtractedRelation] = field(default_factory=list)
    links: list[str] = field(default_factory=list)
    status: str = "done"
    model: str | None = None


_ENVELOPE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "doc_type": {"type": "string"},
        "confidence": {"type": "number"},
        "attributes": {"type": "object"},
        "entities": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "entity_type": {"type": "string"},
                    "name": {"type": "string"},
                    "role": {"type": ["string", "null"]},
                },
                "required": ["entity_type", "name"],
            },
        },
        "relations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "source": {"type": "string"},
                    "target": {"type": "string"},
                    "relation": {"type": "string"},
                },
                "required": ["source", "target", "relation"],
            },
        },
        "links": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["doc_type", "confidence", "attributes", "entities", "relations", "links"],
}


def _truncate(text: str) -> str:
    """Keep the head and tail of very long documents (enough to classify)."""
    if len(text) <= _MAX_INPUT_CHARS:
        return text
    return f"{text[:_INPUT_HEAD]}\n…\n{text[-_INPUT_TAIL:]}"


def _system_prompt() -> str:
    from kb.doc_types.registry import attribute_schemas, type_descriptions

    descriptions = type_descriptions()
    schemas = attribute_schemas()
    lines = [
        "Ты извлекаешь структуру из корпоративного документа на русском языке.",
        "Определи тип документа из списка ниже и извлеки атрибуты строго по схеме этого типа.",
        "Дополнительно извлеки сущности (люди, отделы, продукты, процессы, проекты) с ролью,",
        "связи между сущностями и ссылки (названия/URL) на другие документы.",
        "Верни ТОЛЬКО JSON требуемой структуры, без пояснений и разметки.",
        "",
        "Типы документов и схемы их атрибутов:",
    ]
    for name, description in descriptions.items():
        properties = schemas[name].get("properties", {})
        lines.append(f"- {name}: {description}")
        lines.append(f"  атрибуты: {json.dumps(properties, ensure_ascii=False)}")
    lines += [
        "",
        f"entity_type ∈ {{{', '.join(ENTITY_TYPES)}}}.",
        f"role ∈ {{{', '.join(ROLE_TYPES)}}} или null.",
        f"relation ∈ {{{', '.join(RELATION_TYPES)}}}.",
        "confidence — уверенность в типе документа от 0 до 1.",
        "source и target в relations — имена из списка entities.",
    ]
    return "\n".join(lines)


def _build_messages(text: str) -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": _system_prompt()},
        {"role": "user", "content": f"Документ:\n\n{_truncate(text)}"},
    ]


async def _call(llm: LLMClient, settings: Settings, messages: list[dict[str, Any]]) -> str:
    """Return the raw JSON string, using structured outputs or the tool fallback."""
    global _structured_supported

    if _structured_supported is not False:
        try:
            response = await llm.chat(
                model=settings.extraction_model,
                messages=messages,
                max_tokens=_MAX_OUTPUT_TOKENS,
                response_format={
                    "type": "json_schema",
                    "json_schema": {"name": "extraction", "schema": _ENVELOPE_SCHEMA},
                },
            )
            _structured_supported = True
            return response.choices[0].message.content or ""
        except openai.BadRequestError:
            # Model/provider rejected structured outputs — fall back for the process.
            _structured_supported = False

    tool = {
        "type": "function",
        "function": {
            "name": "emit_extraction",
            "description": "Вернуть извлечённую структуру документа.",
            "parameters": _ENVELOPE_SCHEMA,
        },
    }
    response = await llm.chat(
        model=settings.extraction_model,
        messages=messages,
        max_tokens=_MAX_OUTPUT_TOKENS,
        tools=[tool],
        tool_choice={"type": "function", "function": {"name": "emit_extraction"}},
    )
    message = response.choices[0].message
    for tool_call in message.tool_calls or []:
        if tool_call.type == "function":
            return tool_call.function.arguments or ""
    return message.content or ""


def _resolve_type(value: Any) -> str:
    return value if isinstance(value, str) and is_known_type(value) else GENERIC_TYPE


def _to_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _parse_entities(parsed: dict[str, Any]) -> list[ExtractedEntity]:
    entities: list[ExtractedEntity] = []
    for item in parsed.get("entities") or []:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        entity_type = item.get("entity_type")
        if not name or not entity_type:
            continue
        role = item.get("role")
        entities.append(
            ExtractedEntity(
                entity_type=str(entity_type),
                name=str(name),
                role=str(role) if role else None,
            )
        )
    return entities


def _parse_relations(parsed: dict[str, Any]) -> list[ExtractedRelation]:
    relations: list[ExtractedRelation] = []
    for item in parsed.get("relations") or []:
        if not isinstance(item, dict):
            continue
        source = item.get("source")
        target = item.get("target")
        relation = item.get("relation")
        if source and target and relation:
            relations.append(ExtractedRelation(str(source), str(target), str(relation)))
    return relations


def _parse_links(parsed: dict[str, Any]) -> list[str]:
    return [str(link) for link in (parsed.get("links") or []) if link]


def _validate_attributes(doc_type: str, attributes: Any) -> dict[str, Any]:
    model = get_model(doc_type)
    validated = model.model_validate(attributes if isinstance(attributes, dict) else {})
    return validated.model_dump(mode="json", exclude_none=True)


async def extract_structure(
    text: str,
    *,
    llm: LLMClient | None = None,
    settings: Settings | None = None,
) -> ExtractionResult:
    """Extract type/attributes/entities/relations/links for a document.

    Never raises: LLM or validation failures degrade to ``generic``/``failed``.
    """
    settings = settings or get_settings()
    llm = llm or LLMClient(settings)
    model_name = settings.extraction_model

    messages = _build_messages(text)
    entities: list[ExtractedEntity] = []
    relations: list[ExtractedRelation] = []
    links: list[str] = []

    for attempt in (0, 1):
        error: str | None = None
        try:
            raw = await _call(llm, settings, messages)
        except openai.OpenAIError as exc:
            error = f"ошибка вызова модели: {exc}"
        else:
            try:
                parsed = json.loads(raw)
                if not isinstance(parsed, dict):
                    raise ValueError("ожидался JSON-объект")
            except (json.JSONDecodeError, ValueError) as exc:
                entities, relations, links = [], [], []
                error = f"невалидный JSON: {exc}"
            else:
                entities = _parse_entities(parsed)
                relations = _parse_relations(parsed)
                links = _parse_links(parsed)
                doc_type = _resolve_type(parsed.get("doc_type"))
                confidence = _to_float(parsed.get("confidence"))
                if confidence < CONFIDENCE_MIN:
                    return ExtractionResult(
                        doc_type=GENERIC_TYPE,
                        confidence=confidence,
                        attributes={},
                        entities=entities,
                        relations=relations,
                        links=links,
                        status="done",
                        model=model_name,
                    )
                try:
                    attributes = _validate_attributes(doc_type, parsed.get("attributes"))
                except ValidationError as exc:
                    error = f"ошибка валидации атрибутов: {exc}"
                else:
                    return ExtractionResult(
                        doc_type=doc_type,
                        confidence=confidence,
                        attributes=attributes,
                        entities=entities,
                        relations=relations,
                        links=links,
                        status="done",
                        model=model_name,
                    )

        if attempt == 0:
            messages = [
                *messages,
                {
                    "role": "user",
                    "content": (
                        f"Предыдущий ответ содержал ошибку: {error}. "
                        "Верни корректный JSON строго по требуемой схеме."
                    ),
                },
            ]
            continue

        return ExtractionResult(
            doc_type=GENERIC_TYPE,
            confidence=0.0,
            attributes={},
            entities=entities,
            relations=relations,
            links=links,
            status="failed",
            model=model_name,
        )

    raise AssertionError("unreachable")
