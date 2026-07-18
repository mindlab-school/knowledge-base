"""Registry of document types (spec section 2).

The registry maps a ``doc_type`` string to its Pydantic model. Adding a type is:
model + docstring + one registry entry, then ``reextract --type generic``.
Allowed ``doc_type`` values are enforced here (application-side), not by a DB
CHECK constraint.
"""

from __future__ import annotations

import inspect
from typing import Any

from pydantic import BaseModel

from kb.doc_types.models import (
    FaqDoc,
    GenericDoc,
    InstructionDoc,
    MeetingNotesDoc,
    RegulationDoc,
)

GENERIC_TYPE = "generic"

DOC_TYPE_REGISTRY: dict[str, type[BaseModel]] = {
    GENERIC_TYPE: GenericDoc,
    "regulation": RegulationDoc,
    "instruction": InstructionDoc,
    "faq": FaqDoc,
    "meeting_notes": MeetingNotesDoc,
}


def is_known_type(doc_type: str) -> bool:
    """Return whether ``doc_type`` is registered."""
    return doc_type in DOC_TYPE_REGISTRY


def get_model(doc_type: str) -> type[BaseModel]:
    """Return the Pydantic model for ``doc_type`` (``generic`` if unknown)."""
    return DOC_TYPE_REGISTRY.get(doc_type, GenericDoc)


def type_descriptions() -> dict[str, str]:
    """Return ``{doc_type: description}`` from each model's docstring."""
    return {name: inspect.getdoc(model) or "" for name, model in DOC_TYPE_REGISTRY.items()}


def attribute_schema(doc_type: str) -> dict[str, Any]:
    """Return the JSON schema of the attributes for ``doc_type``."""
    return get_model(doc_type).model_json_schema()


def attribute_schemas() -> dict[str, dict[str, Any]]:
    """Return ``{doc_type: json_schema}`` for every registered type."""
    return {name: model.model_json_schema() for name, model in DOC_TYPE_REGISTRY.items()}
