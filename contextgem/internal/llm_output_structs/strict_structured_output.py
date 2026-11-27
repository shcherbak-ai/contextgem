# SPDX-FileCopyrightText: Shcherbak AI AS
# SPDX-License-Identifier: Apache-2.0

"""Build strict JSON Schema response formats for extraction requests."""

from __future__ import annotations

from collections.abc import Sequence
from functools import reduce
from operator import or_
from typing import Any

from pydantic import TypeAdapter

from contextgem.internal.base.concepts import _Concept
from contextgem.internal.llm_output_structs.aspect_structs import (
    _get_aspect_extraction_output_struct,
)
from contextgem.internal.llm_output_structs.concept_structs import (
    _get_concept_extraction_output_struct,
)
from contextgem.internal.typings.types import ReferenceDepth


def _strict_value_type(concept: _Concept) -> Any:
    """Return the JSON value type emitted for a concept extraction."""
    from contextgem.internal.base.concepts import (
        _DateConcept,
        _JsonObjectConcept,
        _LabelConcept,
    )
    from contextgem.internal.llm_output_structs.concept_structs import (
        _LabelConceptItemValueModel,
    )

    if isinstance(concept, _DateConcept):
        # The extractor returns the formatted date string; ContextGem parses it
        # into ``date`` after validating the response.
        return str
    if isinstance(concept, _JsonObjectConcept):
        return concept._get_structure_validator()
    if isinstance(concept, _LabelConcept):
        return _LabelConceptItemValueModel

    return concept._item_class.model_fields["value"].annotation


def _replace_value_schemas(
    schema: dict[str, Any], value_schema: dict[str, Any]
) -> None:
    """Replace generic concept values in a ContextGem extraction schema."""

    def is_unconstrained(value: Any) -> bool:
        """Recognize Pydantic's placeholder schemas for ``Any`` values."""
        return isinstance(value, dict) and set(value).issubset(
            {"title", "description", "default"}
        )

    properties = schema.get("properties")
    if isinstance(properties, dict):
        if is_unconstrained(properties.get("value")):
            properties["value"] = value_schema
        extracted_items = properties.get("extracted_items")
        if isinstance(extracted_items, dict):
            item_schema = extracted_items.get("items")
            if isinstance(item_schema, dict):
                item_properties = item_schema.get("properties")
                if isinstance(item_properties, dict) and "value" in item_properties:
                    item_properties["value"] = value_schema
                elif is_unconstrained(item_schema):
                    extracted_items["items"] = value_schema

    for value in schema.values():
        if isinstance(value, dict):
            _replace_value_schemas(value, value_schema)
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    _replace_value_schemas(item, value_schema)


def _rename_schema_refs(value: Any, names: dict[str, str]) -> Any:
    """Namespace Pydantic definitions added for concept value schemas."""
    if isinstance(value, dict):
        return {
            key: (
                f"#/$defs/{names[item[8:]]}"
                if key == "$ref"
                and isinstance(item, str)
                and item.startswith("#/$defs/")
                and item[8:] in names
                else _rename_schema_refs(item, names)
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_rename_schema_refs(item, names) for item in value]
    return value


def _normalize_strict_schema(schema: dict[str, Any], path: str = "schema") -> None:
    """Check the generated Pydantic schema against the strict object contract."""
    if schema.get("type") == "object":
        properties = schema.get("properties", {})
        if not isinstance(properties, dict):
            raise ValueError(f"Unsupported strict output schema at `{path}`")
        if schema.get("additionalProperties", False) is not False:
            raise ValueError(
                "Strict structured output does not support arbitrary object keys "
                f"at `{path}`. Use a JsonObjectConcept with fixed fields."
            )
        schema["additionalProperties"] = False
        required = schema.get("required", [])
        if set(required) != set(properties):
            raise ValueError(
                "Strict structured output requires every object field to be "
                f"required at `{path}`. Optional JSON object fields are not supported."
            )

    for key, value in tuple(schema.items()):
        if key in {"title", "description", "default"}:
            schema.pop(key)
        elif key in {
            "properties",
            "$defs",
            "definitions",
            "patternProperties",
        } and isinstance(value, dict):
            for child_name, child_schema in tuple(value.items()):
                if isinstance(child_schema, dict):
                    _normalize_strict_schema(child_schema, f"{path}.{key}.{child_name}")
        elif isinstance(value, dict):
            _normalize_strict_schema(value, f"{path}.{key}")
        elif isinstance(value, list):
            for index, item in enumerate(value):
                if isinstance(item, dict):
                    _normalize_strict_schema(item, f"{path}.{key}[{index}]")


def _set_identifier_enum(
    schema: dict[str, Any], field_name: str, identifiers: Sequence[str]
) -> None:
    """Constrain IDs in the remote schema to those shown in the prompt."""
    properties = schema.get("properties")
    if isinstance(properties, dict):
        field_schema = properties.get(field_name)
        if isinstance(field_schema, dict):
            field_schema["enum"] = list(identifiers)
    for value in schema.values():
        if isinstance(value, dict):
            _set_identifier_enum(value, field_name, identifiers)
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    _set_identifier_enum(item, field_name, identifiers)


def _build_strict_extraction_response_format(
    *,
    extracted_item_type: str,
    concepts: Sequence[_Concept],
    identifiers: Sequence[str],
    add_justifications: bool,
    add_references: bool,
    reference_depth: ReferenceDepth,
) -> dict[str, Any]:
    """Build LiteLLM's OpenAI-compatible strict JSON Schema response format."""
    with_extra_data = add_justifications or add_references
    if extracted_item_type == "aspect":
        root_model = _get_aspect_extraction_output_struct(
            with_extra_data=add_justifications,
            reference_depth=reference_depth,
        )
        value_schema = None
    elif extracted_item_type == "concept":
        root_model = _get_concept_extraction_output_struct(
            with_extra_data=with_extra_data,
            with_references=add_references,
            reference_depth=reference_depth,
            with_justification=add_justifications,
        )
        value_types = list(
            dict.fromkeys(_strict_value_type(concept) for concept in concepts)
        )
        if not value_types or Any in value_types:
            raise ValueError(
                "Strict structured output cannot build a value schema for the "
                "concepts in this extraction request."
            )
        value_type = (
            value_types[0] if len(value_types) == 1 else reduce(or_, value_types)
        )
        value_schema = TypeAdapter(value_type).json_schema()
    else:
        raise ValueError(f"Unsupported extraction type `{extracted_item_type}`")

    root_schema = root_model.model_json_schema()
    if root_schema.get("type") != "array":
        raise ValueError("ContextGem extraction schema must have an array at its root")

    definitions = root_schema.pop("$defs", {})
    root_definition_names = set(definitions)
    if value_schema is not None:
        value_definitions = value_schema.pop("$defs", {})
        if value_definitions:
            renamed = {name: f"StrictValue{name}" for name in value_definitions}
            for name, definition in value_definitions.items():
                definitions[renamed[name]] = _rename_schema_refs(definition, renamed)
            value_schema = _rename_schema_refs(value_schema, renamed)
        _replace_value_schemas(root_schema["items"], value_schema)
        for definition in definitions.values():
            _replace_value_schemas(definition, value_schema)

    id_field = "aspect_id" if extracted_item_type == "aspect" else "concept_id"
    _set_identifier_enum(root_schema["items"], id_field, identifiers)
    for name in root_definition_names:
        _set_identifier_enum(definitions[name], id_field, identifiers)

    schema: dict[str, Any] = {
        "type": "object",
        "properties": {"items": root_schema},
        "required": ["items"],
        "additionalProperties": False,
    }
    if definitions:
        schema["$defs"] = definitions
    _normalize_strict_schema(schema)

    return {
        "type": "json_schema",
        "json_schema": {
            "name": "contextgem_extraction",
            "strict": True,
            "schema": schema,
        },
    }


def _unwrap_strict_extraction_response(
    parsed: dict | list | None,
) -> dict | list | None:
    """Unwrap the object required by strict Structured Outputs."""
    if not isinstance(parsed, dict) or set(parsed) != {"items"}:
        return None
    items = parsed["items"]
    return items if isinstance(items, list) else None
