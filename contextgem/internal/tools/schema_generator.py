#
# ContextGem
#
# Copyright 2025 Shcherbak AI AS. All rights reserved. Developed by Sergii Shcherbak.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""
Module for generating OpenAI-compatible tool schemas from Python function signatures.

Converts Python type hints to JSON Schema and generates complete tool definitions
for use with LLM tool calling.
"""

from __future__ import annotations

import inspect
import types
import warnings
from collections.abc import Callable
from typing import (
    Any,
    Literal,
    Union,
    get_args,
    get_origin,
    get_type_hints,
    is_typeddict,
)

from contextgem.internal.tools.docstring_parser import _ParsedDocstring


def _contains_collection_type(
    type_hint: Any,
    target_types: tuple[type, ...],
) -> bool:
    """
    Recursively check if a type hint contains any of the target collection types.

    :param type_hint: The type hint to check.
    :type type_hint: Any
    :param target_types: Tuple of types to look for (e.g., (set, frozenset, tuple)).
    :type target_types: tuple[type, ...]
    :returns: True if any target type is found in the type hint.
    :rtype: bool
    """
    origin = get_origin(type_hint)

    # Check if the origin is one of the target types
    if origin in target_types:
        return True

    # Recursively check type arguments
    args = get_args(type_hint)
    return any(_contains_collection_type(arg, target_types) for arg in args)


# Type mapping from Python types to JSON Schema types
_JSON_SCHEMA_TYPE_MAP: dict[type, dict[str, Any]] = {
    str: {"type": "string"},
    int: {"type": "integer"},
    float: {"type": "number"},
    bool: {"type": "boolean"},
    type(None): {"type": "null"},
}


def _python_type_to_json_schema(
    type_hint: Any,
    description: str | None = None,
) -> dict[str, Any]:
    """
    Convert a Python type hint to a JSON Schema object.

    Supports:
    - Primitive types (str, int, float, bool, None)
    - list[X] -> array with items
    - dict[str, X] -> object with additionalProperties
    - dict -> object
    - Optional[X] -> schema for X (optional handling at parameter level)
    - Union[X, Y] -> anyOf
    - Literal["a", "b"] -> enum
    - TypedDict -> nested object with properties

    :param type_hint: The Python type hint to convert.
    :type type_hint: Any
    :param description: Optional description to include in the schema.
    :type description: str | None
    :returns: JSON Schema object.
    :rtype: dict[str, Any]
    :raises TypeError: If the type hint is not supported.
    """
    schema: dict[str, Any] = {}

    # Handle basic types
    if type_hint in _JSON_SCHEMA_TYPE_MAP:
        schema = _JSON_SCHEMA_TYPE_MAP[type_hint].copy()
        if description:
            schema["description"] = description
        return schema

    # Handle Any type
    if type_hint is Any:
        schema = {}
        if description:
            schema["description"] = description
        return schema

    origin = get_origin(type_hint)
    args = get_args(type_hint)

    # Handle list[X]
    if origin is list:
        item_type = args[0] if args else Any
        schema = {
            "type": "array",
            "items": _python_type_to_json_schema(item_type),
        }
        if description:
            schema["description"] = description
        return schema

    # Handle tuple[X, ...] as array
    if origin is tuple:
        if args:
            # For tuple[X, Y, Z], use items with prefixItems if different types
            # For tuple[X, ...] use array
            if len(args) == 2 and args[1] is ...:
                schema = {
                    "type": "array",
                    "items": _python_type_to_json_schema(args[0]),
                }
            else:
                schema = {
                    "type": "array",
                    "prefixItems": [_python_type_to_json_schema(a) for a in args],
                    "minItems": len(args),
                    "maxItems": len(args),
                }
        else:
            schema = {"type": "array"}
        if description:
            schema["description"] = description
        return schema

    # Handle set[X] as array with uniqueItems
    if origin is set or origin is frozenset:
        item_type = args[0] if args else Any
        schema = {
            "type": "array",
            "items": _python_type_to_json_schema(item_type),
            "uniqueItems": True,
        }
        if description:
            schema["description"] = description
        return schema

    # Handle dict[K, V]
    if origin is dict:
        if len(args) == 2:
            value_type = args[1]
            schema = {
                "type": "object",
                "additionalProperties": _python_type_to_json_schema(value_type),
            }
        else:
            schema = {"type": "object"}
        if description:
            schema["description"] = description
        return schema

    # Handle Union types (including Optional which is Union[X, None])
    # Check for types.UnionType (X | Y syntax in Python 3.10+)
    is_union = origin is Union or isinstance(type_hint, types.UnionType)

    if is_union:
        # Check if this is Optional[X] (Union with None)
        non_none_types = [a for a in args if a is not type(None)]
        has_none = type(None) in args

        if len(non_none_types) == 1 and has_none:
            # This is Optional[X] - return schema for X
            # The "required" handling happens at the parameter level
            return _python_type_to_json_schema(non_none_types[0], description)

        # Multiple types: use anyOf
        schema = {"anyOf": [_python_type_to_json_schema(a) for a in args]}
        if description:
            schema["description"] = description
        return schema

    # Handle Literal types
    if origin is Literal:
        literal_values = list(args)
        if not literal_values:
            raise TypeError("Literal type must have at least one value")

        # Determine type from first value
        first_val = literal_values[0]
        if isinstance(first_val, bool):
            schema = {"type": "boolean", "enum": literal_values}
        elif isinstance(first_val, int):
            schema = {"type": "integer", "enum": literal_values}
        elif isinstance(first_val, str):
            schema = {"type": "string", "enum": literal_values}
        else:
            schema = {"enum": literal_values}

        if description:
            schema["description"] = description
        return schema

    # Handle TypedDict
    if is_typeddict(type_hint):
        # type_hint is guaranteed to be a TypedDict type here
        return _typeddict_to_json_schema(type_hint, description)  # type: ignore[arg-type]

    # Handle plain classes that might be dict-like or have annotations
    if isinstance(type_hint, type):
        # If it's a basic class without special handling, treat as object
        if hasattr(type_hint, "__annotations__") and type_hint.__annotations__:
            # Class with annotations - try to convert
            return _class_to_json_schema(type_hint, description)

        # Fallback: treat as generic object
        schema = {"type": "object"}
        if description:
            schema["description"] = description
        return schema

    # Unsupported type
    raise TypeError(
        f"Unsupported type hint for tool schema: {type_hint}. "
        "Supported types: str, int, float, bool, None, list[T], dict[K,V], "
        "tuple[T, ...], set[T], Optional[T], Union[...], Literal[...], TypedDict"
    )


def _typeddict_to_json_schema(
    td_type: type,
    description: str | None = None,
) -> dict[str, Any]:
    """
    Convert a TypedDict to a JSON Schema object.

    :param td_type: The TypedDict type to convert.
    :type td_type: type
    :param description: Optional description to include in the schema.
    :type description: str | None
    :returns: JSON Schema object.
    :rtype: dict[str, Any]
    """
    try:
        hints = get_type_hints(td_type)
    except Exception:
        hints = getattr(td_type, "__annotations__", {})

    properties: dict[str, Any] = {}
    for name, hint in hints.items():
        properties[name] = _python_type_to_json_schema(hint)

    # Get required keys (TypedDict has __required_keys__ and __optional_keys__)
    required_keys = getattr(td_type, "__required_keys__", frozenset())
    optional_keys = getattr(td_type, "__optional_keys__", frozenset())

    # If neither is set, all keys are required by default
    if not required_keys and not optional_keys:
        required = sorted(hints.keys())
    else:
        required = sorted(required_keys)

    schema: dict[str, Any] = {
        "type": "object",
        "properties": properties,
    }
    if required:
        schema["required"] = required
    if description:
        schema["description"] = description

    return schema


def _class_to_json_schema(
    cls_type: type,
    description: str | None = None,
) -> dict[str, Any]:
    """
    Convert a class with annotations to a JSON Schema object.

    :param cls_type: The class type to convert.
    :type cls_type: type
    :param description: Optional description to include in the schema.
    :type description: str | None
    :returns: JSON Schema object.
    :rtype: dict[str, Any]
    """
    try:
        hints = get_type_hints(cls_type)
    except Exception:
        hints = getattr(cls_type, "__annotations__", {})

    properties: dict[str, Any] = {}
    for name, hint in hints.items():
        properties[name] = _python_type_to_json_schema(hint)

    schema: dict[str, Any] = {
        "type": "object",
        "properties": properties,
        "required": sorted(hints.keys()),
    }
    if description:
        schema["description"] = description

    return schema


def _generate_tool_schema(
    func: Callable[..., Any],
    parsed_docstring: _ParsedDocstring,
) -> dict[str, Any]:
    """
    Generate an OpenAI-compatible tool schema from a Python function.

    :param func: The function to generate a schema for.
    :type func: Callable[..., Any]
    :param parsed_docstring: Parsed docstring with summary and parameter descriptions.
    :type parsed_docstring: _ParsedDocstring
    :returns: OpenAI-compatible tool schema.
    :rtype: dict[str, Any]
    :raises TypeError: If a parameter has no type annotation.
    """
    # Get function name
    func_name = getattr(func, "__name__", None) or ""
    if not func_name:
        raise ValueError("Function must have a name")

    # Get function signature
    sig = inspect.signature(func)

    # Get type hints
    try:
        hints = get_type_hints(func)
    except Exception:
        hints = {}

    # Build properties and required list
    properties: dict[str, Any] = {}
    required: list[str] = []

    for param_name, param in sig.parameters.items():
        # Skip *args and **kwargs
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            continue

        # Get type hint
        if param_name in hints:
            type_hint = hints[param_name]
        elif param.annotation is not param.empty:
            type_hint = param.annotation
        else:
            raise TypeError(
                f"Tool parameter '{param_name}' has no type annotation. "
                "All tool parameters must have type hints."
            )

        # Warn if parameter uses set or tuple (received as list at runtime due to JSON)
        if _contains_collection_type(type_hint, (set, frozenset)):
            warnings.warn(
                f"Tool parameter '{param_name}' uses 'set' type hint. "
                "Since tool arguments are transmitted as JSON, this parameter will be "
                "received as a Python 'list' at runtime. Convert inside your function "
                "if you need set behavior: `{} = set({})`".format(
                    param_name, param_name
                ),
                UserWarning,
                stacklevel=4,
            )
        elif _contains_collection_type(type_hint, (tuple,)):
            warnings.warn(
                f"Tool parameter '{param_name}' uses 'tuple' type hint. "
                "Since tool arguments are transmitted as JSON, this parameter will be "
                "received as a Python 'list' at runtime. Convert inside your function "
                "if you need tuple behavior: `{} = tuple({})`".format(
                    param_name, param_name
                ),
                UserWarning,
                stacklevel=4,
            )

        # Check if this is Optional (has default or is Optional[X])
        is_optional = param.default is not param.empty
        origin = get_origin(type_hint)
        args = get_args(type_hint)

        # Check for Union with None (Optional)
        is_union_with_none = False
        if origin is Union or isinstance(type_hint, types.UnionType):
            is_union_with_none = type(None) in args

        if is_union_with_none:
            is_optional = True

        # Get parameter description from docstring
        param_description = parsed_docstring.params.get(param_name)

        # Convert type to JSON Schema
        param_schema = _python_type_to_json_schema(type_hint, param_description)
        properties[param_name] = param_schema

        # Add to required list if not optional
        if not is_optional:
            required.append(param_name)

    # Build the full tool schema
    tool_description = parsed_docstring.summary
    if parsed_docstring.description:
        tool_description = (
            f"{parsed_docstring.summary}\n\n{parsed_docstring.description}"
        )

    schema: dict[str, Any] = {
        "type": "function",
        "function": {
            "name": func_name,
            "parameters": {
                "type": "object",
                "properties": properties,
            },
        },
    }

    if tool_description:
        schema["function"]["description"] = tool_description

    if required:
        schema["function"]["parameters"]["required"] = required

    return schema
