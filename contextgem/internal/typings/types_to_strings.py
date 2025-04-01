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
Module for serializing type annotations into string representations.

This module provides functionality to convert Python type annotations into string
representations that are compatible with prompts or suitable for serialization.
It handles various type constructs including basic types, generics, unions, and
custom types, ensuring consistent string formatting across the framework.
"""

from types import GenericAlias, UnionType
from typing import Union, get_args, get_origin, get_type_hints

# Define allowed JSON-serializable types
JSON_SERIALIZABLE_TYPES = (int, float, str, bool, type(None), list, dict)

# Base mapping for simple types
JSON_OBJECT_STRUCTURE_TYPES_MAP = {
    str: "str",
    int: "int",
    float: "float",
    bool: "bool",
    type(None): "null",
}


def _is_json_serializable_type(typ: type | GenericAlias | UnionType) -> bool:
    """
    Determines if a given type is JSON serializable by verifying its compatibility
    with commonly supported JSON serializable types. This function supports both
    modern Python union types (|) and older typing.Union syntax, as well as generic
    types such as lists and dictionaries. It ensures that any nested type arguments
    are recursively checked for JSON serializability.

    :param typ: The type to be checked for JSON serializability. It may include
        standard types, generic types (e.g., dict, list), and union types.
    :returns: A boolean value indicating whether the given type is JSON serializable
        (i.e., compatible with supported JSON types).
    :rtype: bool
    """

    # Handle union types from both the new (|) and legacy (typing.Union) syntax
    if isinstance(typ, UnionType) or getattr(typ, "__origin__", None) is Union:
        return all(_is_json_serializable_type(arg) for arg in get_args(typ))

    # Handle generic types like list[str] or dict[str, int]
    if hasattr(typ, "__origin__"):
        if typ.__origin__ in (list, dict):
            return all(_is_json_serializable_type(arg) for arg in get_args(typ))

    return typ in JSON_SERIALIZABLE_TYPES


def _format_type(typ: type | GenericAlias | UnionType) -> str:
    """
    Formats a type hint into a prompt-compatible string representation.

    This function supports different type hints such as unions (e.g., `str | int` or
    `typing.Union`), generic collections like `list` and `dict`, and special cases like
    `None`. It recursively generates the string representation for nested and
    parameterized types using their arguments.

    :param typ: A type hint to be formatted.
    :return: A prompt-compatible string representation of the given type hint.
    :rtype: str
    """

    # Handle unions (both | syntax and typing.Union)
    if isinstance(typ, UnionType) or getattr(typ, "__origin__", None) is Union:
        return " or ".join(_format_type(arg) for arg in get_args(typ))

    # Handle generic types (like list and dict)
    if hasattr(typ, "__origin__"):
        if typ.__origin__ is list:
            return f"[{_format_type(typ.__args__[0])}, ...]"  # e.g. [str, ...]
        elif typ.__origin__ is dict:
            key_type, value_type = typ.__args__
            return f"{{{_format_type(key_type)}: {_format_type(value_type)}, ...}}"  # e.g. {str: [int, ...]}

    if typ is type(None):
        return "null"

    return getattr(typ, "__name__", str(typ))  # Other types


class _JsonObjectItemStructure:
    """
    API for defining structured data models with type annotations
    and converting them into a prompt-compatible string.
    """

    def __init__(self, schema: type | dict[str, type | GenericAlias | UnionType]):
        """
        Accepts either a class or a dictionary representing field names and types.
        """
        if isinstance(schema, dict):
            schema_struct = schema  # User-provided dictionary of field types
        elif hasattr(schema, "__annotations__"):  # Class with type annotations
            schema_struct = get_type_hints(schema)
        else:
            raise TypeError(
                "Schema must be a dictionary or a class with type annotations."
            )

        for field, typ in schema_struct.items():
            if field.startswith("_"):
                raise ValueError(f"Field {field} cannot start with '_'")
            if not _is_json_serializable_type(typ):
                raise TypeError(
                    f"Invalid type for field '{field}': {typ}. "
                    f"Must be JSON-serializable type or a Union thereof."
                )
        self.schema = schema_struct

    def _to_prompt_string(self) -> str:
        """
        Converts the schema into a properly formatted prompt-compatible string.
        """
        formatted_dict = {
            key: _format_type(value) for key, value in self.schema.items()
        }
        return (
            "{"
            + ", ".join(f'"{key}": {value}' for key, value in formatted_dict.items())
            + "}"
        )


def _serialize_type_hint(tp: type | GenericAlias | UnionType) -> str:
    """
    Serializes type hints into a string representation.

    This function takes a type annotation or a type hint and returns its
    corresponding string representation, considering specific Python type
    constructs like list, dict, and unions. It validates the structure of composite
    type annotations, such as ensuring that lists have one type argument and
    dictionaries have two type arguments.

    :param tp: A type, generic alias, or union type from which the string
        representation would be serialized.
    :type tp: type | GenericAlias | UnionType

    :return: A string representation of the serialized type hint.
    :rtype: str

    :raises ValueError: If serialization fails.
    """
    # If it’s one of the base types, return its string.
    if tp in JSON_OBJECT_STRUCTURE_TYPES_MAP:
        return JSON_OBJECT_STRUCTURE_TYPES_MAP[tp]

    origin = get_origin(tp)
    args = get_args(tp)

    # Handle lists (e.g. list[int] or list[Union[str, float]])
    if origin is list:
        if len(args) != 1:
            raise ValueError("List must have one type argument")
        return f"list[{_serialize_type_hint(args[0])}]"

    # Handle dictionaries (e.g. dict[str, int])
    elif origin is dict:
        if len(args) != 2:
            raise ValueError("Dict must have two type arguments")
        return f"dict[{_serialize_type_hint(args[0])}, {_serialize_type_hint(args[1])}]"

    # Handle unions, including new union syntax
    elif origin in (Union, UnionType):
        # Sorting the serialized parts produces a canonical representation.
        serialized_parts = sorted(_serialize_type_hint(arg) for arg in args)
        return f"union[{', '.join(serialized_parts)}]"

    else:
        raise ValueError(f"Unsupported type hint: {tp}")
