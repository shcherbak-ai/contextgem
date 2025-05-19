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
from typing import Literal, Union, get_args, get_origin

from contextgem.internal.typings.typed_class_utils import (
    _get_model_fields,
    _is_typed_class,
)

# Define basic JSON primitive types
JSON_PRIMITIVE_TYPES = (int, float, str, bool, type(None))

# Mapping of basic primitive types to their string representations
PRIMITIVE_TYPES_STRING_MAP = {
    str: "str",
    int: "int",
    float: "float",
    bool: "bool",
    type(None): "null",
}


def _is_json_serializable_type(
    typ: type | GenericAlias | UnionType | dict | list,
) -> bool:
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

    # Handle Pydantic models and other classes with type annotations
    if _is_typed_class(typ):
        # Process it as a dictionary of fields
        fields = _get_model_fields(typ)
        return all(
            _is_json_serializable_type(field_type) for field_type in fields.values()
        )

    # Handle lists of dictionaries as nested structure definitions
    if isinstance(typ, list) and len(typ) == 1 and isinstance(typ[0], dict):
        return _is_json_serializable_type(typ[0])

    # Handle dictionaries as nested structure definitions
    if isinstance(typ, dict):
        return all(
            _is_json_serializable_type(field_type) for field_type in typ.values()
        )

    # Handle union types from both the new (|) and legacy (typing.Union) syntax
    if isinstance(typ, UnionType) or getattr(typ, "__origin__", None) is Union:
        return all(_is_json_serializable_type(arg) for arg in get_args(typ))

    # Handle Literal types - check that all literal values are JSON serializable
    if getattr(typ, "__origin__", None) is Literal:
        for arg in get_args(typ):
            # All literal values should be of JSON serializable primitive types
            if not isinstance(arg, JSON_PRIMITIVE_TYPES):
                return False
        return True

    # Handle generic types like list[str] or dict[str, int]
    if hasattr(typ, "__origin__"):
        if typ.__origin__ is list:
            # Check if the list item type is a custom class with type annotations
            item_type = typ.__args__[0]
            if _is_typed_class(item_type):
                # Validate the class itself recursively
                return _is_json_serializable_type(item_type)
            return _is_json_serializable_type(item_type)
        elif typ.__origin__ is dict:
            return all(_is_json_serializable_type(arg) for arg in get_args(typ))

    return typ in JSON_PRIMITIVE_TYPES


def _format_dict_structure(dict_structure: dict, indent_level: int = 0) -> str:
    """
    Formats a dictionary structure into a prompt-compatible string representation.

    :param dict_structure: Dictionary structure to format
    :param indent_level: Current indentation level for formatting
    :return: Formatted string representation
    """
    if not dict_structure:
        raise ValueError("Empty dictionary structure is not allowed")

    indent = "  " * indent_level
    next_indent = "  " * (indent_level + 1)

    parts = []
    for key, value in dict_structure.items():
        if isinstance(value, dict):
            # Recursively process nested dictionaries
            formatted_value = _format_dict_structure(value, indent_level + 1)
        else:
            # Process regular type annotations
            formatted_value = _format_type(value, indent_level + 1)

        parts.append(f'{next_indent}"{key}": {formatted_value}')

    # Always use multi-line formatting for consistency
    return "{\n" + ",\n".join(parts) + "\n" + indent + "}"


def _format_type(
    typ: type | GenericAlias | UnionType | dict | list, indent_level: int = 0
) -> str:
    """
    Formats a type hint into a prompt-compatible string representation.

    This function supports different type hints such as unions (e.g., `str | int` or
    `typing.Union`), generic collections like `list` and `dict`, special cases like
    `None` and literal types. It recursively generates the string representation for
    nested and parameterized types using their arguments.

    :param typ: A type hint to be formatted.
    :param indent_level: Current indentation level for formatting
    :return: A prompt-compatible string representation of the given type hint.
    :rtype: str
    """

    # Handle Pydantic models and other classes with type annotations
    if _is_typed_class(typ):
        fields = _get_model_fields(typ)
        return _format_dict_structure(fields, indent_level)

    # Handle dictionaries as nested structure definitions
    if isinstance(typ, dict):
        return _format_dict_structure(typ, indent_level)

    # Handle lists of dictionaries
    if isinstance(typ, list) and len(typ) == 1 and isinstance(typ[0], dict):
        dict_str = _format_dict_structure(typ[0], indent_level + 1)
        next_indent = "  " * (indent_level + 1)
        final_indent = "  " * indent_level
        return f"[\n{next_indent}{dict_str},\n{next_indent}...\n{final_indent}]"

    # Handle unions (both | syntax and typing.Union)
    if isinstance(typ, UnionType) or getattr(typ, "__origin__", None) is Union:
        union_types = [_format_type(arg, indent_level) for arg in get_args(typ)]
        return " or ".join(union_types)

    # Handle Literal types
    if getattr(typ, "__origin__", None) is Literal:
        # Create a list of serialized literal values
        literal_values = []
        for arg in get_args(typ):
            if isinstance(arg, str):
                # Escape quotes in strings and wrap in quotes
                literal_values.append(f'"{arg.replace("\"", "\\\"")}"')
            else:
                # For non-string literals, just convert to string
                literal_values.append(str(arg))

        # Join all literal values with "or" instead of commas
        values_str = " or ".join(literal_values)
        return values_str

    # Handle generic types (like list and dict)
    if hasattr(typ, "__origin__"):
        if typ.__origin__ is list:
            # Get the list item type
            item_type = typ.__args__[0]
            next_indent = "  " * (indent_level + 1)
            final_indent = "  " * indent_level

            # If the item type is a class with type annotations, handle it specially
            if _is_typed_class(item_type):
                item_fields = _format_type(item_type, indent_level + 1)
                return (
                    f"[\n{next_indent}{item_fields},\n{next_indent}...\n{final_indent}]"
                )
            # Otherwise, format it normally
            return f"[\n{next_indent}{_format_type(item_type, indent_level + 1)},\n{next_indent}...\n{final_indent}]"
        elif typ.__origin__ is dict:
            key_type, value_type = typ.__args__
            next_indent = "  " * (indent_level + 1)
            final_indent = "  " * indent_level
            key_str = _format_type(key_type, indent_level + 1)
            value_str = _format_type(value_type, indent_level + 1)
            return f"{{\n{next_indent}{key_str}: {value_str},\n{next_indent}...\n{final_indent}}}"

    if typ is type(None):
        return "null"

    return getattr(typ, "__name__", str(typ))  # Other types


class _JsonObjectItemStructure:
    """
    API for defining structured data models with type annotations
    and converting them into a prompt-compatible string.
    """

    def __init__(
        self, schema: type | dict[str, type | GenericAlias | UnionType | dict | list]
    ):
        """
        Accepts either a class or a dictionary representing field names and types.
        """
        if isinstance(schema, dict):
            schema_struct = schema  # User-provided dictionary of field types
        elif _is_typed_class(schema):  # Class with type annotations or Pydantic model
            schema_struct = _get_model_fields(schema)
        else:
            raise TypeError(
                "Schema must be a dictionary, a class with type annotations, or a Pydantic model."
            )

        self._validate_schema(schema_struct)
        self.schema = schema_struct

    def _validate_schema(self, schema, prefix=""):
        """
        Validates that all fields in the schema (including nested ones) have valid types.
        """
        for field, typ in schema.items():
            # Validate that the field name is a string
            if not isinstance(field, str):
                raise TypeError(
                    f"Invalid field name in schema{' ' + prefix if prefix else ''}: {field}. "
                    f"Dictionary keys must be strings, got {type(field).__name__}"
                )

            full_field_name = f"{prefix}.{field}" if prefix else field
            if field.startswith("_"):
                raise ValueError(f"Field {full_field_name} cannot start with '_'")

            # Handle Pydantic models and classes with type annotations
            if _is_typed_class(typ):
                try:
                    nested_fields = _get_model_fields(typ)
                    self._validate_schema(nested_fields, full_field_name)
                except Exception as e:
                    raise TypeError(
                        f"Invalid type for field '{full_field_name}': {typ}. "
                        f"Failed to extract type annotations: {e}"
                    ) from e
            # Handle nested dictionaries as structure definitions
            elif isinstance(typ, dict):
                self._validate_schema(typ, full_field_name)
            # Handle lists of dictionaries
            elif isinstance(typ, list) and len(typ) == 1 and isinstance(typ[0], dict):
                self._validate_schema(typ[0], f"{full_field_name}[]")
            elif not _is_json_serializable_type(typ):
                supported_types = (
                    "- Basic types: str, int, float, bool, None\n"
                    "- Union types: str | int, Optional[str], etc.\n"
                    "- Collection types: list[T], dict[K, V] where T, K, V are supported types\n"
                    "- Literal types: Literal['value1', 'value2'], Literal[1, 2, 3]\n"
                    "- Classes with type annotations (including dataclasses and Pydantic models)\n"
                    "- Nested dictionaries and lists of dictionaries as structure definitions"
                )
                raise TypeError(
                    f"Invalid type for field '{full_field_name}': {typ}.\n"
                    f"Must be one of the following supported types:\n{supported_types}"
                )

    def _to_prompt_string(self) -> str:
        """
        Converts the schema into a properly formatted prompt-compatible string.
        Recursively handles nested dictionaries and lists of dictionaries.
        """
        return _format_type(self.schema)


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
    # If it's one of the base types, return its string.
    if tp in PRIMITIVE_TYPES_STRING_MAP:
        return PRIMITIVE_TYPES_STRING_MAP[tp]

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

    # Handle Literal types
    elif origin is Literal:
        # Create a list of serialized literal values
        literal_values = []
        for arg in args:
            if isinstance(arg, str):
                # Escape quotes in strings and wrap in quotes
                literal_values.append(f'"{arg.replace("\"", "\\\"")}"')
            else:
                # For non-string literals, just convert to string
                literal_values.append(str(arg))

        # Join all literal values with commas
        values_str = ", ".join(literal_values)
        return f"literal[{values_str}]"

    else:
        raise ValueError(f"Unsupported type hint: {tp}")
