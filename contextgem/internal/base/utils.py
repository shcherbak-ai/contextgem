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
Module defining utility functions for the base classes.
"""

from __future__ import annotations

import re
from typing import Any, get_type_hints

from contextgem.internal.decorators import _disable_direct_initialization
from contextgem.internal.typings.strings_to_types import (
    PRIMITIVE_TYPES_STRING_MAP_REVERSED,
    _deserialize_type_hint,
)
from contextgem.internal.typings.typed_class_utils import _raise_dict_class_type_error
from contextgem.internal.typings.types_normalization import _normalize_type_annotation
from contextgem.internal.typings.types_to_strings import (
    JSON_PRIMITIVE_TYPES,
    _is_json_serializable_type,
    _raise_json_serializable_type_error,
)


@_disable_direct_initialization
class _JsonObjectClassStruct:
    """
    Internal implementation of the JsonObjectClassStruct class.
    """

    # Registry to store all subclasses for type resolution during structure generation:
    # 1. Resolves string type annotations and forward references
    # 2. Enables lookup of already defined classes by name when traversing class hierarchies
    # 3. Provides a shared namespace for the entire type structure
    # 4. Critical for resolving classes used in generic containers like list[Class]
    _registry = {}

    def __init_subclass__(cls, **kwargs):
        """
        Registers each subclass in the registry
        """
        super().__init_subclass__(**kwargs)
        _JsonObjectClassStruct._registry[cls.__name__] = cls

    @classmethod
    def _as_dict_structure(cls) -> dict[str, Any]:
        """
        Converts class hierarchy to a dictionary structure for JsonObjectConcept.

        :return: Dictionary representation of the class structure suitable for JsonObjectConcept.
        :rtype: dict[str, Any]
        """
        result = {}

        # Get annotations using get_type_hints to resolve forward references
        # This correctly handles Pydantic and other typed classes
        try:
            annotations = get_type_hints(cls)
        except Exception:
            # Fallback to raw annotations if get_type_hints fails
            annotations = cls.__annotations__ if hasattr(cls, "__annotations__") else {}

        # Create a combined namespace for resolving types
        namespace = {}
        namespace.update(globals())  # Start with global namespace
        namespace.update(_JsonObjectClassStruct._registry)  # Add registered classes

        for field_name, field_type in annotations.items():
            # Normalize field type to handle generic types consistently
            field_type = _normalize_type_annotation(field_type)

            # Process based on field_type
            processed_field_type = cls._process_field_type(
                field_type, field_name, namespace
            )

            # Validate that the processed field type is JSON serializable
            if not _is_json_serializable_type(processed_field_type):
                _raise_json_serializable_type_error(
                    processed_field_type, field_name=field_name
                )

            result[field_name] = processed_field_type

        return result

    @classmethod
    def _process_field_type(
        cls, field_type: Any, field_name: str, namespace: dict[str, Any]
    ) -> Any:
        """
        Processes a field type and converts it to the appropriate dictionary structure.

        :param field_type: The type annotation to process
        :type field_type: Any
        :param field_name: The name of the field
        :type field_name: str
        :param namespace: The namespace for resolving types
        :type namespace: dict[str, Any]
        :return: Appropriate representation of the type for dictionary structure
        :rtype: Any
        """
        # Case 1: String type annotations (forward references or type strings)
        if isinstance(field_type, str):
            return cls._process_string_type(field_type, field_name, namespace)

        # Case 2: Types in registry (_JsonObjectClassStruct subclasses)
        if (
            hasattr(field_type, "__name__")
            and field_type.__name__ in _JsonObjectClassStruct._registry
        ):
            return _JsonObjectClassStruct._registry[
                field_type.__name__
            ]._as_dict_structure()

        # Case 3: List types
        if hasattr(field_type, "__origin__") and field_type.__origin__ is list:
            return cls._process_list_type(field_type, field_name, namespace)

        # Case 4: Dictionary types
        if hasattr(field_type, "__origin__") and field_type.__origin__ is dict:
            return cls._process_dict_type(field_type, field_name, namespace)

        # Default case: Use the field type directly
        return field_type

    @classmethod
    def _process_string_type(
        cls, field_type: str, field_name: str, namespace: dict[str, Any]
    ) -> Any:
        """
        Processes a string type annotation

        :param field_type: The type annotation to process
        :type field_type: str
        :param field_name: The name of the field
        :type field_name: str
        :param namespace: The namespace for resolving types
        :type namespace: dict[str, Any]
        :return: Appropriate representation of the type for dictionary structure
        :rtype: Any
        """

        # First try using the general type parser for all type annotations
        try:
            return _deserialize_type_hint(field_type)
        except ValueError:
            # If that fails, continue with the rest of the logic
            pass

        # Handle list type syntax in strings with regex
        list_match = re.compile(r"list\[(.*)\]").match(field_type)
        if list_match:
            inner_type_name = list_match.group(1).strip()
            if inner_type_name in _JsonObjectClassStruct._registry:
                inner_class = _JsonObjectClassStruct._registry[inner_type_name]
                return [inner_class._as_dict_structure()]

        # Try to resolve the type from our namespace
        if field_type in namespace:
            resolved_type = namespace[field_type]
            if hasattr(resolved_type, "_as_dict_structure"):
                return resolved_type._as_dict_structure()
            return resolved_type

        # Resolve primitive type by name
        if field_type in PRIMITIVE_TYPES_STRING_MAP_REVERSED:
            return PRIMITIVE_TYPES_STRING_MAP_REVERSED[field_type]

        # If we can't resolve the type, raise an error
        raise ValueError(
            f"Could not resolve type '{field_type}' for field '{field_name}'. "
            f"Make sure the type is either a built-in type or registered "
            f"with JsonObjectClassStruct."
        )

    @classmethod
    def _process_list_type(
        cls, field_type: Any, field_name: str, namespace: dict[str, Any]
    ) -> Any:
        """
        Processes a list type annotation

        :param field_type: The type annotation to process
        :type field_type: Any
        :param field_name: The name of the field
        :type field_name: str
        :param namespace: The namespace for resolving types
        :type namespace: dict[str, Any]
        :return: Appropriate representation of the type for dictionary structure
        :rtype: Any
        """
        # Validate list type has exactly one type argument
        if len(field_type.__args__) != 1:
            raise ValueError(
                f"List type annotation for '{field_name}' must have "
                f"exactly one type argument, got {len(field_type.__args__)}"
            )

        # Get the item type
        item_type = field_type.__args__[0]

        # Check if item_type is registered by name in our registry
        if (
            hasattr(item_type, "__name__")
            and item_type.__name__ in _JsonObjectClassStruct._registry
        ):
            resolved_class = _JsonObjectClassStruct._registry[item_type.__name__]
            return [resolved_class._as_dict_structure()]

        # Handle string item types
        if isinstance(item_type, str):
            # Try to resolve from namespace
            if item_type in namespace:
                resolved_item_type = namespace[item_type]
                if hasattr(resolved_item_type, "_as_dict_structure"):
                    return [resolved_item_type._as_dict_structure()]
                return [resolved_item_type]
            # Try to resolve primitive type
            if item_type in PRIMITIVE_TYPES_STRING_MAP_REVERSED:
                return [PRIMITIVE_TYPES_STRING_MAP_REVERSED[item_type]]
            # Can't resolve - raise error
            raise ValueError(
                f"Could not resolve list item type '{item_type}' for field '{field_name}'. "
                f"Make sure the type is either a built-in type or registered "
                f"with JsonObjectClassStruct."
            )

        # Default case: keep the list type as is
        return field_type

    @classmethod
    def _process_dict_type(
        cls, field_type: Any, field_name: str, namespace: dict[str, Any]
    ) -> Any:
        """
        Processes a dictionary type annotation

        :param field_type: The type annotation to process
        :type field_type: Any
        :param field_name: The name of the field
        :type field_name: str
        :param namespace: The namespace for resolving types
        :type namespace: dict[str, Any]
        :return: Appropriate representation of the type for dictionary structure
        :rtype: Any
        """

        # Validate dictionary has key and value type arguments
        if len(field_type.__args__) != 2:
            return field_type

        key_type, value_type = field_type.__args__

        # Check if key type is str (required for dictionaries)
        if key_type is not str:
            raise TypeError(
                f"Invalid key type for dictionary field '{field_name}': {key_type}.\n"
                f"Dictionary keys must be strings."
            )

        # Check if key or value type is a class that should be disallowed
        for type_arg, type_name in [(key_type, "key"), (value_type, "value")]:
            if (
                hasattr(type_arg, "__module__")
                and hasattr(type_arg, "__name__")
                and not isinstance(type_arg, type)
            ):
                is_basic_type = type_arg in JSON_PRIMITIVE_TYPES
                if not is_basic_type and not (
                    hasattr(type_arg, "__origin__") or isinstance(type_arg, str)
                ):
                    _raise_dict_class_type_error(type_name, field_name, type_arg)
            elif isinstance(type_arg, str):
                if type_arg in namespace:
                    resolved_type = namespace[type_arg]
                    if hasattr(resolved_type, "_as_dict_structure"):
                        _raise_dict_class_type_error(type_name, field_name, type_arg)

        # Keep dictionary type as is - structure is defined by key/value types
        return field_type


def _is_registered_tool(obj: Any) -> bool:
    """
    Returns True if the object is a callable marked as a ContextGem tool
    and has a non-empty tool name.

    :param obj: The object to check.
    :type obj: Any
    :return: True if callable, flagged with ``__contextgem_tool__`` and has a
        non-empty ``__contextgem_tool_name__``; else False.
    :rtype: bool
    """
    if not callable(obj):
        return False
    if not bool(getattr(obj, "__contextgem_tool__", False)):
        return False
    tool_name = getattr(obj, "__contextgem_tool_name__", None)
    return isinstance(tool_name, str) and bool(tool_name.strip())
