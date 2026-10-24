# SPDX-FileCopyrightText: Shcherbak AI AS
# SPDX-License-Identifier: Apache-2.0

"""
Module defining utility functions for the base classes.
"""

from __future__ import annotations

import re
import sys
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
from contextgem.internal.typings.user_type_hints_validation import _is_string_type


def _get_body_classes(cls: type) -> dict[str, type]:
    """
    Returns the classes defined in the body of a class, e.g. its nested classes,
    by their names. Only the classes: the default value of a field, e.g. ``None``
    for ``date: date | None = None``, would hide in the annotations the class that
    its name is of.

    :param cls: The class.
    :type cls: type
    :return: The classes of its body, by their names.
    :rtype: dict[str, type]
    """
    return {name: value for name, value in vars(cls).items() if isinstance(value, type)}


@_disable_direct_initialization
class _JsonObjectClassStruct:
    """
    Internal implementation of the JsonObjectClassStruct class.
    """

    # The subclasses, by their module and their qualified name, e.g.
    # `test_person.<locals>.Address` for a class defined inside a function. They
    # are what the string annotations of a subclass that are not found in its module
    # name, e.g. the classes defined inside a function, in a module with postponed
    # annotations (`from __future__ import annotations`), where the annotations are
    # strings (see `_get_enclosing_classes`). A class is never looked up by its name
    # alone: two classes of the same name in other modules or scopes, e.g. an
    # `Address` in each of two modules, are different classes. A class defined again
    # with the same qualified name, e.g. in each call of a function, replaces
    # the class defined before
    _registry = {}  # noqa: RUF012 - annotated, it would be a field of the user's classes

    def __init_subclass__(cls, **kwargs):
        """
        Registers each subclass in the registry, by its module and its qualified name.
        """
        super().__init_subclass__(**kwargs)
        _JsonObjectClassStruct._registry[(cls.__module__, cls.__qualname__)] = cls

    @classmethod
    def _get_enclosing_classes(cls) -> dict[str, type]:
        """
        Returns the subclasses defined in the scopes that enclose the class, i.e.
        in the functions that it is defined in, by their names, as Python finds
        the names of these scopes from the class: a class of an inner scope
        hides a class of the same name of an outer one. The classes of the module
        itself are found in its namespace.

        :return: The subclasses of the enclosing scopes, by their names.
        :rtype: dict[str, type]
        """
        # The scopes of the functions that the class is defined in, from the outermost:
        # `f.<locals>` and `f.<locals>.g.<locals>` for `f.<locals>.g.<locals>.Address`
        parts = cls.__qualname__.split(".")[:-1]
        scopes = [
            ".".join(parts[: i + 1])
            for i, part in enumerate(parts)
            if part == "<locals>"
        ]
        classes: dict[str, type] = {}
        for scope in scopes:
            for (module, qualname), registered_class in list(
                _JsonObjectClassStruct._registry.items()
            ):
                scope_of_class, _, name = qualname.rpartition(".")
                if module == cls.__module__ and scope_of_class == scope:
                    classes[name] = registered_class
        return classes

    @classmethod
    def _as_dict_structure(cls) -> dict[str, Any]:
        """
        Converts class hierarchy to a dictionary structure for JsonObjectConcept.

        :return: Dictionary representation of the class structure suitable for JsonObjectConcept.
        :rtype: dict[str, Any]
        """
        result = {}

        # The namespace for the names of the annotations, as Python looks them up:
        # the classes defined in the body of the class, e.g. a nested class, then
        # the subclasses of the scopes that enclose the class, then its module
        module = sys.modules.get(cls.__module__)
        namespace: dict[str, Any] = dict(vars(module)) if module else {}
        namespace.update(cls._get_enclosing_classes())
        namespace.update(_get_body_classes(cls))

        # Get annotations using get_type_hints to resolve forward references
        # This correctly handles Pydantic and other typed classes. The names of
        # the enclosing scopes are given to it only for a class defined in a function.
        # Names given to it replace the namespace of each class of the hierarchy,
        # i.e. they include the classes of the bodies of the base classes, e.g. a class
        # that an inherited field names, which the names of the class itself hide
        localns: dict[str, Any] | None = None
        if "<locals>" in cls.__qualname__.split("."):
            localns = {}
            for base in reversed(cls.__mro__):
                if issubclass(base, _JsonObjectClassStruct):
                    localns.update(base._get_enclosing_classes())
                localns.update(_get_body_classes(base))
        try:
            annotations = get_type_hints(cls, localns=localns)
        except Exception:  # noqa: BLE001 - falls back to the raw annotations
            # Fallback to raw annotations if get_type_hints fails
            annotations = cls.__annotations__ if hasattr(cls, "__annotations__") else {}

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

        # Case 2: A subclass, converted by itself. Another class, e.g. a dataclass
        # of the same name, is converted as any other typed class
        if isinstance(field_type, type) and issubclass(
            field_type, _JsonObjectClassStruct
        ):
            return field_type._as_dict_structure()

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
            inner_class = namespace.get(inner_type_name)
            if hasattr(inner_class, "_as_dict_structure"):
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
            f"Make sure the type is either a built-in type, a class of the module, "
            f"or a class that inherits from JsonObjectClassStruct, defined in "
            f"the same function as the class of the field or in a function "
            f"that encloses it."
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

        # A subclass as the item type, converted by itself
        if isinstance(item_type, type) and issubclass(
            item_type, _JsonObjectClassStruct
        ):
            return [item_type._as_dict_structure()]

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
                f"Make sure the type is either a built-in type, a class of the module, "
                f"or a class that inherits from JsonObjectClassStruct, defined in "
                f"the same function as the class of the field or in a function "
                f"that encloses it."
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

        # Check that the keys are strings, which the keys of JSON objects are:
        # `str`, or a Literal of strings, as in a structure given as a dictionary
        if not _is_string_type(key_type):
            raise TypeError(
                f"Invalid key type for dictionary field '{field_name}': {key_type}.\n"
                f"Dictionary keys must be strings: `str`, or a Literal of strings."
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
