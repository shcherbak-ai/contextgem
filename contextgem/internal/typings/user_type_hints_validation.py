# SPDX-FileCopyrightText: Shcherbak AI AS
# SPDX-License-Identifier: Apache-2.0

"""
Module for validating user-provided type hints.

This module provides functionality to validate and process type hints supplied by users
in various contexts throughout the ContextGem framework. It includes utilities for extracting
type information from different input formats, validating type compatibility with JSON
serialization requirements, and ensuring type consistency across the application.
"""

from __future__ import annotations

from types import UnionType
from typing import Annotated, Any, Literal, Union, get_args, get_origin

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, create_model

from contextgem.internal.typings.types_to_strings import (
    _is_json_serializable_type,
    _raise_json_serializable_type_error,
)


def _extract_mapper(
    user_mapper: dict[str, Any],
) -> dict[str, Any]:
    """
    Validates and returns a dictionary mapping field names to their corresponding type hints.

    The user_mapper should be a dictionary where keys are field names and values represent
    type hints, nested dictionaries, or lists of dictionaries for array items.

    :param user_mapper: A dictionary mapping string field names to type hints,
        nested dictionaries, or lists.
    :type user_mapper: dict[str, Any]

    :return: The validated dictionary mapping field names to type hints, nested dictionaries,
        or lists.
    :rtype: dict[str, Any]

    :raises ValueError: If the input user_mapper is not a dictionary.
    """
    if isinstance(user_mapper, dict):
        return user_mapper
    else:
        raise ValueError("`user_mapper` must be a dict.")  # noqa: TRY004 - pydantic and the callers handle ValueError


def _is_optional(type_hint: Any) -> bool:
    """
    Determines whether a given type hint is of an optional type.

    An optional type means that the type hint allows `None` as a valid value. This
    is typically represented by a `Union` that contains `NoneType`.

    :param type_hint: The type hint to check.
    :type type_hint: Any
    :return: A boolean value indicating whether the provided type hint is optional.
    :rtype: bool
    """
    origin = get_origin(type_hint)
    if origin is Union or isinstance(type_hint, UnionType):
        return type(None) in get_args(type_hint)
    return False


def _is_string_type(type_hint: Any) -> bool:
    """
    Checks whether a type hint allows strings only: ``str``, a Literal of strings,
    or a union of these.

    :param type_hint: The type hint to check.
    :type type_hint: Any
    :return: True if the type hint allows strings only, False otherwise.
    :rtype: bool
    """
    if type_hint is str:
        return True
    origin = get_origin(type_hint)
    if origin is Literal:
        return all(isinstance(value, str) for value in get_args(type_hint))
    if origin is Union or isinstance(type_hint, UnionType):
        return all(_is_string_type(arg) for arg in get_args(type_hint))
    return False


def _check_mapping_keys(type_hint: Any, field_name: str) -> None:
    """
    Checks that each mapping in a type hint, at any depth, has string keys: the keys
    of a JSON object are strings, so that a mapping with keys of another type, e.g.
    ``dict[int, str]``, could not be given a value from JSON, other than an empty one.

    :param type_hint: The type hint of a field.
    :type type_hint: Any
    :param field_name: The name of the field, for the error message.
    :type field_name: str
    :return: None
    :rtype: None
    :raises ValueError: If a mapping in the type hint has keys of another type than
        strings.
    """
    if get_origin(type_hint) is dict:
        key_type, value_type = get_args(type_hint)
        if not _is_string_type(key_type):
            raise ValueError(
                f"Field '{field_name}' has a mapping with keys of type {key_type!r}: "
                f"the keys of a JSON object are strings, i.e. a mapping must have "
                f"`str`, or a Literal of strings, as its key type, e.g. "
                f"`dict[str, int]`."
            )
        _check_mapping_keys(value_type, field_name)
        return
    if get_origin(type_hint) is Literal:
        return
    for arg in get_args(type_hint):
        _check_mapping_keys(arg, field_name)


def _match_literals_exactly(type_hint: Any) -> Any:
    """
    Returns a type hint that validates a value as the given one does, except that
    each Literal in it, at any depth, accepts a value only if it is one of
    the values of the Literal and of the same type. Pydantic accepts a value that
    is equal to a value of a Literal, also in strict mode, so that e.g.
    ``Literal[1]`` accepts ``True`` and ``1.0``, and ``Literal[True]`` accepts ``1``.

    :param type_hint: The type hint.
    :type type_hint: Any
    :return: The type hint with the Literals in it matched exactly.
    :rtype: Any
    """
    origin = get_origin(type_hint)
    if origin is Literal:
        allowed_values = get_args(type_hint)

        def check_value(value: Any) -> Any:
            """
            Accepts a value of the Literal only with the type of that value.
            """
            if not any(
                type(value) is type(allowed) and value == allowed
                for allowed in allowed_values
            ):
                raise ValueError(f"Input should be one of {list(allowed_values)!r}")
            return value

        return Annotated[type_hint, BeforeValidator(check_value)]
    if origin is Union or isinstance(type_hint, UnionType):
        return Union[tuple(_match_literals_exactly(arg) for arg in get_args(type_hint))]  # noqa: UP007
    if origin is list:
        (item_type,) = get_args(type_hint)
        exact_item_type = _match_literals_exactly(item_type)
        return list[exact_item_type]
    if origin is dict:
        key_type, value_type = get_args(type_hint)
        exact_key_type = _match_literals_exactly(key_type)
        exact_value_type = _match_literals_exactly(value_type)
        return dict[exact_key_type, exact_value_type]
    return type_hint


def _dynamic_pydantic_model(
    user_mapper: dict[str, Any],
) -> type[BaseModel]:
    """
    Dynamically generates a pydantic model class based on a provided mapping of field names
    to their corresponding type annotations. This function is valuable for scenarios where
    data structures are defined dynamically or at runtime, allowing the user to construct
    strictly-typed schema without manually creating pydantic models for every variation.

    :param user_mapper: A dictionary where keys are the field names (str) and the values are
        valid type hints (e.g., int, float, str | None, etc.). These hints dictate the types
        of the fields included in the generated model. Can also contain nested dictionaries
        to represent nested object structures or lists of dictionaries for array items.
    :type user_mapper: dict[str, Any]

    :return: A dynamically created subclass of pydantic's BaseModel that adheres to the
        type constraints and field definitions supplied in the user_mapper.
    :rtype: type[BaseModel]
    """

    # Build the fields dict expected by create_model. Each field gets a tuple:
    # (type, field info with the default). A field of the model has a name of its
    # own, and the name of the field of the structure as its alias, which values
    # are validated by: a name of the structure may be any string, also one that
    # pydantic gives a meaning to, e.g. "model_config", or that shadows
    # an attribute of BaseModel, e.g. "json", which pydantic warns of each time
    # the model is created.
    fields: dict[str, Any] = {}
    for position, (field_name, field_type) in enumerate(
        _extract_mapper(user_mapper).items()
    ):
        if field_name.startswith("_"):
            raise ValueError(f"Field {field_name} cannot start with '_'")
        model_field_name = f"field_{position}"

        # Handle nested dictionaries as nested models
        if isinstance(field_type, dict):
            # Recursively create a nested model for this field
            nested_model = _dynamic_pydantic_model(field_type)
            fields[model_field_name] = (nested_model, Field(..., alias=field_name))
            continue

        # Handle lists of dictionaries
        if (
            isinstance(field_type, list)
            and len(field_type) == 1
            and isinstance(field_type[0], dict)
        ):
            # Create a nested model for the list item
            nested_model = _dynamic_pydantic_model(field_type[0])
            # Create a list of that model type
            fields[model_field_name] = (
                list[nested_model],  # ty: ignore[invalid-type-form]
                Field(..., alias=field_name),
            )
            continue

        # Handle list instances with non-dictionary content
        # (e.g., [str], [int | float], [Literal["a", "b"]], [SomeClass])
        if (
            isinstance(field_type, list)
            and len(field_type) == 1
            and not isinstance(field_type[0], dict)
        ):
            # Convert list instance to generic list type
            item_type = field_type[0]
            _check_mapping_keys(item_type, field_name)
            exact_item_type = _match_literals_exactly(item_type)
            fields[model_field_name] = (
                list[exact_item_type],
                Field(..., alias=field_name),
            )
            continue

        # Check that the mapper value is a valid type hint.
        if not (isinstance(field_type, type) or get_origin(field_type) is not None):
            raise ValueError(
                f"Field '{field_name}' has an invalid type hint: {field_type!r}. "
                "It must be a type, a union of types, or a generic alias."
            )
        # Check for JSON-serializability of the type hint
        if not _is_json_serializable_type(field_type):
            _raise_json_serializable_type_error(
                field_type,
                field_name=field_name,
                exception_type=ValueError,
            )
        _check_mapping_keys(field_type, field_name)
        # If the type includes None, assume it's optional and give it a default of None.
        default = None if _is_optional(field_type) else ...
        fields[model_field_name] = (
            _match_literals_exactly(field_type),
            Field(default, alias=field_name),
        )

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
    )
    try:
        dynamic_model = create_model("DynamicModel", __config__=model_config, **fields)
    except Exception as e:
        raise RuntimeError(
            "Failed to create dynamic model from user-provided types mapping."
        ) from e
    return dynamic_model
