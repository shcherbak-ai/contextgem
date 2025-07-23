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
Module for validating user-provided type hints.

This module provides functionality to validate and process type hints supplied by users
in various contexts throughout the ContextGem framework. It includes utilities for extracting
type information from different input formats, validating type compatibility with JSON
serialization requirements, and ensuring type consistency across the application.
"""

from typing import Any, Union, get_args, get_origin

from pydantic import BaseModel, ConfigDict, create_model

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
        raise ValueError("`user_mapper` must be a dict.")


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
    if origin is Union:
        return type(None) in get_args(type_hint)
    return False


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

    # Build the fields dict expected by create_model. Each field gets a tuple: (type, default)
    fields = {}
    for field_name, field_type in _extract_mapper(user_mapper).items():
        if field_name.startswith("_"):
            raise ValueError(f"Field {field_name} cannot start with '_'")

        # Handle nested dictionaries as nested models
        if isinstance(field_type, dict):
            # Recursively create a nested model for this field
            nested_model = _dynamic_pydantic_model(field_type)
            fields[field_name] = (nested_model, ...)
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
            fields[field_name] = (list[nested_model], ...)
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

            # Handle nested class types recursively
            if isinstance(item_type, dict):
                nested_model = _dynamic_pydantic_model(item_type)
                fields[field_name] = (list[nested_model], ...)
            else:
                fields[field_name] = (list[item_type], ...)
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
        # If the type includes None, assume it's optional and give it a default of None.
        if _is_optional(field_type):
            fields[field_name] = (field_type, None)
        else:
            fields[field_name] = (field_type, ...)

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
