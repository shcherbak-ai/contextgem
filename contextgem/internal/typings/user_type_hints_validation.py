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

from types import GenericAlias, UnionType
from typing import Union, get_args, get_origin, get_type_hints

from pydantic import BaseModel, ConfigDict, create_model

from contextgem.internal.typings.types_to_strings import _is_json_serializable_type


def _extract_mapper(
    user_mapper: type | dict[str, type | GenericAlias | UnionType],
) -> dict[str, type | GenericAlias | UnionType]:
    """
    Extracts a dictionary mapping field names to their corresponding type hints from a
    provided user_mapper. This utility supports two forms of input for the user_mapper:

    1. A class containing type-annotated attributes.
    2. A dictionary where keys are field names and values represent type hints.

    Raises an error if the provided user_mapper does not adhere to these forms.

    :param user_mapper: A user_mapper defined either as a type-annotated class or a
        dictionary mapping string field names to type hints.
    :type user_mapper: type | dict[str, type | GenericAlias | UnionType]

    :return: A dictionary mapping field names to type hints, derived from the input user_mapper.
    :rtype: dict[str, type | GenericAlias | UnionType]

    :raises ValueError: If the input user_mapper is neither a dictionary nor a class
        containing type-annotated attributes.
    """

    if isinstance(user_mapper, type):
        # Use get_type_hints to resolve annotations, handling forward references.
        type_hints = get_type_hints(user_mapper)
        if not type_hints:
            raise ValueError(
                "user_mapper must contain at least one type-annotated attribute."
            )
        return type_hints

    elif isinstance(user_mapper, dict):
        return user_mapper

    else:
        raise ValueError("user_mapper must be a dict or a class with type annotations.")


def _is_optional(type_hint: type | GenericAlias | UnionType) -> bool:
    """
    Determines whether a given type hint is of an optional type.

    An optional type means that the type hint allows `None` as a valid value. This
    is typically represented by a `Union` that contains `NoneType`.

    :param type_hint: The type hint to check. It can be a standard Python class,
        a `GenericAlias`, or a `UnionType` defined using `Union` or the `|`
        operator.
    :return: A boolean value indicating whether the provided type hint is optional.
    :rtype: bool
    """
    origin = get_origin(type_hint)
    if origin is Union:
        return type(None) in get_args(type_hint)
    return False


def _dynamic_pydantic_model(
    user_mapper: type | dict[str, type | GenericAlias | UnionType],
) -> type[BaseModel]:
    """
    Dynamically generates a pydantic model class based on a provided mapping of field names
    to their corresponding type annotations. This function is valuable for scenarios where
    data structures are defined dynamically or at runtime, allowing the user to construct
    strictly-typed schema without manually creating pydantic models for every variation.

    :param user_mapper: A class with type annotations or a dictionary where keys are the field names
        (str) and the values are valid type hints (e.g., int, float, str | None, etc.). These hints
        dictate the types of the fields included in the generated model.
    :type user_mapper: type | dict[str, type | GenericAlias | UnionType]

    :return: A dynamically created subclass of pydantic's BaseModel that adheres to the
        type constraints and field definitions supplied in the user_mapper.
    :rtype: type[BaseModel]
    """

    # Build the fields dict expected by create_model. Each field gets a tuple: (type, default)
    fields = {}
    for field_name, field_type in _extract_mapper(user_mapper).items():
        if field_name.startswith("_"):
            raise ValueError(f"Field {field_name} cannot start with '_'")
        # Check that the mapper value is a valid type hint.
        if not (isinstance(field_type, type) or get_origin(field_type) is not None):
            raise ValueError(
                f"Field '{field_name}' has an invalid type hint: {field_type!r}. "
                "It must be a type, a union of types, or a generic alias."
            )
        # Check for JSON-serializability of the type hint
        if not _is_json_serializable_type(field_type):
            raise ValueError(
                f"Field '{field_name}' has an invalid type hint: {field_type!r}. "
                "It must be a JSON-serializable type or a Union thereof."
            )
        # If the type includes None, assume it’s optional and give it a default of None.
        if _is_optional(field_type):
            fields[field_name] = (field_type, None)
        else:
            fields[field_name] = (field_type, ...)

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )
    try:
        DynamicModel = create_model("DynamicModel", __config__=model_config, **fields)
    except Exception as e:
        raise RuntimeError(
            "Failed to create dynamic model from user-provided types mapping."
        ) from e
    return DynamicModel
