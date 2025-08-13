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
Utilities for identifying and extracting type information from classes.

Provides functions to detect classes with type annotations (like Pydantic models
or dataclasses) and extract their field type definitions for schema generation.
"""

from __future__ import annotations

from dataclasses import MISSING, is_dataclass
from typing import Any, get_type_hints

from pydantic import BaseModel

from contextgem.internal.loggers import logger


def _has_explicit_field_objects(cls: Any) -> bool:
    """
    Returns True if a dataclass defines at least one field via `field(...)`
    (i.e., has non-MISSING default, non-MISSING default_factory, or non-empty metadata).

    :param cls: Dataclass to check
    :return: True if it has explicit field objects
    """
    if not is_dataclass(cls):
        return False

    for field in cls.__dataclass_fields__.values():
        if (
            field.default is not MISSING
            or field.default_factory is not MISSING
            or bool(field.metadata)
        ):
            return True
    return False


def _has_custom_validation_or_field_options(cls: Any) -> bool:
    """
    Returns True if a Pydantic model has custom validation logic or field options
    that would be lost when extracting only type hints.

    This function checks for:
    1. Field validators (@field_validator decorated methods)
    2. Model validators (@model_validator decorated methods)
    3. Field-level options (defaults, validation constraints, etc.)

    :param cls: Pydantic model class to check
    :return: True if it has validation logic or field options beyond basic type annotations
    """
    if not issubclass(cls, BaseModel):
        return False

    # Check for field and/or model validators
    if hasattr(cls, "__pydantic_decorators__"):
        decorators = cls.__pydantic_decorators__

        # Check field_validators
        if hasattr(decorators, "field_validators") and decorators.field_validators:
            return True

        # Check model_validators
        if hasattr(decorators, "model_validators") and decorators.model_validators:
            return True

    # Check for field-level options
    if hasattr(cls, "model_fields"):
        for _, field in cls.model_fields.items():
            # Check if default is explicitly set
            if str(field.default) != "PydanticUndefined" and field.default is not None:
                return True

            # Check for default factory
            if field.default_factory is not None:
                return True

            # Check for additional validation or serialization options
            if any(
                [
                    getattr(field, "json_schema_extra", None),
                    getattr(field, "validation_alias", None),
                    getattr(field, "serialization_alias", None),
                    getattr(field, "deprecated", False),
                    getattr(field, "discriminator", None),
                ]
            ):
                return True

    return False


def _is_typed_class(cls) -> bool:
    """
    Checks if a class is a Pydantic model or has the necessary structure
    to be treated as a type-annotated class.

    :param cls: Class to check
    :return: True if it's a Pydantic model or class with type annotations
    """
    # Check if it's a class (not an instance)
    if not isinstance(cls, type):
        return False

    # Check for pydantic models
    if issubclass(cls, BaseModel):
        if _has_custom_validation_or_field_options(cls):
            logger.warning(
                f"Pydantic model '{cls.__name__}' contains field validation or serialization logic "
                f"that will be discarded. Only type hints will be included in the structure."
            )
        return True

    # Check for dataclasses
    if is_dataclass(cls):
        if _has_explicit_field_objects(cls):
            logger.warning(
                f"Dataclass '{cls.__name__}' contains field metadata or options "
                f"that will be discarded. Only type hints will be included in the structure."
            )
        return True

    # Check if it has type annotations
    return bool(hasattr(cls, "__annotations__") and cls.__annotations__)


def _get_model_fields(cls) -> dict[str, Any]:
    """
    Extracts field type annotations from a Pydantic model or class with type annotations.

    :param cls: Model class to extract fields from
    :return: Dictionary of field names to types
    """
    return get_type_hints(cls)


def _raise_dict_class_type_error(
    type_name: str, field_name: str, type_arg: Any
) -> None:
    """
    Raises an error for dictionary fields containing class types in their type annotations.

    :param type_name: The type position ("key" or "value") in the dictionary type annotation
    :param field_name: The field name containing the dictionary type
    :param type_arg: The offending class type
    :raises TypeError: Formatted error message about class in dictionary type annotation
    """
    raise TypeError(
        f"Invalid {type_name} type for dictionary field '{field_name}': {type_arg}.\n"
        f"Class types are not allowed as type annotations for dictionary {type_name}s "
        f"(like dict[Class, ...] or dict[..., Class]). "
        f"Use basic types for type annotations or restructure your model. "
        f"Dictionary instances containing class instances "
        f"(like {{'key': ClassInstance}}) are still allowed."
    )
