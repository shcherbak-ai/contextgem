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
Module defining specialized extracted item classes for different data types.

This module implements concrete classes for various data types that extend the
base _ExtractedItem class. Each class provides type-specific validation
and handling for values extracted during aspect and concept analysis.
"""

from __future__ import annotations

from datetime import date
from typing import Any

from pydantic import Field, StrictBool, StrictFloat, StrictInt, field_validator
from typing_extensions import Self

from contextgem.internal.base.items import _ExtractedItem
from contextgem.internal.decorators import _expose_in_registry
from contextgem.internal.typings.types import NonEmptyStr


@_expose_in_registry
class _StringItem(_ExtractedItem):
    """
    Represents an extracted item that holds a string value.

    :ivar value: The string value extracted and validated as non-empty string.
    :vartype value: str
    """

    value: NonEmptyStr = Field(..., frozen=True)


@_expose_in_registry
class _BooleanItem(_ExtractedItem):
    """
    Represents an extracted item that holds a boolean value.

    :ivar value: The strict boolean value associated with this item.
    :vartype value: StrictBool
    """

    value: StrictBool = Field(..., frozen=True)


@_expose_in_registry
class _IntegerItem(_ExtractedItem):
    """
    Represents an extracted item that holds a int value.

    :ivar value: Represents the int value of the item.
    :vartype value: StrictInt
    """

    value: StrictInt = Field(..., frozen=True)


@_expose_in_registry
class _FloatItem(_ExtractedItem):
    """
    Represents an extracted item that holds a float value.

    :ivar value: Represents the float value of the item.
    :vartype value: StrictFloat
    """

    value: StrictFloat = Field(..., frozen=True)


@_expose_in_registry
class _IntegerOrFloatItem(_ExtractedItem):
    """
    Represents an extracted item that holds a int or float value.

    :ivar value: Represents the numerical value of the item. It
        can be either an integer or a float value.
    :vartype value: StrictInt | StrictFloat
    """

    value: StrictInt | StrictFloat = Field(..., frozen=True)


@_expose_in_registry
class _JsonObjectItem(_ExtractedItem):
    """
    Represents an extracted item that holds a JSON object value.

    :ivar value: The JSON object of the item, i.e. a dict with minimum length of 1.
    :vartype value: dict[Any, Any]
    """

    value: dict[Any, Any] = Field(..., min_length=1, frozen=True)

    @field_validator("value")
    @classmethod
    def _validate_value(cls, value: dict[Any, Any]) -> dict[Any, Any]:
        """
        Validates the input dictionary value. Ensures keys and values
        conform to the required structure and data format. If the dictionary has
        nested structures, each level is validated recursively.

        :param value: Dictionary to validate. The dictionary can have nested dicts as
              values, where each level must conform to the specified constraints.
        :type value: dict[Any, Any]
        :return: The same dictionary provided as input, if it passes validation.
        :rtype: dict[Any, Any]
        """

        def validate_recursively(structure: dict[Any, Any]) -> None:
            """
            Recursively validates a dictionary's keys and values.

            :param structure: Dictionary to validate.
            :type structure: dict[Any, Any]
            """

            if not structure:
                raise ValueError("JsonObjectItem value cannot be empty.")

            for key, val in structure.items():
                # Ensure the key is a string
                if not isinstance(key, str):
                    raise ValueError(
                        f"All keys in the structure must be strings. Found key: {repr(key)}"
                    )

                # If the value is a dict, validate it recursively
                if isinstance(val, dict):
                    validate_recursively(val)

        # Start validation from the root structure
        validate_recursively(value)

        return value


@_expose_in_registry
class _DateItem(_ExtractedItem):
    """
    Represents an extracted item that holds a date value.

    :ivar value: The date value extracted and validated as a Python date object.
    :vartype value: date
    """

    value: date = Field(..., frozen=True)

    @field_validator("value")
    @classmethod
    def _validate_date(cls, value: date) -> date:
        """
        Validates that the value is a proper date object.

        :param value: Date object to validate.
        :type value: date
        :return: The same date object if validation passes.
        :rtype: date
        :raises ValueError: If the provided value is not a valid date object.
        """
        if not isinstance(value, date):
            raise ValueError(
                f"Value must be a valid date object. Got {type(value).__name__}."
            )
        return value

    def to_dict(self) -> dict[str, Any]:
        """
        Converts the _DateItem to a dictionary representation.

        Overrides the base to_dict() method to ensure date objects are serialized as strings.

        :return: Dictionary representation of this item with dates converted to strings.
        :rtype: dict[str, Any]
        """
        base_dict = super().to_dict()
        # Convert the date object to ISO 8601 format string for serialization
        base_dict["value"] = self.value.isoformat()
        return base_dict

    @classmethod
    def from_dict(cls, obj_dict: dict[str, Any]) -> Self:
        """
        Reconstructs a _DateItem from its dictionary representation.

        Overrides the base from_dict() method to handle converting the serialized
        date string back to a date object.

        :param obj_dict: Dictionary containing the serialized _DateItem data.
        :type obj_dict: dict[str, Any]
        :return: A new _DateItem instance with the date value restored.
        :rtype: Self
        """
        # Make a copy to avoid modifying the original
        obj_dict_copy = obj_dict.copy()

        # Convert the date string to a date object
        if isinstance(obj_dict_copy["value"], str):
            obj_dict_copy["value"] = date.fromisoformat(obj_dict_copy["value"])

        # Use the parent class's from_dict method
        return super().from_dict(obj_dict_copy)


@_expose_in_registry
class _LabelItem(_ExtractedItem):
    """
    Represents an extracted item that holds a list of label values.

    :ivar value: A list of label strings. Always returns a list for API consistency,
        containing one or more labels depending on the classification type.
    :vartype value: list[str]
    """

    value: list[NonEmptyStr] = Field(..., min_length=1, frozen=True)

    @field_validator("value")
    @classmethod
    def _validate_value(cls, value: list[NonEmptyStr]) -> list[NonEmptyStr]:
        """
        Validates the input list of labels. Ensures there are no duplicates in the list
        (case-insensitive).

        :param value: List of label strings to validate.
        :type value: list[str]
        :return: The same list provided as input, if it passes validation.
        :rtype: list[str]
        :raises ValueError: If the list contains duplicate labels (case-insensitive).
        """
        if len(set(v.lower() for v in value)) < len(value):
            raise ValueError(
                "_LabelItem value cannot contain duplicate labels (case-insensitive)."
            )

        return value
