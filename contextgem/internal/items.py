# SPDX-FileCopyrightText: Shcherbak AI AS
# SPDX-License-Identifier: Apache-2.0

"""
Module defining specialized extracted item classes for different data types.

This module implements concrete classes for various data types that extend the
base _ExtractedItem class. Each class provides type-specific validation
and handling for values extracted during aspect and concept analysis.
"""

from __future__ import annotations

from datetime import date
from typing import Any, Self

from pydantic import Field, StrictBool, StrictFloat, StrictInt, field_validator

from contextgem.internal.base.items import _ExtractedItem
from contextgem.internal.decorators import _expose_in_registry
from contextgem.internal.typings.types import NonEmptyStr


@_expose_in_registry
class _StringItem(_ExtractedItem):
    """
    Represents an extracted item that holds a string value.

    :ivar value: The string value extracted and validated as non-empty string.
    :vartype value: str
    :ivar probabilities: Probabilities of the options of the closed question that
        the item answers, keyed by the options as they appear as item values.
        For an item of an aspect, whose value is the text of a paragraph, ``True``
        is mapped to the probability that the paragraph belongs to the aspect,
        and ``False`` to the probability that it does not, e.g.
        ``{True: 0.83, False: 0.17}``. None (default) if the item has
        no probabilities, which is the case for every item extracted by an LLM.
    :vartype probabilities: dict[bool | int | str, float] | None
    """

    value: NonEmptyStr = Field(..., frozen=True)


@_expose_in_registry
class _BooleanItem(_ExtractedItem):
    """
    Represents an extracted item that holds a boolean value.

    :ivar value: The strict boolean value associated with this item.
    :vartype value: StrictBool
    :ivar probabilities: Probabilities of the options of the yes/no question that
        the item answers, keyed by ``True`` and ``False``, e.g.
        ``{True: 0.83, False: 0.17}``. None (default) if the item has
        no probabilities, which is the case for every item extracted by an LLM.
    :vartype probabilities: dict[bool | int | str, float] | None
    """

    value: StrictBool = Field(..., frozen=True)


@_expose_in_registry
class _IntegerItem(_ExtractedItem):
    """
    Represents an extracted item that holds a int value.

    :ivar value: Represents the int value of the item.
    :vartype value: StrictInt
    :ivar probabilities: Probabilities of the options of the closed question that
        the item answers, keyed by the options as they appear as item values.
        For a rating, these are the integers of the rating scale, e.g.
        ``{1: 0.0, 2: 0.0, 3: 0.15, 4: 0.55, 5: 0.3}``. None (default) if
        the item has no probabilities, which is the case for every item
        extracted by an LLM.
    :vartype probabilities: dict[bool | int | str, float] | None
    """

    value: StrictInt = Field(..., frozen=True)


@_expose_in_registry
class _FloatItem(_ExtractedItem):
    """
    Represents an extracted item that holds a float value.

    :ivar value: Represents the float value of the item.
    :vartype value: StrictFloat
    :ivar probabilities: Probabilities of the options of the closed question that
        the item answers, keyed by the options as they appear as item values.
        None (default) if the item has no probabilities, which is the case
        for every item extracted by an LLM.
    :vartype probabilities: dict[bool | int | str, float] | None
    """

    value: StrictFloat = Field(..., frozen=True)


@_expose_in_registry
class _IntegerOrFloatItem(_ExtractedItem):
    """
    Represents an extracted item that holds a int or float value.

    :ivar value: Represents the numerical value of the item. It
        can be either an integer or a float value.
    :vartype value: StrictInt | StrictFloat
    :ivar probabilities: Probabilities of the options of the closed question that
        the item answers, keyed by the options as they appear as item values.
        None (default) if the item has no probabilities, which is the case
        for every item extracted by an LLM.
    :vartype probabilities: dict[bool | int | str, float] | None
    """

    value: StrictInt | StrictFloat = Field(..., frozen=True)


@_expose_in_registry
class _JsonObjectItem(_ExtractedItem):
    """
    Represents an extracted item that holds a JSON object value.

    :ivar value: The JSON object of the item, i.e. a dict with minimum length of 1.
    :vartype value: dict[Any, Any]
    :ivar probabilities: Probabilities of the options of the closed question that
        the item answers, keyed by the options as they appear as item values.
        None (default) if the item has no probabilities, which is the case
        for every item extracted by an LLM.
    :vartype probabilities: dict[bool | int | str, float] | None
    """

    value: dict[Any, Any] = Field(..., min_length=1, frozen=True)

    @field_validator("value")
    @classmethod
    def _validate_value(cls, value: dict[Any, Any]) -> dict[Any, Any]:
        """
        Validates the input dictionary value. Ensures that the keys are strings,
        also those of the nested dictionaries, at any depth. The dictionary itself
        must not be empty (see the ``min_length`` of the field), while a nested
        dictionary may be, e.g. the value of a mapping field with no entries:
        the values are validated against the structure of the concept.

        :param value: Dictionary to validate. The dictionary can have nested dicts as
              values, whose keys must be strings as well.
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

            for key, val in structure.items():
                # Ensure the key is a string
                if not isinstance(key, str):
                    raise ValueError(  # noqa: TRY004 - pydantic and the callers handle ValueError
                        f"All keys in the structure must be strings. Found key: {key!r}"
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
    :ivar probabilities: Probabilities of the options of the closed question that
        the item answers, keyed by the options as they appear as item values.
        None (default) if the item has no probabilities, which is the case
        for every item extracted by an LLM.
    :vartype probabilities: dict[bool | int | str, float] | None
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
            raise ValueError(  # noqa: TRY004 - pydantic and the callers handle ValueError
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
    :ivar probabilities: Probabilities of the labels that the item could be
        classified with, keyed by the labels. Where only one label can apply,
        these are the probabilities of the alternatives, e.g. ``{"written": 0.71,
        "email": 0.2, "any": 0.06, "not specified": 0.03}``. Where several labels
        can apply, each label has its own probability, i.e. the probabilities
        do not sum to 1, e.g. ``{"NDA": 0.97, "Confidential": 0.62,
        "Employment": 0.08}``. None (default) if the item has no probabilities,
        which is the case for every item extracted by an LLM.
    :vartype probabilities: dict[bool | int | str, float] | None
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
        if len({v.lower() for v in value}) < len(value):
            raise ValueError(
                "_LabelItem value cannot contain duplicate labels (case-insensitive)."
            )

        return value
