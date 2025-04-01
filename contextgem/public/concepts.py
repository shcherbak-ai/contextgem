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
Module for handling concepts at aspect and document levels.

This module provides classes for defining different types of concepts that can be
extracted from documents and aspects. Concepts represent specific pieces of information
to be identified and extracted by LLMs, such as strings, numbers, boolean values,
JSON objects, and ratings.

Each concept type has specific properties and behaviors tailored to the kind of data
it represents, including validation rules, extraction methods, and reference handling.
Concepts can be attached to documents or aspects and can include examples, justifications,
and references to the source text.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, PrivateAttr, field_validator, model_validator

from contextgem.internal.base.concepts import _Concept
from contextgem.internal.items import (
    _BooleanItem,
    _DateItem,
    _FloatItem,
    _IntegerItem,
    _IntegerOrFloatItem,
    _JsonObjectItem,
    _StringItem,
)
from contextgem.internal.typings.aliases import NonEmptyStr, Self
from contextgem.internal.typings.types_to_strings import (
    _format_type,
    _JsonObjectItemStructure,
)
from contextgem.internal.typings.user_type_hints_validation import (
    _dynamic_pydantic_model,
)
from contextgem.public.data_models import RatingScale
from contextgem.public.examples import JsonObjectExample, StringExample


class StringConcept(_Concept):
    """
    A concept model for string-based information extraction from documents and aspects.

    This class provides functionality for defining, extracting, and managing string data
    as conceptual entities within documents or aspects.

    :ivar name: The name of the concept (non-empty string, stripped).
    :type name: NonEmptyStr
    :ivar description: A brief description of the concept (non-empty string, stripped).
    :type description: NonEmptyStr
    :ivar examples: Example strings illustrating the concept usage.
    :type examples: list[StringExample]
    :ivar llm_role: The role of the LLM responsible for extracting the concept
        ("extractor_text", "reasoner_text", "extractor_vision", "reasoner_vision").
        Defaults to "extractor_text".
    :type llm_role: LLMRoleAny
    :ivar add_justifications: Whether to include justifications for extracted items.
    :type add_justifications: bool
    :ivar justification_depth: Justification detail level. Defaults to "brief".
    :type justification_depth: JustificationDepth
    :ivar justification_max_sents: Maximum sentences in justification. Defaults to 2.
    :type justification_max_sents: int
    :ivar add_references: Whether to include source references for extracted items.
    :type add_references: bool
    :ivar reference_depth: Source reference granularity ("paragraphs" or "sentences").
        Defaults to "paragraphs". Only relevant when references are added to extracted items.
        Affects the structure of ``extracted_items``.
    :type reference_depth: ReferenceDepth
    :ivar singular_occurrence: Whether this concept is restricted to having only one extracted item.
        If True, only a single extracted item will be extracted. Defaults to False (multiple
        extracted items are allowed). Note that with advanced LLMs, this constraint may not be
        strictly required as they can often infer the appropriate cardinality from the concept's
        name, description, and type (e.g., "document title" vs "key findings").
    :type singular_occurrence: StrictBool

    Example:
        .. literalinclude:: ../../../dev/usage_examples/docstrings/concepts/def_string_concept.py
            :language: python
            :caption: String concept definition
    """

    examples: list[StringExample] = Field(default_factory=list)

    _extracted_items: list[_StringItem] = PrivateAttr(default_factory=list)

    @property
    def _item_type_in_prompt(self) -> str:
        return _format_type(str)

    @property
    def _item_class(self) -> type[_StringItem]:
        return _StringItem

    def _process_item_value(self, value: str) -> str:
        return value


class BooleanConcept(_Concept):
    """
    A concept model for boolean (True/False) information extraction from documents and aspects.

    This class handles identification and extraction of boolean values that represent
    conceptual properties or attributes within content.

    :ivar name: The name of the concept (non-empty string, stripped).
    :type name: NonEmptyStr
    :ivar description: A brief description of the concept (non-empty string, stripped).
    :type description: NonEmptyStr
    :ivar llm_role: The role of the LLM responsible for extracting the concept
        ("extractor_text", "reasoner_text", "extractor_vision", "reasoner_vision").
        Defaults to "extractor_text".
    :type llm_role: LLMRoleAny
    :ivar add_justifications: Whether to include justifications for extracted items.
    :type add_justifications: bool
    :ivar justification_depth: Justification detail level. Defaults to "brief".
    :type justification_depth: JustificationDepth
    :ivar justification_max_sents: Maximum sentences in justification. Defaults to 2.
    :type justification_max_sents: int
    :ivar add_references: Whether to include source references for extracted items.
    :type add_references: bool
    :ivar reference_depth: Source reference granularity ("paragraphs" or "sentences").
        Defaults to "paragraphs". Only relevant when references are added to extracted items.
        Affects the structure of ``extracted_items``.
    :type reference_depth: ReferenceDepth
    :ivar singular_occurrence: Whether this concept is restricted to having only one extracted item.
        If True, only a single extracted item will be extracted. Defaults to False (multiple
        extracted items are allowed). Note that with advanced LLMs, this constraint may not be
        strictly required as they can often infer the appropriate cardinality from the concept's
        name, description, and type (e.g., "document title" vs "key findings").
    :type singular_occurrence: StrictBool

    Example:
        .. literalinclude:: ../../../dev/usage_examples/docstrings/concepts/def_boolean_concept.py
            :language: python
            :caption: Boolean concept definition
    """

    _extracted_items: list[_BooleanItem] = PrivateAttr(default_factory=list)

    @property
    def _item_type_in_prompt(self) -> str:
        return _format_type(bool)

    @property
    def _item_class(self) -> type[_BooleanItem]:
        return _BooleanItem

    def _process_item_value(self, value: bool) -> bool:
        return value


class NumericalConcept(_Concept):
    """
    A concept model for numerical information extraction from documents and aspects.

    This class handles identification and extraction of numeric values (integers, floats,
    or both) that represent conceptual measurements or quantities within content.

    :ivar name: The name of the concept (non-empty string, stripped).
    :type name: NonEmptyStr
    :ivar description: A brief description of the concept (non-empty string, stripped).
    :type description: NonEmptyStr
    :ivar numeric_type: Type constraint for extracted numbers ("int", "float", or "any").
        Defaults to "any" for auto-detection.
    :type numeric_type: Literal["int", "float", "any"]
    :ivar llm_role: The role of the LLM responsible for extracting the concept
        ("extractor_text", "reasoner_text", "extractor_vision", "reasoner_vision").
        Defaults to "extractor_text".
    :type llm_role: LLMRoleAny
    :ivar add_justifications: Whether to include justifications for extracted items.
    :type add_justifications: bool
    :ivar justification_depth: Justification detail level. Defaults to "brief".
    :type justification_depth: JustificationDepth
    :ivar justification_max_sents: Maximum sentences in justification. Defaults to 2.
    :type justification_max_sents: int
    :ivar add_references: Whether to include source references for extracted items.
    :type add_references: bool
    :ivar reference_depth: Source reference granularity ("paragraphs" or "sentences").
        Defaults to "paragraphs". Only relevant when references are added to extracted items.
        Affects the structure of ``extracted_items``.
    :type reference_depth: ReferenceDepth
    :ivar singular_occurrence: Whether this concept is restricted to having only one extracted item.
        If True, only a single extracted item will be extracted. Defaults to False (multiple
        extracted items are allowed). Note that with advanced LLMs, this constraint may not be
        strictly required as they can often infer the appropriate cardinality from the concept's
        name, description, and type (e.g., "document title" vs "key findings").
    :type singular_occurrence: StrictBool

    Example:
        .. literalinclude:: ../../../dev/usage_examples/docstrings/concepts/def_numerical_concept.py
            :language: python
            :caption: Numerical concept definition
    """

    numeric_type: Literal["int", "float", "any"] = Field(default="any")

    _extracted_items: list[_IntegerItem | _FloatItem | _IntegerOrFloatItem] = (
        PrivateAttr(default_factory=list)
    )

    @property
    def _item_type_in_prompt(self) -> str:
        if self.numeric_type == "int":
            return _format_type(int)
        elif self.numeric_type == "float":
            return _format_type(float)
        else:  # "any"
            return _format_type(int | float)

    @property
    def _item_class(self) -> type:
        if self.numeric_type == "int":
            return _IntegerItem
        elif self.numeric_type == "float":
            return _FloatItem
        else:  # "any"
            return _IntegerOrFloatItem

    def _process_item_value(self, value: int | float) -> int | float:
        return value


class RatingConcept(_Concept):
    """
    A concept model for rating-based information extraction with defined scale boundaries.

    This class handles identification and extraction of integer ratings that must fall within
    the boundaries of a specified rating scale.

    :ivar name: The name of the concept (non-empty string, stripped).
    :type name: NonEmptyStr
    :ivar description: A brief description of the concept (non-empty string, stripped).
    :type description: NonEmptyStr
    :ivar rating_scale: The rating scale defining valid value boundaries.
    :type rating_scale: RatingScale
    :ivar llm_role: The role of the LLM responsible for extracting the concept
        ("extractor_text", "reasoner_text", "extractor_vision", "reasoner_vision").
        Defaults to "extractor_text".
    :type llm_role: LLMRoleAny
    :ivar add_justifications: Whether to include justifications for extracted items.
    :type add_justifications: bool
    :ivar justification_depth: Justification detail level. Defaults to "brief".
    :type justification_depth: JustificationDepth
    :ivar justification_max_sents: Maximum sentences in justification. Defaults to 2.
    :type justification_max_sents: int
    :ivar add_references: Whether to include source references for extracted items.
    :type add_references: bool
    :ivar reference_depth: Source reference granularity ("paragraphs" or "sentences").
        Defaults to "paragraphs". Only relevant when references are added to extracted items.
        Affects the structure of ``extracted_items``.
    :type reference_depth: ReferenceDepth
    :ivar singular_occurrence: Whether this concept is restricted to having only one extracted item.
        If True, only a single extracted item will be extracted. Defaults to False (multiple
        extracted items are allowed). Note that with advanced LLMs, this constraint may not be
        strictly required as they can often infer the appropriate cardinality from the concept's
        name, description, and type (e.g., "document title" vs "key findings").
    :type singular_occurrence: StrictBool

    Example:
        .. literalinclude:: ../../../dev/usage_examples/docstrings/concepts/def_rating_concept.py
            :language: python
            :caption: Rating concept definition
    """

    rating_scale: RatingScale

    _extracted_items: list[_IntegerItem] = PrivateAttr(default_factory=list)

    @property
    def _item_type_in_prompt(self) -> str:
        return _format_type(int)

    @property
    def _item_class(self) -> type[_IntegerItem]:
        return _IntegerItem

    @property
    def extracted_items(self) -> list[_IntegerItem]:
        return self._extracted_items

    @extracted_items.setter
    def extracted_items(self, value: list[_IntegerItem]) -> None:
        """
        Validates that all values are within the rating scale range,
        in addition to the validation performed by the parent class.

        :param value: The new list of extracted items to be set.
        :type value: list[_IntegerItem]
        :raises ValueError: If any extracted item's value is outside the allowed rating scale range.
        :return: None
        """

        # First, perform rating scale validation
        if value:
            for item in value:
                if not self.rating_scale.start <= item.value <= self.rating_scale.end:
                    raise ValueError(
                        f"Invalid value for scaled rating concept: "
                        f"value {item.value} is outside of the scale range {self.rating_scale.start} to {self.rating_scale.end}"
                    )

        # Then, call the parent class setter for final validation and assignment
        super(RatingConcept, type(self)).extracted_items.fset(self, value)

    def _process_item_value(self, value: int) -> int:
        return value


class JsonObjectConcept(_Concept):
    """
    A concept model for structured JSON object extraction from documents and aspects.

    This class handles identification and extraction of structured data in JSON format,
    with validation against a predefined schema structure.

    :ivar name: The name of the concept (non-empty string, stripped).
    :type name: NonEmptyStr
    :ivar description: A brief description of the concept (non-empty string, stripped).
    :type description: NonEmptyStr
    :ivar structure: JSON object schema as a class with type annotations or dictionary where keys
        are field names and values are type annotations. Supports generic aliases and union types.
        All annotated types must be JSON-serializable. Example: ``{"item": str, "amount": int | float}``.
        **Tip**: do not overcomplicate the structure to avoid prompt overloading. If you need to enforce
        a nested structure (e.g. an object within an object), use type hints together with examples
        that will guide the output format. E.g. structure ``{"item": dict[str, str]}`` and
        example ``{"item": {"name": "item1", "description": "description1"}}``.
    :type structure: type | dict[NonEmptyStr, Any]
    :ivar examples: Example JSON objects illustrating the concept usage.
    :type examples: list[JsonObjectExample]
    :ivar llm_role: The role of the LLM responsible for extracting the concept
        ("extractor_text", "reasoner_text", "extractor_vision", "reasoner_vision").
        Defaults to "extractor_text".
    :type llm_role: LLMRoleAny
    :ivar add_justifications: Whether to include justifications for extracted items.
    :type add_justifications: bool
    :ivar justification_depth: Justification detail level. Defaults to "brief".
    :type justification_depth: JustificationDepth
    :ivar justification_max_sents: Maximum sentences in justification. Defaults to 2.
    :type justification_max_sents: int
    :ivar add_references: Whether to include source references for extracted items.
    :type add_references: bool
    :ivar reference_depth: Source reference granularity ("paragraphs" or "sentences").
        Defaults to "paragraphs". Only relevant when references are added to extracted items.
        Affects the structure of ``extracted_items``.
    :type reference_depth: ReferenceDepth
    :ivar singular_occurrence: Whether this concept is restricted to having only one extracted item.
        If True, only a single extracted item will be extracted. Defaults to False (multiple
        extracted items are allowed). Note that with advanced LLMs, this constraint may not be
        strictly required as they can often infer the appropriate cardinality from the concept's
        name, description, and type (e.g., "document title" vs "key findings").
    :type singular_occurrence: StrictBool

    Example:
        .. literalinclude:: ../../../dev/usage_examples/docstrings/concepts/def_json_object_concept.py
            :language: python
            :caption: JSON object concept definition
    """

    structure: type | dict[NonEmptyStr, Any]
    examples: list[JsonObjectExample] = Field(default_factory=list)

    _extracted_items: list[_JsonObjectItem] = PrivateAttr(default_factory=list)

    @property
    def _item_type_in_prompt(self) -> str:
        return _format_type(dict)

    @property
    def _item_class(self) -> type[_JsonObjectItem]:
        return _JsonObjectItem

    def _get_structure_validator(self) -> type[BaseModel]:
        """
        Creates a dynamic pydantic model from the user-provided type structure.

        :return: Pydantic model class for structure validation.
        :rtype: type[BaseModel]
        """
        return _dynamic_pydantic_model(self.structure)

    @field_validator("structure")
    @classmethod
    def _validate_structure(
        cls, structure: type | dict[str, Any]
    ) -> type | dict[str, Any]:
        """
        Validates that the structure adheres to required format and can be properly rendered.

        :param structure: Class or dictionary defining the JSON structure.
        :type structure: type | dict[str, Any]
        :return: Validated structure if no errors are raised.
        :rtype: type | dict[str, Any]
        :raises ValueError: If structure format is invalid or cannot be properly processed.
        """
        if isinstance(structure, dict):
            if not structure:
                raise ValueError(
                    f"Invalid structure for concept `{cls.__name__}`: empty dictionary"
                )
        try:
            # Check that the prompt types can be rendered in a prompt-compatible string
            _JsonObjectItemStructure(structure)._to_prompt_string()
            # Check that the structure dynamic validation model is created properly
            _dynamic_pydantic_model(structure)
        except (TypeError, ValueError, RuntimeError) as e:
            raise ValueError(
                f"Invalid structure for concept `{cls.__name__}`: {e}"
            ) from e
        return structure

    def _process_item_value(self, value: dict[str, Any]) -> dict[str, Any]:
        """
        Validates a dictionary of new data against a dynamically created pydantic model based
        on mappings from a user-provided type-annotated class or dictionary. This function
        ensures type correctness and adheres to defined constraints in the dynamic model.

        :param value: A dictionary containing the new data to validate. Keys should match
            field names in the dynamically created pydantic model, and values should adhere
            to the corresponding types defined in the dynamic model.
        :type value: dict[str, Any]
        :raises ValidationError: Raised if the provided value fails validation against the
            dynamic model.
        """
        self._get_structure_validator().model_validate(value)
        return value

    @model_validator(mode="after")
    def _validate_json_object_post(self) -> Self:
        """
        Validates example JSON objects against the defined structure.

        :return: The validated model instance.
        :rtype: Self
        :raises ValueError: If any example's structure fails validation against the schema.
        """
        structure_validator_model = self._get_structure_validator()
        try:
            for example in self.examples:
                structure_validator_model.model_validate(example.content)
        except ValueError as e:
            raise ValueError(
                f"Invalid JSON object structure for concept `{self.__class__.__name__}`: {e}"
            ) from e
        return self


class DateConcept(_Concept):
    """
    A concept model for date object extraction from documents and aspects.

    This class handles identification and extraction of dates, with support for parsing
    string representations in a specified format into Python date objects.

    :ivar name: The name of the concept (non-empty string, stripped).
    :type name: NonEmptyStr
    :ivar description: A brief description of the concept (non-empty string, stripped).
    :type description: NonEmptyStr
    :ivar llm_role: The role of the LLM responsible for extracting the concept
        ("extractor_text", "reasoner_text", "extractor_vision", "reasoner_vision").
        Defaults to "extractor_text".
    :type llm_role: LLMRoleAny
    :ivar add_justifications: Whether to include justifications for extracted items.
    :type add_justifications: bool
    :ivar justification_depth: Justification detail level. Defaults to "brief".
    :type justification_depth: JustificationDepth
    :ivar justification_max_sents: Maximum sentences in justification. Defaults to 2.
    :type justification_max_sents: int
    :ivar add_references: Whether to include source references for extracted items.
    :type add_references: bool
    :ivar reference_depth: Source reference granularity ("paragraphs" or "sentences").
        Defaults to "paragraphs". Only relevant when references are added to extracted items.
        Affects the structure of ``extracted_items``.
    :type reference_depth: ReferenceDepth
    :ivar singular_occurrence: Whether this concept is restricted to having only one extracted item.
        If True, only a single extracted item will be extracted. Defaults to False (multiple
        extracted items are allowed). Note that with advanced LLMs, this constraint may not be
        strictly required as they can often infer the appropriate cardinality from the concept's
        name, description, and type (e.g., "document title" vs "key findings").
    :type singular_occurrence: StrictBool

    Example:
        .. literalinclude:: ../../../dev/usage_examples/docstrings/concepts/def_date_concept.py
            :language: python
            :caption: Date concept definition
    """

    _extracted_items: list[_DateItem] = PrivateAttr(default_factory=list)

    @property
    def _date_format_in_prompt(self) -> str:
        return "DD-MM-YYYY"

    @property
    def _item_type_in_prompt(self) -> str:
        return _format_type(str) + f" (in '{self._date_format_in_prompt}' format)"

    @property
    def _item_class(self) -> type[_DateItem]:
        return _DateItem

    def _process_item_value(self, value: str) -> date:
        """
        Transforms a string value into an object expected by the concept.

        :param value: String value to transform.
        :type value: str
        """
        return self._string_to_date(value)

    def _string_to_date(self, date_string: str) -> date:
        """
        Converts a string date representation to a Python date object based on the specified format.

        :param date_string: String representation of a date in the format specified by "_date_format".
        :type date_string: str
        :return: Python date object.
        :rtype: date
        :raises ValueError: If the string can't be parsed according to the specified format.
        """

        # Convert the user-friendly format to Python's strptime format
        py_format = self._date_format_in_prompt
        py_format = py_format.replace("DD", "%d")
        py_format = py_format.replace("MM", "%m")
        py_format = py_format.replace("YYYY", "%Y")

        try:
            return datetime.strptime(date_string.strip(), py_format).date()
        except ValueError as e:
            raise ValueError(
                f"Failed to parse date string '{date_string}' with "
                f"format '{self._date_format_in_prompt}': {e}"
            ) from e
