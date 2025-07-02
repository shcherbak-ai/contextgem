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
from types import UnionType
from typing import Any, List, Literal, Union, get_args, get_origin

from pydantic import BaseModel, Field, PrivateAttr, field_validator, model_validator

from contextgem.internal.base.concepts import _Concept
from contextgem.internal.items import (
    _BooleanItem,
    _DateItem,
    _FloatItem,
    _IntegerItem,
    _IntegerOrFloatItem,
    _JsonObjectItem,
    _LabelItem,
    _StringItem,
)
from contextgem.internal.llm_output_structs.concept_structs import (
    _LabelConceptItemValueModel,
)
from contextgem.internal.typings.aliases import ClassificationType, NonEmptyStr, Self
from contextgem.internal.typings.typed_class_utils import (
    _get_model_fields,
    _is_typed_class,
    _raise_dict_class_type_error,
)
from contextgem.internal.typings.types_normalization import _normalize_type_annotation
from contextgem.internal.typings.types_to_strings import (
    JSON_PRIMITIVE_TYPES,
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
    :vartype name: NonEmptyStr
    :ivar description: A brief description of the concept (non-empty string, stripped).
    :vartype description: NonEmptyStr
    :ivar examples: Example strings illustrating the concept usage.
    :vartype examples: list[StringExample]
    :ivar llm_role: The role of the LLM responsible for extracting the concept
        ("extractor_text", "reasoner_text", "extractor_vision", "reasoner_vision").
        Defaults to "extractor_text".
    :vartype llm_role: LLMRoleAny
    :ivar add_justifications: Whether to include justifications for extracted items.
    :vartype add_justifications: bool
    :ivar justification_depth: Justification detail level. Defaults to "brief".
    :vartype justification_depth: JustificationDepth
    :ivar justification_max_sents: Maximum sentences in justification. Defaults to 2.
    :vartype justification_max_sents: int
    :ivar add_references: Whether to include source references for extracted items.
    :vartype add_references: bool
    :ivar reference_depth: Source reference granularity ("paragraphs" or "sentences").
        Defaults to "paragraphs". Only relevant when references are added to extracted items.
        Affects the structure of ``extracted_items``.
    :vartype reference_depth: ReferenceDepth
    :ivar singular_occurrence: Whether this concept is restricted to having only one extracted item.
        If True, only a single extracted item will be extracted. Defaults to False (multiple
        extracted items are allowed). Note that with advanced LLMs, this constraint may not be
        strictly required as they can often infer the appropriate number of items to extract
        from the concept's name, description, and type (e.g., "document title" vs "key findings").
    :vartype singular_occurrence: StrictBool

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
    :vartype name: NonEmptyStr
    :ivar description: A brief description of the concept (non-empty string, stripped).
    :vartype description: NonEmptyStr
    :ivar llm_role: The role of the LLM responsible for extracting the concept
        ("extractor_text", "reasoner_text", "extractor_vision", "reasoner_vision").
        Defaults to "extractor_text".
    :vartype llm_role: LLMRoleAny
    :ivar add_justifications: Whether to include justifications for extracted items.
    :vartype add_justifications: bool
    :ivar justification_depth: Justification detail level. Defaults to "brief".
    :vartype justification_depth: JustificationDepth
    :ivar justification_max_sents: Maximum sentences in justification. Defaults to 2.
    :vartype justification_max_sents: int
    :ivar add_references: Whether to include source references for extracted items.
    :vartype add_references: bool
    :ivar reference_depth: Source reference granularity ("paragraphs" or "sentences").
        Defaults to "paragraphs". Only relevant when references are added to extracted items.
        Affects the structure of ``extracted_items``.
    :vartype reference_depth: ReferenceDepth
    :ivar singular_occurrence: Whether this concept is restricted to having only one extracted item.
        If True, only a single extracted item will be extracted. Defaults to False (multiple
        extracted items are allowed). Note that with advanced LLMs, this constraint may not be
        strictly required as they can often infer the appropriate number of items to extract
        from the concept's name, description, and type (e.g., "contains confidential information"
        vs "compliance violations").
    :vartype singular_occurrence: StrictBool

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
    :vartype name: NonEmptyStr
    :ivar description: A brief description of the concept (non-empty string, stripped).
    :vartype description: NonEmptyStr
    :ivar numeric_type: Type constraint for extracted numbers ("int", "float", or "any").
        Defaults to "any" for auto-detection.
    :vartype numeric_type: Literal["int", "float", "any"]
    :ivar llm_role: The role of the LLM responsible for extracting the concept
        ("extractor_text", "reasoner_text", "extractor_vision", "reasoner_vision").
        Defaults to "extractor_text".
    :vartype llm_role: LLMRoleAny
    :ivar add_justifications: Whether to include justifications for extracted items.
    :vartype add_justifications: bool
    :ivar justification_depth: Justification detail level. Defaults to "brief".
    :vartype justification_depth: JustificationDepth
    :ivar justification_max_sents: Maximum sentences in justification. Defaults to 2.
    :vartype justification_max_sents: int
    :ivar add_references: Whether to include source references for extracted items.
    :vartype add_references: bool
    :ivar reference_depth: Source reference granularity ("paragraphs" or "sentences").
        Defaults to "paragraphs". Only relevant when references are added to extracted items.
        Affects the structure of ``extracted_items``.
    :vartype reference_depth: ReferenceDepth
    :ivar singular_occurrence: Whether this concept is restricted to having only one extracted item.
        If True, only a single extracted item will be extracted. Defaults to False (multiple
        extracted items are allowed). Note that with advanced LLMs, this constraint may not be
        strictly required as they can often infer the appropriate number of items to extract
        from the concept's name, description, and type (e.g., "total revenue" vs
        "monthly sales figures").
    :vartype singular_occurrence: StrictBool

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
    :vartype name: NonEmptyStr
    :ivar description: A brief description of the concept (non-empty string, stripped).
    :vartype description: NonEmptyStr
    :ivar rating_scale: The rating scale defining valid value boundaries.
    :vartype rating_scale: RatingScale
    :ivar llm_role: The role of the LLM responsible for extracting the concept
        ("extractor_text", "reasoner_text", "extractor_vision", "reasoner_vision").
        Defaults to "extractor_text".
    :vartype llm_role: LLMRoleAny
    :ivar add_justifications: Whether to include justifications for extracted items.
    :vartype add_justifications: bool
    :ivar justification_depth: Justification detail level. Defaults to "brief".
    :vartype justification_depth: JustificationDepth
    :ivar justification_max_sents: Maximum sentences in justification. Defaults to 2.
    :vartype justification_max_sents: int
    :ivar add_references: Whether to include source references for extracted items.
    :vartype add_references: bool
    :ivar reference_depth: Source reference granularity ("paragraphs" or "sentences").
        Defaults to "paragraphs". Only relevant when references are added to extracted items.
        Affects the structure of ``extracted_items``.
    :vartype reference_depth: ReferenceDepth
    :ivar singular_occurrence: Whether this concept is restricted to having only one extracted item.
        If True, only a single extracted item will be extracted. Defaults to False (multiple
        extracted items are allowed). Note that with advanced LLMs, this constraint may not be
        strictly required as they can often infer the appropriate number of items to extract
        from the concept's name, description, and type (e.g., "product rating score" vs
        "customer satisfaction ratings").
    :vartype singular_occurrence: StrictBool

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
                self._validate_rating_value(item.value)

        # Then, call the parent class setter for final validation and assignment
        super(RatingConcept, type(self)).extracted_items.fset(self, value)

    def _validate_rating_value(self, rating_value: int) -> None:
        """
        Validates rating values against concept-specific business logic.

        Checks that the rating value is within the defined rating scale range.

        :param rating_value: Rating value to validate.
        :type rating_value: int
        :raises ValueError: If the rating value is outside the allowed rating scale range.
        """
        if not self.rating_scale.start <= rating_value <= self.rating_scale.end:
            raise ValueError(
                f"Invalid value for scaled rating concept: "
                f"value {rating_value} is outside of the scale range "
                f"{self.rating_scale.start} to {self.rating_scale.end}"
            )

    def _process_item_value(self, value: int) -> int:
        """
        Validates the rating value against the rating scale and returns it.

        :param value: Integer rating value to validate.
        :type value: int
        :return: Validated rating value.
        :rtype: int
        :raises ValueError: If the value is outside the allowed rating scale range.
        """
        # Apply concept-specific business logic validation
        self._validate_rating_value(value)

        return value


class JsonObjectConcept(_Concept):
    """
    A concept model for structured JSON object extraction from documents and aspects.

    This class handles identification and extraction of structured data in JSON format,
    with validation against a predefined schema structure.

    :ivar name: The name of the concept (non-empty string, stripped).
    :vartype name: NonEmptyStr
    :ivar description: A brief description of the concept (non-empty string, stripped).
    :vartype description: NonEmptyStr
    :ivar structure: JSON object schema as a class with type annotations or dictionary where keys
        are field names and values are type annotations. All dictionary keys must be strings.
        Supports generic aliases, union types, nested dictionaries for complex hierarchical structures,
        lists of dictionaries for array items, Literal types, and classes with type annotations
        (Pydantic models, dataclasses, etc.) for nested structures. All annotated types must be
        JSON-serializable.
        Examples:

        - Simple structure: ``{"item": str, "amount": int | float}``
        - Nested structure: ``{"item": str, "details": {"price": float, "quantity": int}}``
        - List of objects: ``{"items": [{"name": str, "price": float}]}``
        - List of primitives: ``{"names": [str], "scores": [int | float], "statuses": [Literal["active", "inactive"]]}``
        - List of classes: ``{"addresses": [AddressModel], "users": [UserModel]}``
        - Literal values: ``{"status": Literal["pending", "completed", "failed"]}``
        - With type annotated classes: ``{"address": AddressModel}`` where AddressModel can be a
          Pydantic model, dataclass, or any class with type annotations

        **Note**: For lists, you can use either generic syntax (``list[str]``) or literal syntax
        (``[str]``). List instances support primitive types, unions, literals, and typed classes.
        Both ``{"items": [ClassName]}`` and ``{"items": list[ClassName]}`` are equivalent.

        **Note**: Class types cannot be used as dictionary keys or values. For example,
        ``dict[str, Address]`` is not allowed. Use alternative structures like nested objects
        or lists of objects instead.

        **Note**: When using classes that contain other classes as type hints, inherit from
        ``JsonObjectClassStruct`` in all parts of the class hierarchy, to ensure proper conversion
        of nested class hierarchies to dictionary representations for serialization.

        **Tip**: do not overcomplicate the structure to avoid prompt overloading.
    :vartype structure: type | dict[NonEmptyStr, Any]
    :ivar examples: Example JSON objects illustrating the concept usage.
    :vartype examples: list[JsonObjectExample]
    :ivar llm_role: The role of the LLM responsible for extracting the concept
        ("extractor_text", "reasoner_text", "extractor_vision", "reasoner_vision").
        Defaults to "extractor_text".
    :vartype llm_role: LLMRoleAny
    :ivar add_justifications: Whether to include justifications for extracted items.
    :vartype add_justifications: bool
    :ivar justification_depth: Justification detail level. Defaults to "brief".
    :vartype justification_depth: JustificationDepth
    :ivar justification_max_sents: Maximum sentences in justification. Defaults to 2.
    :vartype justification_max_sents: int
    :ivar add_references: Whether to include source references for extracted items.
    :vartype add_references: bool
    :ivar reference_depth: Source reference granularity ("paragraphs" or "sentences").
        Defaults to "paragraphs". Only relevant when references are added to extracted items.
        Affects the structure of ``extracted_items``.
    :vartype reference_depth: ReferenceDepth
    :ivar singular_occurrence: Whether this concept is restricted to having only one extracted item.
        If True, only a single extracted item will be extracted. Defaults to False (multiple
        extracted items are allowed). Note that with advanced LLMs, this constraint may not be
        strictly required as they can often infer the appropriate number of items to extract
        from the concept's name, description, and type (e.g., "product specifications" vs
        "customer order details").
    :vartype singular_occurrence: StrictBool

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

    def _format_structure_in_prompt(self) -> str:
        """
        Formats the structure for use in prompts, ensuring consistency
        regardless of how the structure was originally defined.

        :return: A string representation of the structure for prompts.
        :rtype: str
        """
        # Use the JsonObjectItemStructure to format the structure consistently
        formatter = _JsonObjectItemStructure(self.structure)
        return formatter._to_prompt_string()

    def _get_structure_validator(self) -> type[BaseModel]:
        """
        Creates a dynamic pydantic model from the user-provided type structure.

        :return: Pydantic model class for structure validation.
        :rtype: type[BaseModel]
        """
        return _dynamic_pydantic_model(self.structure)

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

    @classmethod
    def _convert_structure_to_dict(
        cls, structure: type | dict[str, Any]
    ) -> dict[str, Any]:
        """
        Converts the structure (class or dictionary) to a standardized dictionary format
        with serializable type hints for consistent processing and serialization.

        :param structure: Class or dictionary defining the JSON structure.
        :type structure: type | dict[str, Any]
        :return: Dictionary representation of the structure.
        :rtype: dict[str, Any]
        :raises ValueError: If conversion fails.
        """
        # Normalize any type hints including typing module generics
        structure = _normalize_type_annotation(structure)

        # Convert class-based structures to dictionaries
        if not isinstance(structure, dict) and _is_typed_class(structure):
            try:
                # Handle classes that inherit from JsonObjectClassStruct
                # (classes with _as_dict_structure method)
                if hasattr(structure, "_as_dict_structure"):
                    structure = structure._as_dict_structure()
                # Handle other typed classes (such as Pydantic models, dataclasses)
                else:
                    structure = _get_model_fields(structure)

                # Normalize the result to ensure consistent type representation
                structure = {
                    k: _normalize_type_annotation(v) for k, v in structure.items()
                }
            except Exception as e:
                raise ValueError(
                    f"Invalid structure for concept `{cls.__name__}`: {e}"
                ) from e

        # Ensure we have a dictionary at this point
        if not isinstance(structure, dict):
            raise ValueError(
                f"Invalid structure for concept `{cls.__name__}`: "
                f"Processed structure must be a dict."
            )

        if not structure:
            raise ValueError(
                f"Invalid structure for concept `{cls.__name__}`: empty dictionary"
            )

        # Recursively process dictionary to handle nested classes and special types
        processed_structure = {}
        for key, value in structure.items():
            processed_structure[key] = cls._process_structure_value(value, f"{key}")

        return processed_structure

    @classmethod
    def _process_structure_value(cls, value: Any, path: str = "") -> Any:
        """
        Recursively processes a structure value, handling all nested scenarios.

        :param value: The value to process (class, dict, list, or other type).
        :param path: Current path in the structure for error reporting.
        :return: Processed value with all nested classes converted to dictionaries.
        :raises ValueError: If processing fails for any nested component.
        """

        # Normalize the type annotation for consistent representation
        value = _normalize_type_annotation(value)

        # Handle class with _as_dict_structure method
        if hasattr(value, "_as_dict_structure"):
            try:
                dict_structure = value._as_dict_structure()
                # Process each item in the dictionary
                return {
                    k: cls._process_structure_value(v, f"{path}.{k}")
                    for k, v in dict_structure.items()
                }
            except Exception as e:
                raise ValueError(
                    f"Invalid structure at `{path}` in concept `{cls.__name__}`: {e}"
                ) from e

        # Handle other typed classes
        elif _is_typed_class(value):
            try:
                fields = _get_model_fields(value)
                return {
                    k: cls._process_structure_value(v, f"{path}.{k}")
                    for k, v in fields.items()
                }
            except Exception as e:
                raise ValueError(
                    f"Invalid structure at `{path}` in concept `{cls.__name__}`: {e}"
                ) from e

        # Handle list type hint (must have exactly one element for type annotation)
        elif isinstance(value, list):
            if not value or len(value) != 1:
                raise ValueError(
                    f"Invalid list at `{path}` in concept `{cls.__name__}`: "
                    f"List must contain exactly one element representing the item type"
                )

            item_type = value[0]

            # If it's a dictionary, process as nested structure
            if isinstance(item_type, dict):
                processed_item = cls._process_structure_value(item_type, f"{path}[0]")
                return [processed_item]

            # If it's not a dictionary, validate that it's an allowed type for list instances
            # Allow primitive types, unions, literals, and typed classes
            if (
                # Primitive types
                item_type in JSON_PRIMITIVE_TYPES
                or
                # Union types (including Optional) - both old Union and new | syntax
                (hasattr(item_type, "__origin__") and get_origin(item_type) is Union)
                or isinstance(item_type, UnionType)
                or
                # Literal types
                (hasattr(item_type, "__origin__") and get_origin(item_type) is Literal)
                or
                # Typed classes (Pydantic models, dataclasses, etc.)
                _is_typed_class(item_type)
            ):
                # For allowed types, process normally and return
                processed_item = cls._process_structure_value(item_type, f"{path}[0]")
                return [processed_item]
            else:
                # For other complex types, provide helpful error message
                raise ValueError(
                    f"Invalid list instance at `{path}` in concept `{cls.__name__}`: "
                    f"List instances can only contain primitive types (str, int, float, bool, None), "
                    f"union types (str | int, Optional[str]), literal types (Literal['a', 'b']), "
                    f"or typed classes (Pydantic models, dataclasses, etc.). "
                    f"Got: {item_type}"
                )

        # Handle nested dictionary
        elif isinstance(value, dict):
            processed_dict = {}
            for k, v in value.items():
                # Validate that the key is a string
                if not isinstance(k, str):
                    raise ValueError(
                        f"Invalid dictionary key at `{path}`: {k}. "
                        f"Dictionary keys must be strings, got {type(k).__name__}"
                    )
                processed_dict[k] = cls._process_structure_value(v, f"{path}.{k}")
            return processed_dict

        # Handle special generic types like list[TypedClass]
        elif hasattr(value, "__origin__") and getattr(value, "__origin__", None) in (
            list,
            List,
        ):
            if len(value.__args__) > 1:
                raise ValueError(
                    f"Invalid list type annotation at `{path}` in concept `{cls.__name__}`: "
                    f"List must have exactly one type argument, got {len(value.__args__)}"
                )
            if len(value.__args__) == 1:
                arg_type = value.__args__[0]
                # Normalize the argument type
                arg_type = _normalize_type_annotation(arg_type)

                if _is_typed_class(arg_type):
                    if hasattr(arg_type, "_as_dict_structure"):
                        dict_structure = arg_type._as_dict_structure()
                        return [dict_structure]
                    else:
                        fields = _get_model_fields(arg_type)
                        processed_fields = {
                            k: cls._process_structure_value(v, f"{path}[0].{k}")
                            for k, v in fields.items()
                        }
                        return [processed_fields]
                else:
                    # For basic types, ensure we're using built-in list
                    return list[arg_type]

        # Handle Optional type hints and ensure they only contain primitive types
        elif (
            hasattr(value, "__origin__") and getattr(value, "__origin__", None) is Union
        ):
            # Check if it's an Optional (Union with None)
            args = get_args(value)
            is_optional = type(None) in args

            if is_optional:
                # For each type argument (except None), check if it's a primitive type or Literal
                for arg in args:
                    if arg is not type(None):
                        # Allow Literal types in Optional
                        if getattr(arg, "__origin__", None) is Literal:
                            # Ensure all literal values are JSON serializable
                            for literal_arg in get_args(arg):
                                if not isinstance(literal_arg, JSON_PRIMITIVE_TYPES):
                                    raise ValueError(
                                        f"Invalid Literal value in Optional[Literal] at `{path}` "
                                        f"in concept `{cls.__name__}`: "
                                        f"Literal value '{literal_arg}' is not a JSON primitive type."
                                    )
                        # For non-Literal types, check for primitive types
                        elif arg not in JSON_PRIMITIVE_TYPES:
                            raise ValueError(
                                f"Invalid Optional type at `{path}` in concept `{cls.__name__}`: "
                                f"Optional[{arg.__name__}] is not allowed. "
                                f"Optional[] can only be used with primitive types "
                                f"(str, int, float, bool) or Literal types, "
                                f"not with classes or complex types. "
                                f"Use basic types or restructure your model to avoid "
                                f"Optional with non-primitive types."
                            )

            # Return the original value after validation
            return value

        # Handle dictionary type hints
        elif (
            hasattr(value, "__origin__") and getattr(value, "__origin__", None) is dict
        ):
            if len(value.__args__) == 2:
                key_type, val_type = value.__args__

                # Check if key or value types are class types (which should be disallowed)
                for type_arg, type_name in [(key_type, "key"), (val_type, "value")]:
                    # Check if the type is a class (either directly or through a string reference)
                    if _is_typed_class(type_arg) or (
                        isinstance(type_arg, str)
                        and hasattr(globals().get(type_arg, None), "__annotations__")
                    ):
                        # Use ValueError here because we're in a structure validation context
                        # but reuse the error message structure from the shared function
                        try:
                            _raise_dict_class_type_error(type_name, path, type_arg)
                        except TypeError as e:
                            # Convert to ValueError to maintain consistent error type
                            raise ValueError(str(e))

            return value

        # All other types remain normalized
        return value

    @field_validator("structure")
    @classmethod
    def _validate_structure(
        cls, structure: type | dict[str, Any]
    ) -> type | dict[str, Any]:
        """
        Validates that the structure adheres to required format and can be properly rendered.
        Converts all structure types (including classes) to a standardized dictionary format
        with serializable type hints.

        :param structure: Class or dictionary defining the JSON structure.
        :type structure: type | dict[str, Any]
        :return: Validated structure converted to a dictionary representation.
        :rtype: dict[str, Any]
        :raises ValueError: If structure format is invalid or cannot be properly processed.
        """
        processed_structure = cls._convert_structure_to_dict(structure)

        try:
            # Check that the prompt types can be rendered in a prompt-compatible string
            _JsonObjectItemStructure(processed_structure)._to_prompt_string()
            # Check that the structure dynamic validation model is created properly
            _dynamic_pydantic_model(processed_structure)
        except (TypeError, ValueError, RuntimeError) as e:
            raise ValueError(
                f"Invalid structure for concept `{cls.__name__}`: {e}"
            ) from e
        return processed_structure

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
    :vartype name: NonEmptyStr
    :ivar description: A brief description of the concept (non-empty string, stripped).
    :vartype description: NonEmptyStr
    :ivar llm_role: The role of the LLM responsible for extracting the concept
        ("extractor_text", "reasoner_text", "extractor_vision", "reasoner_vision").
        Defaults to "extractor_text".
    :vartype llm_role: LLMRoleAny
    :ivar add_justifications: Whether to include justifications for extracted items.
    :vartype add_justifications: bool
    :ivar justification_depth: Justification detail level. Defaults to "brief".
    :vartype justification_depth: JustificationDepth
    :ivar justification_max_sents: Maximum sentences in justification. Defaults to 2.
    :vartype justification_max_sents: int
    :ivar add_references: Whether to include source references for extracted items.
    :vartype add_references: bool
    :ivar reference_depth: Source reference granularity ("paragraphs" or "sentences").
        Defaults to "paragraphs". Only relevant when references are added to extracted items.
        Affects the structure of ``extracted_items``.
    :vartype reference_depth: ReferenceDepth
    :ivar singular_occurrence: Whether this concept is restricted to having only one extracted item.
        If True, only a single extracted item will be extracted. Defaults to False (multiple
        extracted items are allowed). Note that with advanced LLMs, this constraint may not be
        strictly required as they can often infer the appropriate number of items to extract
        from the concept's name, description, and type (e.g., "contract signing date" vs
        "meeting dates").
    :vartype singular_occurrence: StrictBool

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


class LabelConcept(_Concept):
    """
    A concept model for label-based classification of documents and aspects.

    This class handles identification and classification using predefined labels,
    supporting both multi-class (single label selection) and multi-label (multiple
    label selection) classification approaches.

    **Note**: When none of the predefined labels apply to the content being classified,
    no extracted items will be returned (empty ``extracted_items`` list). This ensures
    that only valid, predefined labels are selected and prevents forced classification
    when no appropriate label exists.

    :ivar name: The name of the concept (non-empty string, stripped).
    :vartype name: NonEmptyStr
    :ivar description: A brief description of the concept (non-empty string, stripped).
    :vartype description: NonEmptyStr
    :ivar labels: List of predefined labels for classification. Must contain at least 2 unique labels.
    :vartype labels: list[NonEmptyStr]
    :ivar classification_type: Classification mode - "multi_class" for single label selection,
        "multi_label" for multiple label selection. Defaults to "multi_class".
    :vartype classification_type: ClassificationType
    :ivar llm_role: The role of the LLM responsible for extracting the concept
        ("extractor_text", "reasoner_text", "extractor_vision", "reasoner_vision").
        Defaults to "extractor_text".
    :vartype llm_role: LLMRoleAny
    :ivar add_justifications: Whether to include justifications for extracted items.
    :vartype add_justifications: bool
    :ivar justification_depth: Justification detail level. Defaults to "brief".
    :vartype justification_depth: JustificationDepth
    :ivar justification_max_sents: Maximum sentences in justification. Defaults to 2.
    :vartype justification_max_sents: int
    :ivar add_references: Whether to include source references for extracted items.
    :vartype add_references: bool
    :ivar reference_depth: Source reference granularity ("paragraphs" or "sentences").
        Defaults to "paragraphs". Only relevant when references are added to extracted items.
        Affects the structure of ``extracted_items``.
    :vartype reference_depth: ReferenceDepth
    :ivar singular_occurrence: Whether this concept is restricted to having only one extracted item.
        If True, only a single extracted item will be extracted. Defaults to False (multiple
        extracted items are allowed). Note that with advanced LLMs, this constraint may not be
        strictly required as they can often infer the appropriate number of items to extract
        from the concept's name, description, and type (e.g., "document type" vs
        "content topics").
    :vartype singular_occurrence: StrictBool

    Example:
        .. literalinclude:: ../../../dev/usage_examples/docstrings/concepts/def_label_concept.py
            :language: python
            :caption: Label concept definition
    """

    labels: list[NonEmptyStr] = Field(..., min_length=2)
    classification_type: ClassificationType = Field(default="multi_class")

    _extracted_items: list[_LabelItem] = PrivateAttr(default_factory=list)

    @field_validator("labels")
    @classmethod
    def _validate_labels(cls, labels: list[NonEmptyStr]) -> list[NonEmptyStr]:
        """
        Validates that all labels are unique.

        :param labels: List of labels to validate.
        :type labels: list[NonEmptyStr]
        :return: The validated list of labels.
        :rtype: list[NonEmptyStr]
        :raises ValueError: If labels are not unique.
        """
        if len(labels) != len(set(labels)):
            raise ValueError("All labels must be unique.")
        return labels

    @property
    def _item_type_in_prompt(self) -> str:
        return _format_type(dict)

    @property
    def _item_class(self) -> type[_LabelItem]:
        return _LabelItem

    @property
    def _format_labels_in_prompt(self) -> str:
        """
        Formats the available labels for display in prompts.

        :return: Formatted string listing all available labels.
        :rtype: str
        """
        return "[" + ", ".join(f'"{label}"' for label in self.labels) + "]"

    @property
    def extracted_items(self) -> list[_LabelItem]:
        return self._extracted_items

    @extracted_items.setter
    def extracted_items(self, value: list[_LabelItem]) -> None:
        """
        Validates that all label values are from the predefined set and conform
        to classification type constraints, in addition to the validation
        performed by the parent class.

        :param value: The new list of extracted items to be set.
        :type value: list[_LabelItem]
        :raises ValueError: If any extracted item contains invalid labels or
            violates classification type constraints.
        :return: None
        """
        # First, perform label validation for each item
        if value:
            for item in value:
                self._validate_label_values(item.value)

        # Then, call the parent class setter for final validation and assignment
        super(LabelConcept, type(self)).extracted_items.fset(self, value)

    def _validate_label_values(self, labels_list: list[str]) -> None:
        """
        Validates label values against concept-specific business logic.

        Checks that all labels are from the predefined set and enforces
        classification type constraints.

        :param labels_list: List of label strings to validate.
        :type labels_list: list[str]
        :raises ValueError: If any label is not in the predefined set, or if
            classification type constraints are violated.
        """
        # Validate that all labels are from the predefined set
        invalid_labels = [label for label in labels_list if label not in self.labels]
        if invalid_labels:
            raise ValueError(
                f"Invalid labels found for concept `{self.__class__.__name__}`: "
                f"{invalid_labels}. Must be from predefined labels: {self.labels}"
            )

        # Validate classification type constraints
        if self.classification_type == "multi_class" and len(labels_list) > 1:
            raise ValueError(
                f"Multi-class classification allows only one label, got {len(labels_list)}: {labels_list}"
            )

    def _process_item_value(self, value: dict[str, list[str]]) -> list[str]:
        """
        Processes the object format of an extracted item's value, returned in the LLM response.
        Validates the value structure using a Pydantic model and then applies concept-specific
        business logic validation (labels from predefined set, classification type constraints).

        :param value: Dictionary with "labels" key containing list of label strings.
        :type value: dict[str, list[str]]
        :return: Validated list of labels for _LabelItem.value.
        :rtype: list[str]
        :raises ValueError: If the structure is invalid, any label is not in the predefined
            set, or if classification type constraints are violated.
        """

        # First, validate the extracted item's value structure using the Pydantic model
        _LabelConceptItemValueModel.model_validate(value)

        # Extract the labels list
        labels_list = value["labels"]

        # Apply concept-specific business logic validation
        self._validate_label_values(labels_list)

        # Return just the list of labels for _LabelItem.value
        return labels_list
