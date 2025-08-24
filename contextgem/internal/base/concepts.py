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
Module defining the base classes for concept subclasses.
"""

from __future__ import annotations

import warnings
from abc import abstractmethod
from datetime import date, datetime
from types import UnionType
from typing import Any, List, Literal, Union, get_args, get_origin  # noqa: UP035

from pydantic import (
    BaseModel,
    Field,
    PrivateAttr,
    StrictBool,
    StrictInt,
    field_validator,
    model_validator,
)
from typing_extensions import Self

from contextgem.internal.base.attrs import _ExtractedItemsAttributeProcessor
from contextgem.internal.base.data_models import _RatingScale
from contextgem.internal.base.examples import _JsonObjectExample, _StringExample
from contextgem.internal.base.items import _ExtractedItem
from contextgem.internal.decorators import (
    _disable_direct_initialization,
    _post_init_method,
)
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
from contextgem.internal.loggers import logger
from contextgem.internal.typings.aliases import (
    ClassificationType,
    LLMRoleAny,
    NonEmptyStr,
    ReferenceDepth,
)
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


class _Concept(_ExtractedItemsAttributeProcessor):
    """
    Internal implementation of the Concept class.
    """

    name: NonEmptyStr = Field(
        ..., description="Concept name (required, non-empty string)."
    )
    description: NonEmptyStr = Field(
        ..., description="Concept description (required, non-empty string)."
    )
    llm_role: LLMRoleAny = Field(
        default="extractor_text",
        description=(
            "LLM role used for this concept. Valid values: 'extractor_text', 'reasoner_text', "
            "'extractor_vision', 'reasoner_vision', 'extractor_multimodal', 'reasoner_multimodal'."
        ),
    )
    add_references: StrictBool = Field(
        default=False,
        description="Whether to include references for extracted items.",
    )
    reference_depth: ReferenceDepth = Field(
        default="paragraphs",
        description=(
            "Reference granularity when references are included: 'paragraphs' or 'sentences'. "
            "Affects the structure of extracted_items."
        ),
    )
    singular_occurrence: StrictBool = Field(
        default=False,
        description="If True, restrict extraction to a single item.",
    )

    _extracted_items: list[_ExtractedItem] = PrivateAttr(default_factory=list)
    _is_processed: StrictBool = PrivateAttr(default=False)

    @property
    @abstractmethod
    def _item_type_in_prompt(self) -> str:
        """
        Abstract property, to be implemented by subclasses.

        The type of item as it should appear in prompts.
        """
        pass

    @property
    @abstractmethod
    def _item_class(self) -> type[_ExtractedItem]:
        """
        Abstract property, to be implemented by subclasses.

        The class used for extracted items of this concept.
        """
        pass

    @abstractmethod
    def _process_item_value(self, value: Any) -> Any:
        """
        Abstract method, to be implemented by subclasses.

        Process the item value with a custom function on the concept.
        """
        pass

    @classmethod
    def _validate_concept_extraction_params(cls, data: dict[str, Any]):
        """
        Validates the parameters used for concept extraction within a specific context of
        LLM roles.

        :param data: A dictionary containing parameters related to concept extraction.
        :type data: dict[str, Any]
        :raises ValueError: If the LLM role ends with "_vision" and `add_references`
            is set to True, as this combination is not supported.
        """
        llm_role = data.get("llm_role")
        add_references = data.get("add_references")
        if llm_role and llm_role.endswith("_vision") and add_references:
            raise ValueError("Vision concepts do not support references.")

    @model_validator(mode="before")
    @classmethod
    def _validate_concept_pre(cls, data: Any) -> Any:
        """
        Validates the concept's raw input data, which could be a dict with input values,
        an instance of the model, or another type depending on what is passed to the model.

        :param data: The input data to validate. It can be of any type.
        :return: The validated input data.
        """
        if isinstance(data, dict):
            cls._validate_concept_extraction_params(data)
        return data


@_disable_direct_initialization
class _StringConcept(_Concept):
    """
    Internal implementation of the StringConcept class.
    """

    examples: list[_StringExample] = Field(
        default_factory=list,
        description="Example strings illustrating how this concept should be extracted.",
    )

    _extracted_items: list[_StringItem] = PrivateAttr(default_factory=list)

    @property
    def _item_type_in_prompt(self) -> str:
        """
        Returns the formatted type string for use in LLM prompts.

        :return: Formatted string representation of the item type.
        :rtype: str
        """
        return _format_type(str)

    @property
    def _item_class(self) -> type[_StringItem]:
        """
        Returns the item class type for string concepts.

        :return: The string item class.
        :rtype: type[_StringItem]
        """
        return _StringItem

    def _process_item_value(self, value: str) -> str:
        """
        Processes and validates a string value for string concepts.

        :param value: The string value to process.
        :type value: str
        :return: The processed string value.
        :rtype: str
        """
        return value


@_disable_direct_initialization
class _BooleanConcept(_Concept):
    """
    Internal implementation of the BooleanConcept class.
    """

    _extracted_items: list[_BooleanItem] = PrivateAttr(default_factory=list)

    @property
    def _item_type_in_prompt(self) -> str:
        """
        Returns the formatted type string for use in LLM prompts.

        :return: Formatted string representation of the boolean type.
        :rtype: str
        """
        return _format_type(bool)

    @property
    def _item_class(self) -> type[_BooleanItem]:
        """
        Returns the item class type for boolean concepts.

        :return: The boolean item class.
        :rtype: type[_BooleanItem]
        """
        return _BooleanItem

    def _process_item_value(self, value: bool) -> bool:
        """
        Processes and validates a boolean value for boolean concepts.

        :param value: The boolean value to process.
        :type value: bool
        :return: The processed boolean value.
        :rtype: bool
        """
        return value


@_disable_direct_initialization
class _NumericalConcept(_Concept):
    """
    Internal implementation of the NumericalConcept class.
    """

    numeric_type: Literal["int", "float", "any"] = Field(
        default="any",
        description=(
            "Type constraint for extracted numbers: 'int', 'float', or 'any' (auto-detect)."
        ),
    )

    _extracted_items: list[_IntegerItem | _FloatItem | _IntegerOrFloatItem] = (
        PrivateAttr(default_factory=list)
    )

    @property
    def _item_type_in_prompt(self) -> str:
        """
        Returns the formatted type string for use in LLM prompts based on numeric type.

        :return: Formatted string representation of the numeric type.
        :rtype: str
        """
        if self.numeric_type == "int":
            return _format_type(int)
        elif self.numeric_type == "float":
            return _format_type(float)
        else:  # "any"
            return _format_type(int | float)

    @property
    def _item_class(self) -> type:
        """
        Returns the item class type for numerical concepts based on numeric type.

        :return: The appropriate numerical item class.
        :rtype: type
        """
        if self.numeric_type == "int":
            return _IntegerItem
        elif self.numeric_type == "float":
            return _FloatItem
        else:  # "any"
            return _IntegerOrFloatItem

    def _process_item_value(self, value: int | float) -> int | float:
        """
        Processes and validates a numerical value for numerical concepts.

        :param value: The numerical value to process.
        :type value: int | float
        :return: The processed numerical value.
        :rtype: int | float
        """
        return value


@_disable_direct_initialization
class _RatingConcept(_Concept):
    """
    Internal implementation of the RatingConcept class.
    """

    rating_scale: _RatingScale | tuple[StrictInt, StrictInt] = Field(  # type: ignore
        ...,
        description=(
            "Rating scale boundaries. Prefer a tuple of (start, end) integers; "
            "'RatingScale' is deprecated and will be removed in v1.0.0."
        ),
    )

    _extracted_items: list[_IntegerItem] = PrivateAttr(default_factory=list)

    @field_validator("rating_scale")
    @classmethod
    def _validate_rating_scale(
        cls,
        value: _RatingScale | tuple[int, int],  # type: ignore
    ) -> _RatingScale | tuple[int, int]:  # type: ignore
        """
        Validates the rating scale and issues deprecation warning for _RatingScale.

        :param value: The rating scale value to validate.
        :type value: _RatingScale | tuple[int, int]
        :return: The validated rating scale.
        :rtype: _RatingScale | tuple[int, int]
        :raises ValueError: If the rating scale is invalid.
        """
        if isinstance(value, _RatingScale):
            warnings.warn(
                "RatingScale is deprecated and will be removed in v1.0.0. "
                "Use a tuple of (start, end) integers instead, e.g. (1, 5) "
                "instead of RatingScale(start=1, end=5).",
                DeprecationWarning,
                stacklevel=2,
            )
            return value
        elif isinstance(value, tuple):
            if len(value) != 2:
                raise ValueError(
                    "Rating scale tuple must contain exactly 2 elements: (start, end)"
                )
            start, end = value
            if not isinstance(start, int) or not isinstance(end, int):
                raise ValueError("Rating scale tuple elements must be integers")
            if start < 0:
                raise ValueError("Rating scale start value must be >= 0")
            if end <= 0:
                raise ValueError("Rating scale end value must be > 0")
            if end <= start:
                raise ValueError(
                    f"Rating scale end value ({end}) must be greater than start value ({start})"
                )
            return value
        else:
            raise ValueError(
                "Rating scale must be either a RatingScale object or a tuple of (start, end) integers"
            )

    @property
    def _item_type_in_prompt(self) -> str:
        """
        Returns the formatted type string for use in LLM prompts.

        :return: Formatted string representation of the integer type.
        :rtype: str
        """
        return _format_type(int)

    @property
    def _item_class(self) -> type[_IntegerItem]:
        """
        Returns the item class type for rating concepts.

        :return: The integer item class.
        :rtype: type[_IntegerItem]
        """
        return _IntegerItem

    @property
    def _rating_start(self) -> int:
        """
        Gets the start value of the rating scale.
        """
        if isinstance(self.rating_scale, _RatingScale):
            return self.rating_scale.start
        return self.rating_scale[0]

    @property
    def _rating_end(self) -> int:
        """
        Gets the end value of the rating scale.
        """
        if isinstance(self.rating_scale, _RatingScale):
            return self.rating_scale.end
        return self.rating_scale[1]

    @property
    def extracted_items(self) -> list[_IntegerItem]:
        """
        Gets the list of extracted rating items.

        :return: List of extracted integer items representing ratings.
        :rtype: list[_IntegerItem]
        """
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
        # Need to call parent setter explicitly after custom validation -
        # .fset() pattern required when overriding property setter
        super(_RatingConcept, type(self)).extracted_items.fset(self, value)  # type: ignore[attr-defined]

    def _validate_rating_value(self, rating_value: int) -> None:
        """
        Validates rating values against concept-specific business logic.

        Checks that the rating value is within the defined rating scale range.

        :param rating_value: Rating value to validate.
        :type rating_value: int
        :raises ValueError: If the rating value is outside the allowed rating scale range.
        """
        if not self._rating_start <= rating_value <= self._rating_end:
            raise ValueError(
                f"Invalid value for scaled rating concept: "
                f"value {rating_value} is outside of the scale range "
                f"{self._rating_start} to {self._rating_end}"
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


@_disable_direct_initialization
class _JsonObjectConcept(_Concept):
    """
    Internal implementation of the JsonObjectConcept class.
    """

    structure: type | dict[NonEmptyStr, Any] = Field(
        ...,
        description=(
            "JSON object schema as a type-annotated class or a dictionary where keys are field "
            "names and values are type annotations. Supports unions, Literals, nested objects, "
            "lists (e.g., [str] or list[str]), and typed classes (e.g., Pydantic models, "
            "dataclasses) for nested structures. Class types are not allowed as dictionary keys "
            "or values."
        ),
    )
    examples: list[_JsonObjectExample] = Field(
        default_factory=list,
        description="Example JSON objects illustrating how this concept should be extracted.",
    )

    _extracted_items: list[_JsonObjectItem] = PrivateAttr(default_factory=list)

    @property
    def _item_type_in_prompt(self) -> str:
        """
        Returns the formatted type string for use in LLM prompts.

        :return: Formatted string representation of the dictionary type.
        :rtype: str
        """
        return _format_type(dict)

    @property
    def _item_class(self) -> type[_JsonObjectItem]:
        """
        Returns the item class type for JSON object concepts.

        :return: The JSON object item class.
        :rtype: type[_JsonObjectItem]
        """
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
        # Structure is validated and normalized in field validator, safe to pass to dynamic model creation
        return _dynamic_pydantic_model(self.structure)  # type: ignore[arg-type]

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
            List,  # noqa: UP006
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
                            raise ValueError(str(e)) from e

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


@_disable_direct_initialization
class _DateConcept(_Concept):
    """
    Internal implementation of the DateConcept class.
    """

    _extracted_items: list[_DateItem] = PrivateAttr(default_factory=list)

    @property
    def _date_format_in_prompt(self) -> str:
        """
        Returns the date format string used in LLM prompts.

        :return: Date format string for prompt instructions.
        :rtype: str
        """
        return "DD-MM-YYYY"

    @property
    def _item_type_in_prompt(self) -> str:
        """
        Returns the formatted type string for use in LLM prompts with date format.

        :return: Formatted string representation of the string type with date format.
        :rtype: str
        """
        return _format_type(str) + f" (in '{self._date_format_in_prompt}' format)"

    @property
    def _item_class(self) -> type[_DateItem]:
        """
        Returns the item class type for date concepts.

        :return: The date item class.
        :rtype: type[_DateItem]
        """
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


@_disable_direct_initialization
class _LabelConcept(_Concept):
    """
    Internal implementation of the LabelConcept class.
    """

    labels: list[NonEmptyStr] = Field(
        ...,
        min_length=2,
        description=(
            "Predefined labels for classification. Must contain at least 2 labels and all labels must be unique."
        ),
    )
    classification_type: ClassificationType = Field(
        default="multi_class",
        description=(
            "Classification mode: 'multi_class' for a single selected label, or 'multi_label' for multiple labels."
        ),
    )

    _extracted_items: list[_LabelItem] = PrivateAttr(default_factory=list)

    @_post_init_method
    def _post_init(self, __context: Any):
        """
        Post-initialization method that provides guidance for multi-class classification.

        :param __context: Pydantic context (unused).
        :type __context: Any
        """
        if self.classification_type == "multi_class":
            logger.info(
                f"For multi-class classification in concept '{self.name}', you should consider including "
                f"a general 'other' (or 'N/A', 'misc', etc.) label to handle cases where none "
                f"of the specific labels apply, unless you already have such a label, your labels are "
                f"broad enough to cover all cases, or you know that the classified content always falls "
                f"under one of the predefined labels without edge cases. Multi-class classification "
                f"should always return a label, so having a catch-all option ensures appropriate "
                f"classification when no specific label fits the content."
            )

    @field_validator("labels")
    @classmethod
    def _validate_labels(cls, labels: list[NonEmptyStr]) -> list[NonEmptyStr]:
        """
        Validates that all labels are unique (case-insensitive).

        :param labels: List of labels to validate.
        :type labels: list[str]
        :return: The validated list of labels.
        :rtype: list[str]
        :raises ValueError: If labels are not unique (case-insensitive).
        """
        if len(set(label.lower() for label in labels)) < len(labels):
            raise ValueError("All labels must be unique (case-insensitive).")
        return labels

    @property
    def _item_type_in_prompt(self) -> str:
        """
        Returns the formatted type string for use in LLM prompts.

        :return: Formatted string representation of the dictionary type.
        :rtype: str
        """
        return _format_type(dict)

    @property
    def _item_class(self) -> type[_LabelItem]:
        """
        Returns the item class type for label concepts.

        :return: The label item class.
        :rtype: type[_LabelItem]
        """
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
        """
        Gets the list of extracted label items.

        :return: List of extracted label items.
        :rtype: list[_LabelItem]
        """
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
        # Need to call parent setter explicitly after custom validation -
        # .fset() pattern required when overriding property setter
        super(_LabelConcept, type(self)).extracted_items.fset(self, value)  # type: ignore[attr-defined]

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
