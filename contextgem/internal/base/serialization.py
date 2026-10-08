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
Module providing serialization and deserialization functionality for document instances.

This module defines constants, classes, and utilities for converting document objects
(such as aspects, concepts, examples, and other extracted items) between their native
Python representation and serialized formats like dictionaries and JSON. It handles
the preservation of object relationships and special attributes during the serialization
and deserialization process.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from contextvars import ContextVar
from copy import deepcopy
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Any

from aiolimiter import AsyncLimiter
from pydantic import BaseModel
from typing_extensions import Self

from contextgem.internal.loggers import logger
from contextgem.internal.registry import _resolve_public_type
from contextgem.internal.typings.strings_to_types import _deserialize_type_hint
from contextgem.internal.typings.types_normalization import _normalize_type_annotation
from contextgem.internal.typings.types_to_strings import _serialize_type_hint


if TYPE_CHECKING:
    from contextgem.internal.data_models import _ModelCost


# Public attrs
KEY_ASPECTS_PUBLIC = "aspects"
KEY_CONCEPTS_PUBLIC = "concepts"
KEY_STRUCTURE_PUBLIC = "structure"
KEY_RATING_SCALE_PUBLIC = "rating_scale"
KEY_EXAMPLES_PUBLIC = "examples"
KEY_IMAGES_PUBLIC = "images"
KEY_PARAGRAPHS_PUBLIC = "paragraphs"
KEY_SENTENCES_PUBLIC = "sentences"
KEY_PROBABILITIES_PUBLIC = "probabilities"
# LLM attrs
KEY_API_KEY_PUBLIC = "api_key"  # always redacted
KEY_API_BASE_PUBLIC = "api_base"  # always redacted
KEY_FALLBACK_MODELS_PUBLIC = "fallback_models"
KEY_MODEL_PRICING_PUBLIC = "pricing_details"
KEY_MODELS_PUBLIC = "models"
KEY_TOOLS_PUBLIC = "tools"

# Private attrs
KEY_UNIQUE_ID_PRIVATE = "_unique_id"
KEY_EXTRACTED_ITEMS_PRIVATE = "_extracted_items"
KEY_REFERENCE_PARAGRAPHS_PRIVATE = "_reference_paragraphs"
KEY_REFERENCE_SENTENCES_PRIVATE = "_reference_sentences"
KEY_IS_PROCESSED_PRIVATE = "_is_processed"
KEY_NESTING_LEVEL_PRIVATE = "_nesting_level"
KEY_CLASS_PRIVATE = "__class__"
# LLM attrs
KEY_ASYNC_LIMITER_PRIVATE = "_async_limiter"
KEY_MODEL_USAGE_PRIVATE = "_usage"
KEY_MODEL_COST_PRIVATE = "_cost"
KEY_LLM_SUPPORTS_VISION_PRIVATE = "_supports_vision"
KEY_LLM_SUPPORTS_REASONING_PRIVATE = "_supports_reasoning"
KEY_LLM_SUPPORTS_TOOLS_PRIVATE = "_supports_tools"
KEY_LLM_SUPPORTS_PARALLEL_TOOL_CALLS_PRIVATE = "_supports_parallel_tool_calls"
KEY_PRIVATE_ATTRS_INITIALIZED_PRIVATE = "_private_attrs_initialized"
# Chat message attrs
KEY_TIME_CREATED_PRIVATE = "_time_created"
KEY_RESPONSE_SUCCEEDED_PRIVATE = "_response_succeeded"
# Chat session attrs
KEY_MESSAGES_PRIVATE = "_messages"


# Private attrs whose validation lives on a public property of the same name
# (without the leading underscore). During deserialization these are assigned
# through the public property setter so that its validation (e.g. element-type
# and uniqueness checks) runs, instead of writing the backing attribute directly
# and silently bypassing those invariants.
_PRIVATE_ATTR_PUBLIC_PROPERTY: dict[str, str] = {
    KEY_EXTRACTED_ITEMS_PRIVATE: "extracted_items",
    KEY_REFERENCE_PARAGRAPHS_PRIVATE: "reference_paragraphs",
    KEY_REFERENCE_SENTENCES_PRIVATE: "reference_sentences",
}

# Set while an instance is being created from its serialized data, for the
# validation that cannot be applied to such data (see the function below)
_DESERIALIZATION_IN_PROGRESS: ContextVar[bool] = ContextVar(
    "_DESERIALIZATION_IN_PROGRESS", default=False
)


def _is_deserialization_in_progress() -> bool:
    """
    Checks whether an instance is currently being created from its serialized data.

    Serialized data is the data of an instance that was valid when it was serialized,
    but not all of that data is serialized, e.g. the API credentials of an LLM are
    removed. The validation that relies on the removed data gives a wrong result
    for the serialized data, and is to be skipped while such data is deserialized.

    :return: True if called while an instance is being deserialized, False otherwise.
    :rtype: bool
    """
    return _DESERIALIZATION_IN_PROGRESS.get()


class _InstanceSerializer(BaseModel):
    """
    Base class that provides reusable methods for serialization and deserialization of instances.

    This class serves as a foundation for converting document objects between their native Python
    representation and serialized formats like dictionaries and JSON. It implements common
    serialization and deserialization logic that can be inherited by various document components
    and LLMs.
    """

    def to_dict(self) -> dict[str, Any]:
        """
        Transforms the current object into a dictionary representation.

        Converts the object to a dictionary that includes:
        - All public attributes
        - Special handling for specific public and private attributes

        When an LLM or model group is serialized, its API credentials and usage/cost stats are removed.

        :return: A dictionary representation of the current object with all necessary data for serialization
        :rtype: dict[str, Any]
        """

        from contextgem.internal.base.llms import _DocumentLLM
        from contextgem.internal.base.models import _DocumentModelGroup
        from contextgem.internal.data_models import _ModelCost, _ModelUsage

        if isinstance(self, _DocumentLLM | _DocumentModelGroup):
            logger.info(
                "API credentials and usage/cost stats are removed from the serialized LLM/model group."
            )

        # Start with normal public fields serialization
        base_dict = super().model_dump()

        # Transform relevant public attributes
        for key in base_dict:
            val = getattr(self, key)

            if key in [
                KEY_ASPECTS_PUBLIC,
                KEY_CONCEPTS_PUBLIC,
                KEY_EXAMPLES_PUBLIC,
                KEY_IMAGES_PUBLIC,
                KEY_PARAGRAPHS_PUBLIC,
                KEY_SENTENCES_PUBLIC,
                KEY_MODELS_PUBLIC,
                KEY_FALLBACK_MODELS_PUBLIC,
            ]:
                base_dict[key] = [i.to_dict() for i in val]

            elif key == KEY_STRUCTURE_PUBLIC:
                # Handle structure serialization for JsonObjectConcept structure
                base_dict[key] = self._serialize_structure_dict(val)

            elif key == KEY_RATING_SCALE_PUBLIC:
                # Convert the tuple to a list for JSON serialization
                base_dict[key] = list(val)

            elif key == KEY_PROBABILITIES_PUBLIC:
                # Convert the dict of an extracted item to a list of pairs for
                # JSON serialization, only when provided
                base_dict[key] = (
                    self._serialize_probabilities(val) if val is not None else val
                )

            elif key == KEY_MODEL_PRICING_PUBLIC:
                # Serialize only when provided
                base_dict[key] = val.to_dict() if val is not None else val

            elif key in [KEY_API_KEY_PUBLIC, KEY_API_BASE_PUBLIC]:
                # Reset API credentials when LLM is serialized
                base_dict[key] = None

            elif key == KEY_TOOLS_PUBLIC and isinstance(self, _DocumentLLM):
                # The functions passed as tools cannot be serialized: the tool
                # definitions generated from them are stored instead. When the LLM
                # is deserialized, the functions are found by the names of the tools.
                base_dict[key] = (
                    [
                        registration["definition"]
                        for registration in self._tool_registry.values()
                    ]
                    if val is not None
                    else None
                )

        # Include relevant private attributes
        for key in self.__private_attributes__:
            val = getattr(self, key)

            if key in [
                KEY_EXTRACTED_ITEMS_PRIVATE,
                KEY_REFERENCE_PARAGRAPHS_PRIVATE,
                KEY_REFERENCE_SENTENCES_PRIVATE,
                KEY_MESSAGES_PRIVATE,
            ]:
                base_dict[key] = [i.to_dict() for i in val]

            elif key in [
                KEY_UNIQUE_ID_PRIVATE,
                KEY_IS_PROCESSED_PRIVATE,
                KEY_NESTING_LEVEL_PRIVATE,
                KEY_LLM_SUPPORTS_VISION_PRIVATE,
                KEY_LLM_SUPPORTS_REASONING_PRIVATE,
                KEY_LLM_SUPPORTS_TOOLS_PRIVATE,
                KEY_LLM_SUPPORTS_PARALLEL_TOOL_CALLS_PRIVATE,
                KEY_PRIVATE_ATTRS_INITIALIZED_PRIVATE,
            ]:
                base_dict[key] = val

            elif key == KEY_ASYNC_LIMITER_PRIVATE:
                # Store only the limiter config, not the limiter object
                base_dict[key] = {
                    "max_rate": val.max_rate,
                    "time_period": val.time_period,
                }

            elif key == KEY_MODEL_USAGE_PRIVATE:
                # Reset usage stats when LLM is serialized
                base_dict[key] = _ModelUsage().to_dict()

            elif key == KEY_MODEL_COST_PRIVATE:
                # Reset cost stats when LLM is serialized
                cost_dict = _ModelCost().to_dict()
                # Convert Decimal objects to floats in the cost dictionary
                base_dict[key] = self._convert_decimal_to_float(cost_dict)

            elif key == KEY_TIME_CREATED_PRIVATE:
                # Relevant for _Message instances.
                # Convert time created to ISO format for serialization
                base_dict[key] = val.isoformat()

            elif key == KEY_RESPONSE_SUCCEEDED_PRIVATE:
                # Relevant for _Message instances.
                # Only serialize the value for user messages.
                # For messages with other roles, value cannot be set
                # (default value None will be set on instance creation).
                if base_dict["role"] == "user":
                    base_dict[key] = val
                else:
                    pass

        # Add class name for deserialization
        base_dict[KEY_CLASS_PRIVATE] = self.__class__.__name__

        return {**base_dict}

    def _serialize_structure_dict(
        self, structure_dict: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Relevant for JsonObjectConcept structure serialization.

        Recursively serializes a dictionary containing type hints to ensure proper serialization.
        Handles nested dictionaries, lists of dictionaries, and various type hints.

        :param structure_dict: Dictionary containing type hints to serialize
        :type structure_dict: dict[str, Any]
        :return: Dictionary with serialized type hints
        :rtype: dict[str, Any]
        """
        result = {}
        for key, value in structure_dict.items():
            # Normalize the value for consistent type representation
            value = _normalize_type_annotation(value)

            # Handle nested dictionaries
            if isinstance(value, dict):
                # Class structs (if passed) are already converted to a dict structure
                # during JsonObjectConcept initialization.
                result[key] = self._serialize_structure_dict(value)
            # Handle list of dictionaries (only need to serialize the first item)
            elif (
                isinstance(value, list)
                and len(value) == 1
                and isinstance(value[0], dict)
            ):
                # Class structs (if passed) are already converted to a dict structure
                # during JsonObjectConcept initialization.
                result[key] = [self._serialize_structure_dict(value[0])]
            # Other cases
            else:
                # Safe cast: After normalization, value should be a type hint
                # Type checker needs help to recognize this is a valid type hint format
                result[key] = _serialize_type_hint(value)

        return result

    @staticmethod
    def _serialize_probabilities(
        probabilities: dict[bool | int | str, float],
    ) -> list[list[bool | int | str | float]]:
        """
        Relevant for the serialization of the probabilities of an extracted item.

        Converts the probabilities of the options to a list of [option, probability]
        pairs, in the order of the options. The keys of a JSON object can only be
        strings, i.e. as such keys the options ``True``, ``1`` and ``"1"`` could not
        be told apart, while as the elements of a list they keep their types.

        :param probabilities: Options mapped to their probabilities.
        :type probabilities: dict[bool | int | str, float]
        :return: List of [option, probability] pairs.
        :rtype: list[list[bool | int | str | float]]
        """
        return [[option, probability] for option, probability in probabilities.items()]

    @staticmethod
    def _deserialize_probabilities(pairs: Any) -> dict[bool | int | str, float]:
        """
        Relevant for the deserialization of the probabilities of an extracted item.

        Converts a list of [option, probability] pairs back to the options mapped
        to their probabilities, with the types and the order of the options kept.
        The probabilities are validated by the item that is created from them.

        :param pairs: List of [option, probability] pairs.
        :type pairs: Any
        :return: Options mapped to their probabilities.
        :rtype: dict[bool | int | str, float]
        :raises ValueError: If the data is not a list of [option, probability] pairs,
            if an option is not a boolean, an integer or a string, or if an option
            is repeated.
        """
        if not isinstance(pairs, list) or not all(
            isinstance(pair, list) and len(pair) == 2 for pair in pairs
        ):
            raise ValueError(
                "Serialized probabilities must be a list of [option, probability] "
                f"pairs. Got: {pairs!r}"
            )
        probabilities: dict[bool | int | str, float] = {}
        for option, probability in pairs:
            if not isinstance(option, bool | int | str):
                raise ValueError(
                    "An option in serialized probabilities must be a boolean, "
                    f"an integer or a string. Got: {option!r}"
                )
            # The options that are equal as dict keys, e.g. `True` and `1`,
            # cannot be restored as separate options
            if option in probabilities:
                raise ValueError(
                    f"Serialized probabilities repeat the option {option!r}."
                )
            probabilities[option] = probability
        return probabilities

    def _convert_decimal_to_float(self, obj: Any) -> Any:
        """
        Recursively converts Decimal objects to floats for JSON serialization.

        :param obj: The object to convert.
        :type obj: Any
        :return: The converted object.
        :rtype: Any
        """
        if isinstance(obj, Decimal):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: self._convert_decimal_to_float(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_decimal_to_float(i) for i in obj]
        return obj

    def to_json(self) -> str:
        """
        Converts the object to its JSON string representation.

        Serializes the object into a JSON-formatted string using the dictionary
        representation provided by the `to_dict()` method.

        :return: A JSON string representation of the object.
        :rtype: str
        """
        return json.dumps(self.to_dict(), ensure_ascii=False)

    def to_disk(self, file_path: str | Path) -> None:
        """
        Saves the serialized instance to a JSON file at the specified path.

        This method converts the instance to a dictionary representation using `to_dict()`,
        then writes it to disk as a formatted JSON file with UTF-8 encoding.

        :param file_path: Path where the JSON file should be saved (must end with '.json').
            Can be a string or a Path object.
        :type file_path: str | Path
        :return: None
        :raises ValueError: If the file path doesn't end with '.json'.
        :raises RuntimeError: If there's an error during the file writing process.
        """
        # Convert to Path for consistent handling
        path_obj = Path(file_path)
        if path_obj.suffix.lower() != ".json":
            raise ValueError("The file path must end with '.json'")
        try:
            # Dump the JSON representation
            data = self.to_dict()
            with open(path_obj, "w", encoding="utf-8") as file:
                json.dump(data, file, ensure_ascii=False, indent=2)
        except Exception as e:
            raise RuntimeError(f"Failed to save the instance to {path_obj}: {e}") from e

    @classmethod
    def from_disk(cls, file_path: str | Path) -> Self:
        """
        Loads an instance of the class from a JSON file stored on disk.

        This method reads the JSON content from the specified file path and
        deserializes it into an instance of the class using the `from_json`
        method.

        :param file_path: Path to the JSON file to load (must end with '.json').
            Can be a string or a Path object.
        :type file_path: str | Path
        :return: An instance of the class populated with the data from the file.
        :rtype: Self
        :raises ValueError: If the file path doesn't end with '.json'.
        :raises RuntimeError: If deserialization fails.
        """
        # Convert to Path for consistent handling
        path_obj = Path(file_path)
        if path_obj.suffix.lower() != ".json":
            raise ValueError("The file path must end with '.json'")
        try:
            with open(path_obj, encoding="utf-8") as file:
                # We do not use json.load() here as we need to transform specific attributes,
                # which are serialized in the JSON string, by using cls.from_json().
                json_data = file.read()
            # Deserialize the JSON content into an instance
            return cls.from_json(json_data)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load the instance from {path_obj}: {e}"
            ) from e

    @classmethod
    def from_dict(cls, obj_dict: dict[str, Any]) -> Self:
        """
        Reconstructs an instance of the class from a dictionary representation.

        This method deserializes a dictionary containing the object's attributes and values
        into a new instance of the class. It handles complex nested structures like aspects,
        concepts, and extracted items, properly reconstructing each component.

        A deserialized LLM or model group has no API credentials, as they are not
        included in the serialized data, i.e. they are to be set again.

        :param obj_dict: Dictionary containing the serialized object data.
        :type obj_dict: dict[str, Any]
        :return: A new instance of the class with restored attributes.
        :rtype: Self
        """

        from contextgem.internal.base.data_models import _ModelPricing
        from contextgem.internal.data_models import _ModelUsage

        # Create a copy of the object dict due to further modification
        obj_dict = deepcopy(obj_dict)

        object_class_name = obj_dict.get(KEY_CLASS_PRIVATE)
        del obj_dict[KEY_CLASS_PRIVATE]
        if object_class_name != cls.__name__:
            raise TypeError(f"Class {object_class_name} does not match {cls.__name__}")

        def reconstruct_entity_from_dict(
            entity_d: dict[str, Any],
        ) -> Any:
            """
            Reconstructs an entity object from its dictionary representation.

            :param entity_d: Dictionary containing the serialized entity data.
            :type entity_d: dict[str, Any]
            :return: Reconstructed entity object.
            :rtype: Any
            :raises ValueError: If class name is missing from the dictionary.
            :raises RuntimeError: If class name is not a string or class not found.
            """
            class_name = entity_d.get(KEY_CLASS_PRIVATE)
            if class_name is None:
                raise ValueError("Missing class name in serialized dictionary")
            if not isinstance(class_name, str):
                raise TypeError(
                    f"Class name must be a string, got {type(class_name).__name__}"
                )
            entity_class = _resolve_public_type(class_name)
            if entity_class is None:
                raise TypeError(f"{class_name} not found.")
            return entity_class.from_dict(entity_d)

        def lambda_list_val() -> Callable[[Any], Any]:
            """
            Creates a lambda function to reconstruct a list of objects from dictionaries.

            :return: Lambda function that reconstructs objects from a list of dictionaries.
            :rtype: Callable[[Any], Any]
            """
            return lambda val: [reconstruct_entity_from_dict(d) for d in val]

        def _deserialize_structure_dict(
            structure_dict: dict[str, Any],
        ) -> dict[str, Any]:
            """
            Relevant for JsonObjectConcept structure deserialization.

            Recursively deserializes a dictionary containing string representations of type hints
            into actual Python type objects. Handles nested dictionaries, lists of dictionaries,
            and various type hint formats.

            :param structure_dict: Dictionary containing serialized type hints to deserialize
            :type structure_dict: dict[str, Any]
            :return: Dictionary with deserialized type hints
            :rtype: dict[str, Any]
            """

            result = {}
            for k, v in structure_dict.items():
                # Class structs (if passed) are already converted to a dict structure
                # during JsonObjectConcept initialization.
                if isinstance(v, dict):
                    result[k] = _deserialize_structure_dict(v)
                elif isinstance(v, list) and len(v) == 1 and isinstance(v[0], dict):
                    result[k] = [_deserialize_structure_dict(v[0])]
                elif isinstance(v, str):
                    try:
                        # Deserialize the type hint
                        type_hint = _deserialize_type_hint(v)

                        # Normalize the type hint for consistent representation
                        # This converts between typing module generics and built-in equivalents
                        normalized_type = _normalize_type_annotation(type_hint)

                        result[k] = normalized_type
                    except ValueError:
                        # Keep as string if can't deserialize
                        result[k] = v
                else:
                    result[k] = v
            return result

        # Create a map for known keys → reconstruction logic
        rebuild_map: dict[str, Callable[[Any], Any]] = {
            # Public attrs
            KEY_ASPECTS_PUBLIC: lambda_list_val(),
            KEY_CONCEPTS_PUBLIC: lambda_list_val(),
            KEY_EXAMPLES_PUBLIC: lambda_list_val(),
            KEY_PARAGRAPHS_PUBLIC: lambda_list_val(),
            KEY_SENTENCES_PUBLIC: lambda_list_val(),
            KEY_IMAGES_PUBLIC: lambda_list_val(),
            KEY_STRUCTURE_PUBLIC: lambda val: (
                # JsonObjectConcept structure is always converted to a dict
                _deserialize_structure_dict(val)
            ),
            KEY_RATING_SCALE_PUBLIC: lambda val: tuple(val),
            KEY_PROBABILITIES_PUBLIC: lambda val: (
                cls._deserialize_probabilities(val) if val is not None else val
            ),
            # LLM attrs
            KEY_MODEL_PRICING_PUBLIC: lambda val: (
                _resolve_public_type(_ModelPricing).from_dict(val)
                if val is not None
                else val
            ),
            KEY_FALLBACK_MODELS_PUBLIC: lambda_list_val(),
            KEY_MODELS_PUBLIC: lambda_list_val(),
            # Private attrs
            KEY_EXTRACTED_ITEMS_PRIVATE: lambda_list_val(),
            KEY_REFERENCE_PARAGRAPHS_PRIVATE: lambda_list_val(),
            KEY_REFERENCE_SENTENCES_PRIVATE: lambda_list_val(),
            # LLM attrs
            KEY_MODEL_USAGE_PRIVATE: lambda val: _ModelUsage.from_dict(val),
            KEY_MODEL_COST_PRIVATE: lambda val: cls._convert_model_cost_dict(val),
            KEY_ASYNC_LIMITER_PRIVATE: lambda val: AsyncLimiter(
                max_rate=val["max_rate"], time_period=val["time_period"]
            ),
            # Chat message attrs
            KEY_TIME_CREATED_PRIVATE: lambda val: datetime.fromisoformat(val),
            # Chat session attrs
            KEY_MESSAGES_PRIVATE: lambda_list_val(),
        }

        constructor_kwargs: dict[str, Any] = {}
        private_attrs: dict[str, Any] = {}

        for k, v in obj_dict.items():
            final_val = rebuild_map[k](v) if k in rebuild_map else v
            # If it's a private attr, collect for assignment separately
            if k.startswith("_"):
                private_attrs[k] = final_val
            else:
                constructor_kwargs[k] = final_val

        # Let the validation know that the instance is created from serialized data
        token = _DESERIALIZATION_IN_PROGRESS.set(True)
        try:
            new_instance = cls(**constructor_kwargs)
        finally:
            _DESERIALIZATION_IN_PROGRESS.reset(token)

        # Set private attrs separately. Where a private attr is exposed through a
        # public property carrying validation, assign via that property so the
        # validation runs (instead of writing the backing attr directly and
        # bypassing it); otherwise set the private attr directly.
        for priv_k, priv_v in private_attrs.items():
            public_name = _PRIVATE_ATTR_PUBLIC_PROPERTY.get(priv_k)
            if public_name is not None and isinstance(
                getattr(type(new_instance), public_name, None), property
            ):
                setattr(new_instance, public_name, priv_v)
            else:
                setattr(new_instance, priv_k, priv_v)

        return new_instance

    @classmethod
    def _convert_model_cost_dict(cls, cost_dict: dict[str, Any]) -> _ModelCost:
        """
        Converts a dictionary containing _ModelCost data to an _ModelCost instance,
        ensuring float values are converted to Decimal.

        :param cost_dict: Dictionary containing _ModelCost data
        :type cost_dict: dict[str, Any]
        :return: An _ModelCost instance
        :rtype: _ModelCost
        """
        from contextgem.internal.data_models import _ModelCost

        # Convert float values to Decimal
        cost_dict["input"] = Decimal(str(cost_dict["input"]))
        cost_dict["output"] = Decimal(str(cost_dict["output"]))
        cost_dict["total"] = Decimal(str(cost_dict["total"]))

        return _ModelCost.from_dict(cost_dict)

    @classmethod
    def from_json(cls, json_string: str) -> Self:
        """
        Creates an instance of the class from a JSON string representation.

        This method deserializes the provided JSON string into a dictionary and uses
        the `from_dict` method to construct the class instance. It validates that the
        class name in the serialized data matches the current class.

        :param json_string: JSON string containing the serialized object data.
        :type json_string: str
        :return: A new instance of the class with restored state.
        :rtype: Self
        :raises TypeError: If the class name in the serialized data doesn't match.
        """
        obj_dict = json.loads(json_string)
        if obj_dict[KEY_CLASS_PRIVATE] != cls.__name__:
            raise TypeError(
                f"Class {obj_dict[KEY_CLASS_PRIVATE]} does not match {cls.__name__}"
            )
        return cls.from_dict(obj_dict)

    def model_dump(self, *args, **kwargs):
        """
        Raises NotImplementedError to redirect users to use to_dict() method instead.

        This method is intentionally disabled in favor of the to_dict() method which
        provides proper serialization logic for ContextGem objects.

        :param args: Positional arguments (ignored).
        :param kwargs: Keyword arguments (ignored).
        :raises NotImplementedError: Always raised to direct users to use to_dict().
        """
        raise NotImplementedError("Use `to_dict()` instead")

    def model_dump_json(self, *args, **kwargs):
        """
        Raises NotImplementedError to redirect users to use to_json() method instead.

        This method is intentionally disabled in favor of the to_json() method which
        provides proper JSON serialization logic for ContextGem objects.

        :param args: Positional arguments (ignored).
        :param kwargs: Keyword arguments (ignored).
        :raises NotImplementedError: Always raised to direct users to use to_json().
        """
        raise NotImplementedError("Use `to_json()` instead")
