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

from contextgem.internal.base.concepts import (
    _BooleanConcept,
    _DateConcept,
    _JsonObjectConcept,
    _LabelConcept,
    _NumericalConcept,
    _RatingConcept,
    _StringConcept,
)
from contextgem.internal.decorators import _expose_in_registry


@_expose_in_registry(additional_key=_StringConcept)
class StringConcept(_StringConcept):
    """
    A concept model for string-based information extraction from documents and aspects.

    This class provides functionality for defining, extracting, and managing string data
    as conceptual entities within documents or aspects.

    :ivar name: The name of the concept (non-empty string, stripped).
    :vartype name: str
    :ivar description: A brief description of the concept (non-empty string, stripped).
    :vartype description: str
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

    pass


@_expose_in_registry(additional_key=_BooleanConcept)
class BooleanConcept(_BooleanConcept):
    """
    A concept model for boolean (True/False) information extraction from documents and aspects.

    This class handles identification and extraction of boolean values that represent
    conceptual properties or attributes within content.

    :ivar name: The name of the concept (non-empty string, stripped).
    :vartype name: str
    :ivar description: A brief description of the concept (non-empty string, stripped).
    :vartype description: str
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

    pass


@_expose_in_registry(additional_key=_NumericalConcept)
class NumericalConcept(_NumericalConcept):
    """
    A concept model for numerical information extraction from documents and aspects.

    This class handles identification and extraction of numeric values (integers, floats,
    or both) that represent conceptual measurements or quantities within content.

    :ivar name: The name of the concept (non-empty string, stripped).
    :vartype name: str
    :ivar description: A brief description of the concept (non-empty string, stripped).
    :vartype description: str
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

    pass


@_expose_in_registry(additional_key=_RatingConcept)
class RatingConcept(_RatingConcept):
    """
    A concept model for rating-based information extraction with defined scale boundaries.

    This class handles identification and extraction of integer ratings that must fall within
    the boundaries of a specified rating scale.

    :ivar name: The name of the concept (non-empty string, stripped).
    :vartype name: str
    :ivar description: A brief description of the concept (non-empty string, stripped).
    :vartype description: str
    :ivar rating_scale: The rating scale defining valid value boundaries. Can be either a RatingScale
        object (deprecated, will be removed in v1.0.0) or a tuple of (start, end) integers.
    :vartype rating_scale: RatingScale | tuple[int, int]
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

    pass


@_expose_in_registry(additional_key=_JsonObjectConcept)
class JsonObjectConcept(_JsonObjectConcept):
    """
    A concept model for structured JSON object extraction from documents and aspects.

    This class handles identification and extraction of structured data in JSON format,
    with validation against a predefined schema structure.

    :ivar name: The name of the concept (non-empty string, stripped).
    :vartype name: str
    :ivar description: A brief description of the concept (non-empty string, stripped).
    :vartype description: str
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
    :vartype structure: type | dict[str, Any]
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

    pass


@_expose_in_registry(additional_key=_DateConcept)
class DateConcept(_DateConcept):
    """
    A concept model for date object extraction from documents and aspects.

    This class handles identification and extraction of dates, with support for parsing
    string representations in a specified format into Python date objects.

    :ivar name: The name of the concept (non-empty string, stripped).
    :vartype name: str
    :ivar description: A brief description of the concept (non-empty string, stripped).
    :vartype description: str
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

    pass


@_expose_in_registry(additional_key=_LabelConcept)
class LabelConcept(_LabelConcept):
    """
    A concept model for label-based classification of documents and aspects.

    This class handles identification and classification using predefined labels,
    supporting both multi-class (single label selection) and multi-label (multiple
    label selection) classification approaches.

    **Note**: Behavior depends on ``classification_type``:

    - ``multi_class``: exactly one label is always returned for each extracted item. If
      none of the specific labels apply, include a catch-all label (e.g., ``"other"``,
      ``"N/A"``) among ``labels`` so the model can select it.
    - ``multi_label``: when none of the predefined labels apply, no extracted items may
      be returned (empty ``extracted_items`` list). This prevents forced classification
      when no appropriate label exists.

    :ivar name: The name of the concept (non-empty string, stripped).
    :vartype name: str
    :ivar description: A brief description of the concept (non-empty string, stripped).
    :vartype description: str
    :ivar labels: List of predefined labels (non-empty strings, stripped) for classification.
        Must contain at least 2 unique labels.
    :vartype labels: list[str]
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
    :vartype singular_occurrence: bool

    Example:
        .. literalinclude:: ../../../dev/usage_examples/docstrings/concepts/def_label_concept.py
            :language: python
            :caption: Label concept definition
    """

    pass
