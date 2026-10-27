# SPDX-FileCopyrightText: Shcherbak AI AS
# SPDX-License-Identifier: Apache-2.0

"""
Module for handling concepts at aspect and document levels.

This module provides classes for defining different types of concepts that can be
extracted from documents and aspects. Concepts represent specific pieces of information
to be identified and extracted by models, such as strings, numbers, boolean values,
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
    :ivar model_role: The role label of the model that extracts the concept when
        a ``DocumentModelGroup`` is used: the concept is routed to the model in the group
        with the same ``role``. Defaults to None, which routes the concept to the model
        without a role. Ignored by a standalone model, which extracts every
        concept it is given.
    :vartype model_role: str | None
    :ivar modality: The document content the concept is extracted from: "text", "image",
        or "multimodal" (text and/or images, whichever the document has). "image" and
        "multimodal" are supported for document-level concepts only. Defaults to "text".
    :vartype modality: Modality
    :ivar add_justifications: Whether to include justifications for extracted items.
    :vartype add_justifications: bool
    :ivar justification_depth: Justification detail level. Defaults to "brief".
        Concepts with different values are extracted in separate calls.
    :vartype justification_depth: JustificationDepth
    :ivar justification_max_sents: Maximum sentences in justification. Defaults to 2.
    :vartype justification_max_sents: int
    :ivar add_references: Whether to include source references for extracted items.
    :vartype add_references: bool
    :ivar reference_depth: Source reference granularity ("paragraphs" or "sentences").
        Defaults to "paragraphs". Only relevant when references are added to extracted items.
        Affects the structure of ``extracted_items``.
    :vartype reference_depth: ReferenceDepth
    :ivar singular_occurrence: Whether this concept has at most one extracted item. If True,
        at most one item is extracted, and none if there is nothing to extract: an LLM that
        returns more than one is asked again, as for any other invalid data. Defaults to
        False (multiple extracted items are allowed). Set it for a
        concept with one value, e.g. "document title",
        rather than several, e.g. "key findings". Without it,
        an LLM infers the number of items from the concept's name and description.
    :vartype singular_occurrence: StrictBool
    :ivar extracted_items: The items extracted for the concept, set by the extraction
        methods: each has the extracted ``value``, and its ``justification``, references
        (``reference_paragraphs`` and ``reference_sentences``, the document's own objects),
        ``reference_page_numbers`` and ``probabilities`` where these apply. For
        the attributes of the items, see
        https://contextgem.shcherbak.ai/concepts/string_concept/. Defaults to an empty list.
    :vartype extracted_items: list[_StringItem]
    :ivar custom_data: A dictionary for data of your own about the concept, e.g. the ID
        that your application has for it. Must be JSON-serializable. Not sent to the models,
        and kept when the concept is serialized or cloned. Defaults to an empty dict.
    :vartype custom_data: dict[str, Any]
    :ivar unique_id: The unique ID of the concept, a ULID assigned when the concept is
        created. Read-only. Kept when the concept is serialized or cloned, i.e. a restored
        or cloned concept has the ID of the original.
    :vartype unique_id: str

    Example:
        .. literalinclude:: ../../../dev/usage_examples/docstrings/concepts/def_string_concept.py
            :language: python
            :caption: String concept definition

        .. The example above is not in the installed package. It is shown at
           https://contextgem.shcherbak.ai/api/concepts/index.html.md
    """


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
    :ivar model_role: The role label of the model that extracts the concept when
        a ``DocumentModelGroup`` is used: the concept is routed to the model in the group
        with the same ``role``. Defaults to None, which routes the concept to the model
        without a role. Ignored by a standalone model, which extracts every
        concept it is given.
    :vartype model_role: str | None
    :ivar modality: The document content the concept is extracted from: "text", "image",
        or "multimodal" (text and/or images, whichever the document has). "image" and
        "multimodal" are supported for document-level concepts only. Defaults to "text".
    :vartype modality: Modality
    :ivar add_justifications: Whether to include justifications for extracted items.
    :vartype add_justifications: bool
    :ivar justification_depth: Justification detail level. Defaults to "brief".
        Concepts with different values are extracted in separate calls.
    :vartype justification_depth: JustificationDepth
    :ivar justification_max_sents: Maximum sentences in justification. Defaults to 2.
    :vartype justification_max_sents: int
    :ivar add_references: Whether to include source references for extracted items.
    :vartype add_references: bool
    :ivar reference_depth: Source reference granularity ("paragraphs" or "sentences").
        Defaults to "paragraphs". Only relevant when references are added to extracted items.
        Affects the structure of ``extracted_items``.
    :vartype reference_depth: ReferenceDepth
    :ivar singular_occurrence: Whether this concept has at most one extracted item. If True,
        at most one item is extracted, and none if there is nothing to extract: an LLM that
        returns more than one is asked again, as for any other invalid data. Defaults to
        False (multiple extracted items are allowed). Set it for a
        concept with one value, e.g. "contains confidential information",
        rather than several, e.g. "compliance violations". Without it,
        an LLM infers the number of items from the concept's name and description.
        A decision model
        (``DocumentDecisionModel``) extracts at most one item for a concept in any case,
        unless the text is sent in parts (``max_paragraphs_to_analyze_per_call`` passed to
        the extraction method): it then returns an item for each part, and rejects a concept
        with ``singular_occurrence=True`` with ``ValueError``.
    :vartype singular_occurrence: StrictBool
    :ivar decision_threshold: The lowest probability of "yes" (above 0, up to 1) with which
        a decision model (``DocumentDecisionModel``) sets the value of the concept to True.
        Defaults to None, in which case the ``decision_threshold`` of the decision model
        applies. Ignored when the concept is extracted by an LLM.
    :vartype decision_threshold: float | None
    :ivar extracted_items: The items extracted for the concept, set by the extraction
        methods: each has the extracted ``value``, and its ``justification``, references
        (``reference_paragraphs`` and ``reference_sentences``, the document's own objects),
        ``reference_page_numbers`` and ``probabilities`` where these apply. For
        the attributes of the items, see
        https://contextgem.shcherbak.ai/concepts/boolean_concept/. Defaults to an empty list.
    :vartype extracted_items: list[_BooleanItem]
    :ivar custom_data: A dictionary for data of your own about the concept, e.g. the ID
        that your application has for it. Must be JSON-serializable. Not sent to the models,
        and kept when the concept is serialized or cloned. Defaults to an empty dict.
    :vartype custom_data: dict[str, Any]
    :ivar unique_id: The unique ID of the concept, a ULID assigned when the concept is
        created. Read-only. Kept when the concept is serialized or cloned, i.e. a restored
        or cloned concept has the ID of the original.
    :vartype unique_id: str

    Example:
        .. literalinclude:: ../../../dev/usage_examples/docstrings/concepts/def_boolean_concept.py
            :language: python
            :caption: Boolean concept definition

        .. The example above is not in the installed package. It is shown at
           https://contextgem.shcherbak.ai/api/concepts/index.html.md
    """


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
    :ivar model_role: The role label of the model that extracts the concept when
        a ``DocumentModelGroup`` is used: the concept is routed to the model in the group
        with the same ``role``. Defaults to None, which routes the concept to the model
        without a role. Ignored by a standalone model, which extracts every
        concept it is given.
    :vartype model_role: str | None
    :ivar modality: The document content the concept is extracted from: "text", "image",
        or "multimodal" (text and/or images, whichever the document has). "image" and
        "multimodal" are supported for document-level concepts only. Defaults to "text".
    :vartype modality: Modality
    :ivar add_justifications: Whether to include justifications for extracted items.
    :vartype add_justifications: bool
    :ivar justification_depth: Justification detail level. Defaults to "brief".
        Concepts with different values are extracted in separate calls.
    :vartype justification_depth: JustificationDepth
    :ivar justification_max_sents: Maximum sentences in justification. Defaults to 2.
    :vartype justification_max_sents: int
    :ivar add_references: Whether to include source references for extracted items.
    :vartype add_references: bool
    :ivar reference_depth: Source reference granularity ("paragraphs" or "sentences").
        Defaults to "paragraphs". Only relevant when references are added to extracted items.
        Affects the structure of ``extracted_items``.
    :vartype reference_depth: ReferenceDepth
    :ivar singular_occurrence: Whether this concept has at most one extracted item. If True,
        at most one item is extracted, and none if there is nothing to extract: an LLM that
        returns more than one is asked again, as for any other invalid data. Defaults to
        False (multiple extracted items are allowed). Set it for a
        concept with one value, e.g. "total revenue",
        rather than several, e.g. "monthly sales figures". Without it,
        an LLM infers the number of items from the concept's name and description.
    :vartype singular_occurrence: StrictBool
    :ivar extracted_items: The items extracted for the concept, set by the extraction
        methods: each has the extracted ``value``, and its ``justification``, references
        (``reference_paragraphs`` and ``reference_sentences``, the document's own objects),
        ``reference_page_numbers`` and ``probabilities`` where these apply. For
        the attributes of the items, see
        https://contextgem.shcherbak.ai/concepts/numerical_concept/. Defaults to an empty list.
    :vartype extracted_items: list[_IntegerItem | _FloatItem | _IntegerOrFloatItem]
    :ivar custom_data: A dictionary for data of your own about the concept, e.g. the ID
        that your application has for it. Must be JSON-serializable. Not sent to the models,
        and kept when the concept is serialized or cloned. Defaults to an empty dict.
    :vartype custom_data: dict[str, Any]
    :ivar unique_id: The unique ID of the concept, a ULID assigned when the concept is
        created. Read-only. Kept when the concept is serialized or cloned, i.e. a restored
        or cloned concept has the ID of the original.
    :vartype unique_id: str

    Example:
        .. literalinclude:: ../../../dev/usage_examples/docstrings/concepts/def_numerical_concept.py
            :language: python
            :caption: Numerical concept definition

        .. The example above is not in the installed package. It is shown at
           https://contextgem.shcherbak.ai/api/concepts/index.html.md
    """


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
    :ivar rating_scale: The rating scale defining valid value boundaries, as a tuple of
        (start, end) integers.
    :vartype rating_scale: tuple[int, int]
    :ivar model_role: The role label of the model that extracts the concept when
        a ``DocumentModelGroup`` is used: the concept is routed to the model in the group
        with the same ``role``. Defaults to None, which routes the concept to the model
        without a role. Ignored by a standalone model, which extracts every
        concept it is given.
    :vartype model_role: str | None
    :ivar modality: The document content the concept is extracted from: "text", "image",
        or "multimodal" (text and/or images, whichever the document has). "image" and
        "multimodal" are supported for document-level concepts only. Defaults to "text".
    :vartype modality: Modality
    :ivar add_justifications: Whether to include justifications for extracted items.
    :vartype add_justifications: bool
    :ivar justification_depth: Justification detail level. Defaults to "brief".
        Concepts with different values are extracted in separate calls.
    :vartype justification_depth: JustificationDepth
    :ivar justification_max_sents: Maximum sentences in justification. Defaults to 2.
    :vartype justification_max_sents: int
    :ivar add_references: Whether to include source references for extracted items.
    :vartype add_references: bool
    :ivar reference_depth: Source reference granularity ("paragraphs" or "sentences").
        Defaults to "paragraphs". Only relevant when references are added to extracted items.
        Affects the structure of ``extracted_items``.
    :vartype reference_depth: ReferenceDepth
    :ivar singular_occurrence: Whether this concept has at most one extracted item. If True,
        at most one item is extracted, and none if there is nothing to extract: an LLM that
        returns more than one is asked again, as for any other invalid data. Defaults to
        False (multiple extracted items are allowed). Set it for a
        concept with one value, e.g. "product rating score",
        rather than several, e.g. "customer satisfaction ratings". Without it,
        an LLM infers the number of items from the concept's name and description.
        A decision model
        (``DocumentDecisionModel``) extracts at most one item for a concept in any case,
        unless the text is sent in parts (``max_paragraphs_to_analyze_per_call`` passed to
        the extraction method): it then returns an item for each part, and rejects a concept
        with ``singular_occurrence=True`` with ``ValueError``.
    :vartype singular_occurrence: StrictBool
    :ivar extracted_items: The items extracted for the concept, set by the extraction
        methods: each has the extracted ``value``, and its ``justification``, references
        (``reference_paragraphs`` and ``reference_sentences``, the document's own objects),
        ``reference_page_numbers`` and ``probabilities`` where these apply. For
        the attributes of the items, see
        https://contextgem.shcherbak.ai/concepts/rating_concept/. Defaults to an empty list.
    :vartype extracted_items: list[_IntegerItem]
    :ivar custom_data: A dictionary for data of your own about the concept, e.g. the ID
        that your application has for it. Must be JSON-serializable. Not sent to the models,
        and kept when the concept is serialized or cloned. Defaults to an empty dict.
    :vartype custom_data: dict[str, Any]
    :ivar unique_id: The unique ID of the concept, a ULID assigned when the concept is
        created. Read-only. Kept when the concept is serialized or cloned, i.e. a restored
        or cloned concept has the ID of the original.
    :vartype unique_id: str

    Example:
        .. literalinclude:: ../../../dev/usage_examples/docstrings/concepts/def_rating_concept.py
            :language: python
            :caption: Rating concept definition

        .. The example above is not in the installed package. It is shown at
           https://contextgem.shcherbak.ai/api/concepts/index.html.md
    """


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
        JSON-serializable, and mappings must have string keys (``dict[str, ...]``), as JSON
        objects do. Extracted values are validated strictly against the structure.
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

        **Note**: A class may have fields whose types are other classes, at any depth: they
        are converted to nested structures, with no base class needed. Only the classes
        defined inside a function, in a module with ``from __future__ import annotations``,
        are to inherit from ``JsonObjectClassStruct``, as the classes that their annotations
        name cannot be found otherwise.

        **Tip**: do not overcomplicate the structure to avoid prompt overloading.
    :vartype structure: type | dict[str, Any]
    :ivar examples: Example JSON objects illustrating the concept usage.
    :vartype examples: list[JsonObjectExample]
    :ivar model_role: The role label of the model that extracts the concept when
        a ``DocumentModelGroup`` is used: the concept is routed to the model in the group
        with the same ``role``. Defaults to None, which routes the concept to the model
        without a role. Ignored by a standalone model, which extracts every
        concept it is given.
    :vartype model_role: str | None
    :ivar modality: The document content the concept is extracted from: "text", "image",
        or "multimodal" (text and/or images, whichever the document has). "image" and
        "multimodal" are supported for document-level concepts only. Defaults to "text".
    :vartype modality: Modality
    :ivar add_justifications: Whether to include justifications for extracted items.
    :vartype add_justifications: bool
    :ivar justification_depth: Justification detail level. Defaults to "brief".
        Concepts with different values are extracted in separate calls.
    :vartype justification_depth: JustificationDepth
    :ivar justification_max_sents: Maximum sentences in justification. Defaults to 2.
    :vartype justification_max_sents: int
    :ivar add_references: Whether to include source references for extracted items.
    :vartype add_references: bool
    :ivar reference_depth: Source reference granularity ("paragraphs" or "sentences").
        Defaults to "paragraphs". Only relevant when references are added to extracted items.
        Affects the structure of ``extracted_items``.
    :vartype reference_depth: ReferenceDepth
    :ivar singular_occurrence: Whether this concept has at most one extracted item. If True,
        at most one item is extracted, and none if there is nothing to extract: an LLM that
        returns more than one is asked again, as for any other invalid data. Defaults to
        False (multiple extracted items are allowed). Set it for a
        concept with one value, e.g. "product specifications",
        rather than several, e.g. "customer order details". Without it,
        an LLM infers the number of items from the concept's name and description.
    :vartype singular_occurrence: StrictBool
    :ivar extracted_items: The items extracted for the concept, set by the extraction
        methods: each has the extracted ``value``, and its ``justification``, references
        (``reference_paragraphs`` and ``reference_sentences``, the document's own objects),
        ``reference_page_numbers`` and ``probabilities`` where these apply. For
        the attributes of the items, see
        https://contextgem.shcherbak.ai/concepts/json_object_concept/. Defaults to an empty list.
    :vartype extracted_items: list[_JsonObjectItem]
    :ivar custom_data: A dictionary for data of your own about the concept, e.g. the ID
        that your application has for it. Must be JSON-serializable. Not sent to the models,
        and kept when the concept is serialized or cloned. Defaults to an empty dict.
    :vartype custom_data: dict[str, Any]
    :ivar unique_id: The unique ID of the concept, a ULID assigned when the concept is
        created. Read-only. Kept when the concept is serialized or cloned, i.e. a restored
        or cloned concept has the ID of the original.
    :vartype unique_id: str

    Example:
        .. literalinclude:: ../../../dev/usage_examples/docstrings/concepts/def_json_object_concept.py
            :language: python
            :caption: JSON object concept definition

        .. The example above is not in the installed package. It is shown at
           https://contextgem.shcherbak.ai/api/concepts/index.html.md
    """


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
    :ivar model_role: The role label of the model that extracts the concept when
        a ``DocumentModelGroup`` is used: the concept is routed to the model in the group
        with the same ``role``. Defaults to None, which routes the concept to the model
        without a role. Ignored by a standalone model, which extracts every
        concept it is given.
    :vartype model_role: str | None
    :ivar modality: The document content the concept is extracted from: "text", "image",
        or "multimodal" (text and/or images, whichever the document has). "image" and
        "multimodal" are supported for document-level concepts only. Defaults to "text".
    :vartype modality: Modality
    :ivar add_justifications: Whether to include justifications for extracted items.
    :vartype add_justifications: bool
    :ivar justification_depth: Justification detail level. Defaults to "brief".
        Concepts with different values are extracted in separate calls.
    :vartype justification_depth: JustificationDepth
    :ivar justification_max_sents: Maximum sentences in justification. Defaults to 2.
    :vartype justification_max_sents: int
    :ivar add_references: Whether to include source references for extracted items.
    :vartype add_references: bool
    :ivar reference_depth: Source reference granularity ("paragraphs" or "sentences").
        Defaults to "paragraphs". Only relevant when references are added to extracted items.
        Affects the structure of ``extracted_items``.
    :vartype reference_depth: ReferenceDepth
    :ivar singular_occurrence: Whether this concept has at most one extracted item. If True,
        at most one item is extracted, and none if there is nothing to extract: an LLM that
        returns more than one is asked again, as for any other invalid data. Defaults to
        False (multiple extracted items are allowed). Set it for a
        concept with one value, e.g. "contract signing date",
        rather than several, e.g. "meeting dates". Without it,
        an LLM infers the number of items from the concept's name and description.
    :vartype singular_occurrence: StrictBool
    :ivar extracted_items: The items extracted for the concept, set by the extraction
        methods: each has the extracted ``value``, and its ``justification``, references
        (``reference_paragraphs`` and ``reference_sentences``, the document's own objects),
        ``reference_page_numbers`` and ``probabilities`` where these apply. For
        the attributes of the items, see
        https://contextgem.shcherbak.ai/concepts/date_concept/. Defaults to an empty list.
    :vartype extracted_items: list[_DateItem]
    :ivar custom_data: A dictionary for data of your own about the concept, e.g. the ID
        that your application has for it. Must be JSON-serializable. Not sent to the models,
        and kept when the concept is serialized or cloned. Defaults to an empty dict.
    :vartype custom_data: dict[str, Any]
    :ivar unique_id: The unique ID of the concept, a ULID assigned when the concept is
        created. Read-only. Kept when the concept is serialized or cloned, i.e. a restored
        or cloned concept has the ID of the original.
    :vartype unique_id: str

    Example:
        .. literalinclude:: ../../../dev/usage_examples/docstrings/concepts/def_date_concept.py
            :language: python
            :caption: Date concept definition

        .. The example above is not in the installed package. It is shown at
           https://contextgem.shcherbak.ai/api/concepts/index.html.md
    """


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
    :ivar model_role: The role label of the model that extracts the concept when
        a ``DocumentModelGroup`` is used: the concept is routed to the model in the group
        with the same ``role``. Defaults to None, which routes the concept to the model
        without a role. Ignored by a standalone model, which extracts every
        concept it is given.
    :vartype model_role: str | None
    :ivar modality: The document content the concept is extracted from: "text", "image",
        or "multimodal" (text and/or images, whichever the document has). "image" and
        "multimodal" are supported for document-level concepts only. Defaults to "text".
    :vartype modality: Modality
    :ivar add_justifications: Whether to include justifications for extracted items.
    :vartype add_justifications: bool
    :ivar justification_depth: Justification detail level. Defaults to "brief".
        Concepts with different values are extracted in separate calls.
    :vartype justification_depth: JustificationDepth
    :ivar justification_max_sents: Maximum sentences in justification. Defaults to 2.
    :vartype justification_max_sents: int
    :ivar add_references: Whether to include source references for extracted items.
    :vartype add_references: bool
    :ivar reference_depth: Source reference granularity ("paragraphs" or "sentences").
        Defaults to "paragraphs". Only relevant when references are added to extracted items.
        Affects the structure of ``extracted_items``.
    :vartype reference_depth: ReferenceDepth
    :ivar singular_occurrence: Whether this concept has at most one extracted item. If True,
        at most one item is extracted, and none if there is nothing to extract: an LLM that
        returns more than one is asked again, as for any other invalid data. Defaults to
        False (multiple extracted items are allowed). Set it for a
        concept with one value, e.g. "document type",
        rather than several, e.g. "content topics". Without it,
        an LLM infers the number of items from the concept's name and description.
        A decision model (``DocumentDecisionModel``)
        extracts at most one item for a concept in any case, unless the text is sent in
        parts (``max_paragraphs_to_analyze_per_call`` passed to the extraction method): it
        then returns an item for each part, and rejects a concept with
        ``singular_occurrence=True`` with ``ValueError``.
    :vartype singular_occurrence: bool
    :ivar decision_threshold: The lowest probability (above 0, up to 1) with which
        a decision model (``DocumentDecisionModel``) selects a label of the concept.
        Can be set for the "multi_label" classification type only: with "multi_class",
        the most probable label is always selected. Defaults to None, in which case
        the ``decision_threshold`` of the decision model applies. Ignored when
        the concept is extracted by an LLM.
    :vartype decision_threshold: float | None
    :ivar extracted_items: The items extracted for the concept, set by the extraction
        methods: each has the extracted ``value``, and its ``justification``, references
        (``reference_paragraphs`` and ``reference_sentences``, the document's own objects),
        ``reference_page_numbers`` and ``probabilities`` where these apply. For
        the attributes of the items, see
        https://contextgem.shcherbak.ai/concepts/label_concept/. Defaults to an empty list.
    :vartype extracted_items: list[_LabelItem]
    :ivar custom_data: A dictionary for data of your own about the concept, e.g. the ID
        that your application has for it. Must be JSON-serializable. Not sent to the models,
        and kept when the concept is serialized or cloned. Defaults to an empty dict.
    :vartype custom_data: dict[str, Any]
    :ivar unique_id: The unique ID of the concept, a ULID assigned when the concept is
        created. Read-only. Kept when the concept is serialized or cloned, i.e. a restored
        or cloned concept has the ID of the original.
    :vartype unique_id: str

    Example:
        .. literalinclude:: ../../../dev/usage_examples/docstrings/concepts/def_label_concept.py
            :language: python
            :caption: Label concept definition

        .. The example above is not in the installed package. It is shown at
           https://contextgem.shcherbak.ai/api/concepts/index.html.md
    """
