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
Module defining the base classes for Aspect class.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Annotated, Any, cast

from pydantic import (
    BeforeValidator,
    Field,
    PrivateAttr,
    StrictBool,
    StrictFloat,
    StrictInt,
    field_validator,
    model_validator,
)
from typing_extensions import Self

from contextgem.internal.base.attrs import (
    _AssignedInstancesProcessor,
    _ExtractedItemsAttributeProcessor,
    _RefParasAndSentsAttrituteProcessor,
)
from contextgem.internal.base.concepts import _Concept
from contextgem.internal.base.paras_and_sents import _Paragraph, _Sentence
from contextgem.internal.decorators import _disable_direct_initialization
from contextgem.internal.items import _StringItem
from contextgem.internal.typings.types import (
    ModelRole,
    NonEmptyStr,
    ReferenceDepth,
)
from contextgem.internal.typings.validators import _validate_sequence_is_list


# Defines the maximum nesting level of sub-aspects.
_MAX_NESTING_LEVEL = 1


@_disable_direct_initialization
class _Aspect(
    _AssignedInstancesProcessor,
    _ExtractedItemsAttributeProcessor,
    _RefParasAndSentsAttrituteProcessor,
):
    """
    Internal implementation of the Aspect class.
    """

    name: NonEmptyStr = Field(
        ..., description="Aspect name (required, non-empty string)."
    )
    description: NonEmptyStr = Field(
        ..., description="Aspect description (required, non-empty string)."
    )
    aspects: Annotated[
        Sequence[_Aspect], BeforeValidator(_validate_sequence_is_list)
    ] = Field(
        default_factory=list,
        description=(
            "Sub-aspects of this aspect. Max nesting level: 1. Names and descriptions "
            "must differ from the parent."
        ),
    )  # using Sequence type with list validator for type checking
    concepts: Annotated[
        Sequence[_Concept], BeforeValidator(_validate_sequence_is_list)
    ] = Field(
        default_factory=list,
        description=(
            "Concepts associated with this aspect. Must be unique by name and description. "
            "Only concepts with 'text' modality are allowed."
        ),
    )  # using Sequence field with list validator for type checking
    model_role: ModelRole | None = Field(
        default=None,
        description=(
            "Role label of the LLM that extracts this aspect when a DocumentModelGroup "
            "is used: the aspect is routed to the LLM in the group with the same `role`. "
            "Defaults to None, which routes to the LLM without a role. "
            "Ignored by a standalone DocumentLLM, which extracts every aspect it is given."
        ),
    )
    reference_depth: ReferenceDepth = Field(
        default="paragraphs",
        description=(
            "Reference granularity for extraction outputs: 'paragraphs' or 'sentences'. "
            "Affects the structure of extracted_items."
        ),
    )
    decision_threshold: StrictFloat | None = Field(
        default=None,
        gt=0,
        le=1,
        description=(
            "Lowest probability with which a decision model takes a paragraph, or "
            "a sentence of such a paragraph if the reference depth is 'sentences', as "
            "belonging to this aspect (0 < threshold <= 1). Defaults to None, in which "
            "case the `decision_threshold` of the decision model applies. Not passed on "
            "to the sub-aspects. Ignored when the aspect is extracted by an LLM."
        ),
    )

    _extracted_items: list[_StringItem] = PrivateAttr(default_factory=list)
    _reference_paragraphs: list[_Paragraph] = PrivateAttr(default_factory=list)
    _reference_sentences: list[_Sentence] = PrivateAttr(default_factory=list)
    _nesting_level: StrictInt = PrivateAttr(default=0)
    _is_processed: StrictBool = PrivateAttr(default=False)

    def __setattr__(self, name: str, value: Any) -> None:
        """
        Custom attribute setter that applies processing for the 'aspects' attribute
        on each assignment, which requires comparing such sub-aspects to the parent aspect.

        :param name: The name of the attribute to set
        :type name: str
        :param value: The value to assign to the attribute
        :type value: Any
        :return: None
        :rtype: None
        """

        if name == "aspects":
            # The sub-aspects are validated here, without the attribute setter of
            # the base class, i.e. the fields are to be restored here as well
            # if the sub-aspects are rejected
            fields_snapshot = self._get_fields_snapshot()
            try:
                field_validator = _Aspect.__pydantic_validator__.validate_assignment
                # Safe cast: Pydantic validator returns validated instance, we know it's an Aspect
                # since we're validating the "aspects" field of an Aspect object
                validated_instance = cast(
                    _Aspect, field_validator(self, "aspects", value)
                )
                aspects = validated_instance.aspects
                self._validate_and_process_sub_aspects(aspects)  # ty: ignore[invalid-argument-type]
                super().__setattr__(name, aspects)
            except Exception:
                self._restore_fields(fields_snapshot)
                raise
        else:
            super().__setattr__(name, value)

    @property
    def _item_class(self) -> type[_StringItem]:
        """
        Returns the item class type for aspects.

        :return: The string item class used for aspect extracted items.
        :rtype: type[_StringItem]
        """
        return _StringItem

    @field_validator("concepts")
    @classmethod
    def _validate_concepts_in_aspect(cls, concepts: list[_Concept]) -> list[_Concept]:
        """
        Validates the input list of '_Concept' instances.

        :param concepts: List of '_Concept' instances to be validated.
        :type concepts: list[_Concept]
        :raises ValueError: If multiple concepts have the same name.
        :raises ValueError: If multiple concepts have the same description.
        :raises ValueError: If any concept has a modality other than 'text'.
        :return: The validated list of '_Concept' instances.
        :rtype: list[_Concept]
        """

        if any(i.modality != "text" for i in concepts):
            # Validate for Aspect-specific constraints.
            raise ValueError(
                "Aspect-level concepts are extracted from text only. Concepts with "
                "`image` or `multimodal` modality can be extracted only at the document level."
            )
        return concepts

    def _validate_and_process_sub_aspects(self, aspects: list[_Aspect]) -> None:
        """
        Validates and processes sub-aspects by setting their nesting levels
        and checking for duplicate names or descriptions with parent.

        :param aspects: List of sub-aspects to validate
        :type aspects: list["Aspect"]
        :return: None
        :rtype: None
        :raises ValueError: If any sub-aspect has the same name or description as parent
        """

        if not aspects:
            return

        parent_level = self._nesting_level

        if parent_level >= _MAX_NESTING_LEVEL:
            raise ValueError(
                f"Aspect `{self.name}` is already a sub-aspect with the maximum nesting level "
                f"{parent_level}. No further sub-aspects can be assigned to this aspect."
            )

        parent_name = self.name
        parent_description = self.description

        for aspect in aspects:
            # Check for duplicate names or descriptions with parent
            if aspect.name == parent_name:
                raise ValueError(
                    f"Sub-aspect `{aspect.name}` cannot have "
                    f"the same name as parent aspect: "
                    f"'{parent_name}'"
                )
            if aspect.description == parent_description:
                raise ValueError(
                    f"Sub-aspect `{aspect.name}` cannot have "
                    f"the same description as parent aspect: "
                    f"'{parent_name}'"
                )
            # Set nesting level
            aspect._nesting_level = parent_level + 1

            # Recursively process sub-aspects of this aspect
            if aspect.aspects:
                aspect._validate_and_process_sub_aspects(aspect.aspects)  # ty: ignore[invalid-argument-type]

    def _reset_nesting_levels(self, level: int = 0) -> None:
        """
        Recursively (re)assigns nesting levels, treating this aspect as residing
        at ``level`` and its sub-aspects at the subsequent levels.

        ``_nesting_level`` is assigned relative to the parent when an aspect is
        attached as a sub-aspect, and is never otherwise reset. When an aspect
        that was previously used as a sub-aspect (nesting level > 0) is reused as
        a top-level aspect (e.g. attached to a ``Document``), its stale level
        must be normalized so that nesting-level validation and sub-aspect
        extraction behave correctly. Top-level containers call this with
        ``level=0``.

        :param level: The nesting level to assign to this aspect.
        :type level: int
        :return: None
        :rtype: None
        """
        self._nesting_level = level
        for sub_aspect in self.aspects:
            sub_aspect._reset_nesting_levels(level + 1)

    def _add_references_of_extracted_items(self) -> None:
        """
        Adds the references of the extracted items of the aspect to the references
        of the aspect itself: the reference paragraphs of the items, and, for
        an aspect with the "sentences" reference depth, their reference sentences,
        i.e. the extracted sentences, and not all the sentences of the reference
        paragraphs. A sentence is added also when its paragraph is a reference
        paragraph already, e.g. that of another sentence extracted from it.

        The references that the aspect already has are kept, and those of the items
        follow them in the order of the items. The items may have overlapping
        references: a paragraph or a sentence becomes a reference of the aspect
        once. It is told from the others by its unique ID, as it is when
        the references are assigned, and not by equality: a copy with the same
        unique ID and other content, e.g. one without the sentences of
        the original, is not equal to the original, while it is the same reference.
        Each list is assigned when it is complete, so that it is validated before
        the references of the aspect are changed.

        :return: None
        :rtype: None
        """
        with_sentences = self.reference_depth == "sentences"
        reference_paragraphs = list(self.reference_paragraphs)
        reference_sentences = list(self.reference_sentences)
        paragraph_ids = {paragraph.unique_id for paragraph in reference_paragraphs}
        sentence_ids = {sentence.unique_id for sentence in reference_sentences}
        for extracted_item in self.extracted_items:
            for paragraph in extracted_item.reference_paragraphs:
                if paragraph.unique_id not in paragraph_ids:
                    paragraph_ids.add(paragraph.unique_id)
                    reference_paragraphs.append(paragraph)
            if with_sentences:
                for sentence in extracted_item.reference_sentences:
                    if sentence.unique_id not in sentence_ids:
                        sentence_ids.add(sentence.unique_id)
                        reference_sentences.append(sentence)
        self.reference_paragraphs = reference_paragraphs
        if with_sentences:
            self.reference_sentences = reference_sentences

    @model_validator(mode="after")
    def _validate_aspect_post(self) -> Self:
        """
        Validates and processes the assigned sub-aspects post-initialization.

        :return: The validated Aspect instance
        :rtype: Self
        """
        if self.aspects:
            self._validate_and_process_sub_aspects(list(self.aspects))
        return self
