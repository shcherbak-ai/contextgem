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
Module for handling document aspects.

This module provides the Aspect class, which represents a defined area or topic within a document
that requires focused attention. Aspects are used to identify and extract specific subjects or themes
from documents according to predefined criteria.

Aspects can be associated with concepts, reference paragraphs and sentences from the source document,
and can be configured with different LLM roles for extraction and reasoning tasks.

The module integrates with the broader ContextGem framework for document analysis and information extraction.
"""

from __future__ import annotations

from typing import Any

from pydantic import (
    Field,
    PrivateAttr,
    StrictBool,
    StrictInt,
    field_validator,
    model_validator,
)

from contextgem.internal.base.attrs import (
    _AssignedInstancesProcessor,
    _ExtractedItemsAttributeProcessor,
    _RefParasAndSentsAttrituteProcessor,
)
from contextgem.internal.base.concepts import _Concept
from contextgem.internal.items import _StringItem
from contextgem.internal.typings.aliases import (
    LLMRoleAspect,
    NonEmptyStr,
    ReferenceDepth,
    Self,
)
from contextgem.public.paragraphs import Paragraph
from contextgem.public.sentences import Sentence

# Defines the maximum nesting level of sub-aspects.
MAX_NESTING_LEVEL = 1


class Aspect(
    _AssignedInstancesProcessor,
    _ExtractedItemsAttributeProcessor,
    _RefParasAndSentsAttrituteProcessor,
):
    """
    Represents an aspect with associated metadata, sub-aspects, concepts, and logic for validation.

    An aspect is a defined area or topic within a document that requires focused attention.
    Each aspect corresponds to a specific subject or theme described in the task.

    :ivar name: The name of the aspect. Required, non-empty string.
    :type name: NonEmptyStr
    :ivar description: A detailed description of the aspect. Required, non-empty string.
    :type description: NonEmptyStr
    :ivar concepts: A list of concepts associated with the aspect. These concepts must be
        unique in both name and description and cannot include concepts with vision LLM roles.
    :type concepts: list[_Concept]
    :ivar llm_role: The role of the LLM responsible for aspect extraction.
        Default is "extractor_text". Valid roles are "extractor_text" and "reasoner_text".
    :type llm_role: LLMRoleAspect
    :ivar reference_depth: The structural depth of references (paragraphs or sentences).
        Defaults to "paragraphs". Affects the structure of ``extracted_items``.
    :type reference_depth: ReferenceDepth
    :ivar add_justifications: Whether the LLM will output justification for each extracted item.
        Inherited from base class. Defaults to False.
    :type add_justifications: bool
    :ivar justification_depth: The level of detail for justifications.
        Inherited from base class. Defaults to "brief".
    :type justification_depth: JustificationDepth
    :ivar justification_max_sents: Maximum number of sentences in a justification.
        Inherited from base class. Defaults to 2.
    :type justification_max_sents: int

    Example:
        .. literalinclude:: ../../../dev/usage_examples/docstrings/aspects/def_aspect.py
            :language: python
            :caption: Aspect definition
    """

    name: NonEmptyStr
    description: NonEmptyStr
    aspects: list[Aspect] = Field(default_factory=list)  # sub-aspects
    concepts: list[_Concept] = Field(default_factory=list)
    llm_role: LLMRoleAspect = Field(default="extractor_text")
    reference_depth: ReferenceDepth = Field(default="paragraphs")

    _extracted_items: list[_StringItem] = PrivateAttr(default_factory=list)
    _reference_paragraphs: list[Paragraph] = PrivateAttr(default_factory=list)
    _reference_sentences: list[Sentence] = PrivateAttr(default_factory=list)
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
        """

        if name == "aspects":
            field_validator = Aspect.__pydantic_validator__.validate_assignment
            aspects = field_validator(self, "aspects", value).aspects
            self._validate_and_process_sub_aspects(aspects)
            super().__setattr__(name, aspects)
        else:
            super().__setattr__(name, value)

    @property
    def _item_class(self) -> type[_StringItem]:
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
        :raises ValueError: If any concept has an LLM role ending with '_vision'.
        :return: The validated list of '_Concept' instances.
        :rtype: list[_Concept]
        """

        if concepts:
            # Validate for Aspect-specific constraints.
            if any(i.llm_role.endswith("_vision") for i in concepts):
                raise ValueError(
                    "Aspect concepts extraction using vision LLMs is not supported. "
                    "Vision LLMs can be used only for document concept extraction."
                )
        return concepts

    def _validate_and_process_sub_aspects(self, aspects: list[Aspect]) -> None:
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

        if parent_level >= MAX_NESTING_LEVEL:
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
                aspect._validate_and_process_sub_aspects(aspect.aspects)

    @model_validator(mode="after")
    def _validate_aspect_post(self) -> Self:
        """
        Validates and processes the assigned sub-aspects post-initialization.

        :return: The validated Aspect instance
        :rtype: Self
        """
        if self.aspects:
            self._validate_and_process_sub_aspects(self.aspects)
        return self
