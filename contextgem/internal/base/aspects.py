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
from contextgem.internal.typings.aliases import (
    LLMRoleAspect,
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
            "Concepts with vision LLM roles are not allowed."
        ),
    )  # using Sequence field with list validator for type checking
    llm_role: LLMRoleAspect = Field(
        default="extractor_text",
        description=(
            "LLM role used for aspect extraction. Valid values: 'extractor_text' or "
            "'reasoner_text'."
        ),
    )
    reference_depth: ReferenceDepth = Field(
        default="paragraphs",
        description=(
            "Reference granularity for extraction outputs: 'paragraphs' or 'sentences'. "
            "Affects the structure of extracted_items."
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
        """

        if name == "aspects":
            field_validator = _Aspect.__pydantic_validator__.validate_assignment
            # Safe cast: Pydantic validator returns validated instance, we know it's an Aspect
            # since we're validating the "aspects" field of an Aspect object
            validated_instance = cast(_Aspect, field_validator(self, "aspects", value))
            aspects = validated_instance.aspects
            self._validate_and_process_sub_aspects(aspects)
            super().__setattr__(name, aspects)
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
        :raises ValueError: If any concept has an LLM role ending with '_vision'.
        :return: The validated list of '_Concept' instances.
        :rtype: list[_Concept]
        """

        if concepts and any(i.llm_role.endswith("_vision") for i in concepts):
            # Validate for Aspect-specific constraints.
            raise ValueError(
                "Aspect concepts extraction using vision LLMs is not supported. "
                "Vision LLMs can be used only for document concept extraction."
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
                aspect._validate_and_process_sub_aspects(aspect.aspects)

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
