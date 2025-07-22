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

from abc import abstractmethod
from typing import Any

from pydantic import Field, PrivateAttr, StrictBool, model_validator

from contextgem.internal.base.attrs import _ExtractedItemsAttributeProcessor
from contextgem.internal.base.items import _ExtractedItem
from contextgem.internal.typings.aliases import LLMRoleAny, NonEmptyStr, ReferenceDepth


class _Concept(_ExtractedItemsAttributeProcessor):
    """
    Base class for all concept types in the ContextGem framework.

    A concept represents a specific unit of information to be extracted from documents.
    It can be derived from an aspect or directly from document content, capturing
    meaningful data points such as names, dates, values, or conclusions.

    :ivar name: The name of the concept.
    :vartype name: str
    :ivar description: A brief description of the concept.
    :vartype description: str
    :ivar llm_role: The role of the LLM when processing this concept.
        Options: "extractor_text", "reasoner_text", "extractor_vision", "reasoner_vision".
        Defaults to "extractor_text".
    :vartype llm_role: LLMRoleAny
    :ivar add_justifications: Whether to include justifications for extracted items.
        Defaults to False.
    :vartype add_justifications: StrictBool
    :ivar justification_depth: Detail level for justifications. Defaults to "brief".
    :vartype justification_depth: JustificationDepth
    :ivar justification_max_sents: Maximum number of sentences in a justification.
        Defaults to 2.
    :vartype justification_max_sents: StrictInt
    :ivar add_references: Whether to include references for extracted items.
        Defaults to False.
    :vartype add_references: StrictBool
    :ivar reference_depth: Structural depth of references ("paragraphs" or "sentences").
        Defaults to "paragraphs". Only relevant when references are added to extracted items.
        Affects the structure of ``extracted_items``.
    :vartype reference_depth: ReferenceDepth
    :ivar singular_occurrence: Whether this concept is restricted to having only one extracted item.
        If True, only a single extracted item will be extracted. Defaults to False (multiple
        extracted items are allowed). Note that with advanced LLMs, this constraint may not be
        strictly required as they can often infer the appropriate number of items to extract
        from the concept's name, description, and type.
    :vartype singular_occurrence: StrictBool
    """

    name: NonEmptyStr
    description: NonEmptyStr
    llm_role: LLMRoleAny = Field(default="extractor_text")
    add_references: StrictBool = Field(default=False)
    reference_depth: ReferenceDepth = Field(default="paragraphs")
    singular_occurrence: StrictBool = Field(default=False)

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
