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
Module for computing and defining structures used for validating LLM responses parsed as JSON during
concept extraction from a document or aspect.

This module contains:
1. Dynamic structures that match the JSON schema specified in LLM prompts for concept extraction
2. Static Pydantic models for validating specific concept types' extracted item values
returned in LLM responses

All structures ensure proper validation of LLM outputs according to the expected response formats.
"""

from __future__ import annotations

from functools import cache
from typing import Any, cast

from pydantic import BaseModel, ConfigDict, Field, RootModel, create_model

from contextgem.internal.llm_output_structs.utils import _create_root_model
from contextgem.internal.typings.aliases import NonEmptyStr, ReferenceDepth


@cache
def _get_concept_extraction_output_struct(
    with_extra_data: bool,
    with_references: bool,
    reference_depth: ReferenceDepth,
    with_justification: bool,
) -> type[RootModel]:
    """
    Computes, caches and returns a dynamically generated root model for concept extraction
    based on the specified parameters. The model is used for validating LLM responses parsed as JSON.
    The structure of JSON response is defined in the relevant LLM prompt.

    :param with_extra_data: Flag indicating whether additional data fields need to
        be included in the output structure.
    :type with_extra_data: bool
    :param with_references: Boolean flag specifying whether references should be
        included for the extracted concepts' items.
    :type with_references: bool
    :param reference_depth: Specifies the depth level for references, such as
        "paragraphs" or "sentences".
    :type reference_depth: ReferenceDepth
    :param with_justification: Boolean indicating whether justification fields need
        to be added to the model.
    :type with_justification: bool
    :return: A dynamically generated `RootModel` class encapsulating concepts, each
        containing the relevant fields based on the parameters provided.
    :rtype: type[RootModel]
    """
    if with_extra_data:
        extracted_item_model_kwargs: dict[str, Any] = {
            "value": (Any, ...),
        }
        if with_justification:
            # Safe cast: type checker can't infer tuple types for create_model field definitions
            extracted_item_model_kwargs["justification"] = cast(Any, (NonEmptyStr, ...))
        if with_references:
            # Sentence-level reference depth
            if reference_depth == "sentences":
                reference_paragraph_model_kwargs = {
                    "reference_paragraph_id": (NonEmptyStr, ...),
                    "reference_sentence_ids": (list[NonEmptyStr], ...),
                }
                # Safe cast: type checker can't infer types when unpacking kwargs to create_model
                reference_paragraph_model = create_model(
                    "ReferenceParagraphModel",
                    __config__=ConfigDict(extra="forbid"),
                    **cast(Any, reference_paragraph_model_kwargs),
                )
                extracted_item_model_kwargs["reference_paragraphs"] = cast(
                    Any,
                    (
                        list[reference_paragraph_model],
                        ...,
                    ),
                )
            # Paragraph-level reference depth
            else:
                extracted_item_model_kwargs["reference_paragraph_ids"] = cast(
                    Any,
                    (
                        list[NonEmptyStr],
                        ...,
                    ),
                )
        # Safe cast: type checker can't infer types when unpacking kwargs to create_model
        extracted_item_model = create_model(
            "ExtractedItemModel",
            __config__=ConfigDict(extra="forbid"),
            **cast(Any, extracted_item_model_kwargs),
        )
        concept_model_kwargs = {
            "concept_id": (NonEmptyStr, ...),
            "extracted_items": (list[extracted_item_model], ...),
        }

    else:
        concept_model_kwargs = {
            "concept_id": (NonEmptyStr, ...),
            "extracted_items": (list[Any], ...),
        }

    # Safe cast: type checker can't infer types when unpacking kwargs to create_model
    concept_model = create_model(
        "ConceptModel",
        __config__=ConfigDict(extra="forbid"),
        **cast(Any, concept_model_kwargs),
    )

    dynamic_root_model = _create_root_model("DynamicRootModel", list[concept_model])
    return dynamic_root_model


# Dedicated models for specific concept types' extracted item value validation


class _LabelConceptItemValueModel(BaseModel):
    """
    Pydantic model for validating LabelConcept extracted item values from LLM responses.

    This model validates the structure and constraints of label classification responses
    where the LLM returns a dictionary containing a list of selected labels.

    Expected structure: {"labels": [str, ...]} with at least one label.

    :ivar labels: List of selected label strings. Must contain at least one item.
    :vartype labels: list[str]
    """

    labels: list[NonEmptyStr] = Field(
        ...,
        min_length=1,
        description="List of selected label strings for the concept; must contain at least one label.",
    )

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
