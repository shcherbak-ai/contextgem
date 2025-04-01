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
Module for computing structures used for validating LLM responses parsed as JSON during
aspect extraction from a document. Such structures must match the JSON schema specified in the relevant
LLM prompt.
"""

from functools import lru_cache

from pydantic import ConfigDict, RootModel, create_model

from contextgem.internal.llm_output_structs.utils import _create_root_model
from contextgem.internal.typings.aliases import NonEmptyStr, ReferenceDepth


@lru_cache(maxsize=None)
def _get_aspect_extraction_output_struct(
    with_extra_data: bool, reference_depth: ReferenceDepth
) -> RootModel:
    """
    Computes, caches and returns a dynamically generated root model for aspect extraction
    based on the specified parameters. The model is used for validating LLM responses parsed as JSON.
    The structure of JSON response is defined in the relevant LLM prompt.

    :param with_extra_data: A boolean flag indicating whether to include additional
        data attributes (e.g., `justification`) in the generated models.
    :type with_extra_data: bool
    :param reference_depth: The structural depth of the references, i.e. whether to provide
        paragraphs as references or sentences as references. Defaults to "paragraphs".
        ``extracted_items`` will have values based on this parameter.
    :type reference_depth: ReferenceDepth
    :return: A dynamically generated `RootModel` object encapsulating aspects, each
        containing the relevant fields based on the parameters provided.
    :rtype: RootModel
    """

    if with_extra_data:
        # Sentence-level reference depth
        if reference_depth == "sentences":
            sentence_model_kwargs = {
                "sentence_id": (NonEmptyStr, ...),
                "justification": (NonEmptyStr, ...),
            }
            SentenceModel = create_model(
                "SentenceModel",
                __config__=ConfigDict(extra="forbid"),
                **sentence_model_kwargs,
            )
            paragraph_model_kwargs = {
                "paragraph_id": (NonEmptyStr, ...),
                "sentences": (list[SentenceModel], ...),
            }
        # Paragraph-level reference depth
        else:
            paragraph_model_kwargs = {
                "paragraph_id": (NonEmptyStr, ...),
                "justification": (NonEmptyStr, ...),
            }
        ParagraphModel = create_model(
            "ParagraphModel",
            __config__=ConfigDict(extra="forbid"),
            **paragraph_model_kwargs,
        )
        aspect_model_kwargs = {
            "aspect_id": (NonEmptyStr, ...),
            "paragraphs": (list[ParagraphModel], ...),
        }

    else:
        # Sentence-level reference depth
        if reference_depth == "sentences":
            sentence_model_kwargs = {
                "sentence_id": (NonEmptyStr, ...),
            }
            SentenceModel = create_model(
                "SentenceModel",
                __config__=ConfigDict(extra="forbid"),
                **sentence_model_kwargs,
            )
            paragraph_model_kwargs = {
                "paragraph_id": (NonEmptyStr, ...),
                "sentences": (list[SentenceModel], ...),
            }
            ParagraphModel = create_model(
                "ParagraphModel",
                __config__=ConfigDict(extra="forbid"),
                **paragraph_model_kwargs,
            )
            aspect_model_kwargs = {
                "aspect_id": (NonEmptyStr, ...),
                "paragraphs": (list[ParagraphModel], ...),
            }
        # Paragraph-level reference depth
        else:
            aspect_model_kwargs = {
                "aspect_id": (NonEmptyStr, ...),
                "paragraph_ids": (list[NonEmptyStr], ...),
            }

    AspectModel = create_model(
        "AspectModel",
        __config__=ConfigDict(extra="forbid"),
        **aspect_model_kwargs,
    )
    DynamicRootModel = _create_root_model("DynamicRootModel", list[AspectModel])
    return DynamicRootModel
