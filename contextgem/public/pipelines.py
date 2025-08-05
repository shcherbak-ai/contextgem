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
Module for handling document processing pipelines.

This module provides the ExtractionPipeline class, which represents a reusable collection
of pre-defined aspects and concepts that can be assigned to documents. Pipelines enable
standardized document analysis by packaging common extraction patterns into reusable units.

Pipelines serve as templates for document processing, allowing consistent application of
the same analysis approach across multiple documents. They encapsulate both the structural
organization (aspects) and the specific information to extract (concepts) in a single,
assignable object.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import Annotated, Any

from pydantic import BeforeValidator, Field

from contextgem.internal.base.attrs import _AssignedInstancesProcessor
from contextgem.internal.base.concepts import _Concept
from contextgem.internal.typings.validators import _validate_sequence_is_list
from contextgem.public.aspects import Aspect


class ExtractionPipeline(_AssignedInstancesProcessor):
    """
    Represents a reusable collection of predefined aspects and concepts for document analysis.

    Extraction pipelines serve as templates that can be assigned to multiple documents,
    ensuring consistent application of the same analysis criteria. They package common
    extraction patterns into reusable units, allowing for standardized document processing.

    :ivar aspects: A list of aspects to extract from documents. Aspects represent structural
                  categories of information. Defaults to an empty list.
    :vartype aspects: list[Aspect]
    :ivar concepts: A list of concepts to identify within documents. Concepts represent
                   specific information elements to extract. Defaults to an empty list.
    :vartype concepts: list[_Concept]

    Note:
        A pipeline is a reusable configuration of extraction steps. You can use the same pipeline
        to extract data from multiple documents.

    Example:
        .. literalinclude:: ../../../dev/usage_examples/docstrings/pipelines/def_pipeline.py
            :language: python
            :caption: Extraction pipeline definition
    """

    aspects: list[Aspect] = Field(default_factory=list)
    concepts: Annotated[
        Sequence[_Concept], BeforeValidator(_validate_sequence_is_list)
    ] = Field(
        default_factory=list
    )  # using Sequence field with list validator for type checking


# TODO: Remove this class in v1.0.0.
class DocumentPipeline(ExtractionPipeline):
    """
    Deprecated alias for ExtractionPipeline.

    .. deprecated:: 0.14.1
        DocumentPipeline is deprecated and will be removed in v1.0.0.
        Use ExtractionPipeline instead.

    This class was renamed to ExtractionPipeline to better reflect its purpose and scope:

    * **Clearer semantics**: "ExtractionPipeline" explicitly describes what the pipeline does
    * **Consistency**: Aligns with the framework's naming conventions for extraction-focused components

    **Migration**: Simply replace ``DocumentPipeline`` with ``ExtractionPipeline`` in your imports.
    All functionality remains identical.

    **Future-proofing**: Update your code before v1.0.0 to avoid breaking changes:
    """

    def __init__(self, **data: Any) -> None:
        """
        Initialize DocumentPipeline with deprecation warning.
        """
        warnings.warn(
            "DocumentPipeline is deprecated and will be removed in v1.0.0. "
            "Use ExtractionPipeline instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(**data)
