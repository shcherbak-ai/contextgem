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
Module that defines base classes for extraction pipeline subclasses.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import Annotated, Any

from pydantic import BeforeValidator, Field

from contextgem.internal.base.aspects import _Aspect
from contextgem.internal.base.attrs import _AssignedInstancesProcessor
from contextgem.internal.base.concepts import _Concept
from contextgem.internal.decorators import _disable_direct_initialization
from contextgem.internal.typings.validators import _validate_sequence_is_list


@_disable_direct_initialization
class _ExtractionPipeline(_AssignedInstancesProcessor):
    """
    Internal implementation of the ExtractionPipeline class.
    """

    aspects: Annotated[
        Sequence[_Aspect], BeforeValidator(_validate_sequence_is_list)
    ] = Field(
        default_factory=list,
        description="Aspects to extract; define structural categories of information.",
    )  # using Sequence field with list validator for type checking
    concepts: Annotated[
        Sequence[_Concept], BeforeValidator(_validate_sequence_is_list)
    ] = Field(
        default_factory=list,
        description="Concepts to extract; specific data points.",
    )  # using Sequence field with list validator for type checking


# TODO: Remove this class in v1.0.0.
@_disable_direct_initialization
class _DocumentPipeline(_ExtractionPipeline):
    """
    Internal implementation of the DocumentPipeline class.
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
