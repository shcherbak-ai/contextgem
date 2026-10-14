# SPDX-FileCopyrightText: Shcherbak AI AS
# SPDX-License-Identifier: Apache-2.0

"""
Module that defines base classes for extraction pipeline subclasses.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Annotated

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
