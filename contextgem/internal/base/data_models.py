# SPDX-FileCopyrightText: Shcherbak AI AS
# SPDX-License-Identifier: Apache-2.0

"""
Module defining base classes for data validation models.
"""

from __future__ import annotations

from pydantic import ConfigDict, Field, StrictFloat

from contextgem.internal.base.serialization import _InstanceSerializer
from contextgem.internal.decorators import _disable_direct_initialization


@_disable_direct_initialization
class _ModelPricing(_InstanceSerializer):
    """
    Internal implementation of the ModelPricing class.
    """

    input_per_1m_tokens: StrictFloat = Field(
        ...,
        ge=0,
        description="Cost for processing 1 million input tokens.",
    )
    output_per_1m_tokens: StrictFloat = Field(
        ...,
        ge=0,
        description="Cost for generating 1 million output tokens.",
    )
    cache_read_per_1m_tokens: StrictFloat | None = Field(
        default=None,
        ge=0,
        description=(
            "Cost for processing 1 million input tokens read from the prompt cache. "
            "If not set, such tokens are priced as regular input tokens."
        ),
    )
    cache_write_per_1m_tokens: StrictFloat | None = Field(
        default=None,
        ge=0,
        description=(
            "Cost for processing 1 million input tokens written to the prompt cache. "
            "If not set, such tokens are priced as regular input tokens."
        ),
    )

    model_config = ConfigDict(
        extra="forbid", frozen=True
    )  # make immutable once created
