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
Module defining public data validation models.
"""

import warnings

from pydantic import ConfigDict, Field, StrictFloat, StrictInt, model_validator
from typing_extensions import Self

from contextgem.internal.base.serialization import _InstanceSerializer


class LLMPricing(_InstanceSerializer):
    """
    Represents the pricing details for an LLM.

    Defines the cost structure for processing input tokens and generating output tokens,
    with prices specified per million tokens.

    :ivar input_per_1m_tokens: The cost in currency units for processing 1M input tokens.
    :vartype input_per_1m_tokens: StrictFloat
    :ivar output_per_1m_tokens: The cost in currency units for generating 1M output tokens.
    :vartype output_per_1m_tokens: StrictFloat

    Example:
        .. literalinclude:: ../../../dev/usage_examples/docstrings/data_models/def_llm_pricing.py
            :language: python
            :caption: LLM pricing definition
    """

    input_per_1m_tokens: StrictFloat = Field(
        ...,
        ge=0,
    )
    output_per_1m_tokens: StrictFloat = Field(
        ...,
        ge=0,
    )

    model_config = ConfigDict(
        extra="forbid", frozen=True
    )  # make immutable once created


# TODO: remove this class in v1.0.0.
class RatingScale(_InstanceSerializer):
    """
    Represents a rating scale with defined minimum and maximum values.

    .. deprecated:: 0.10.0
        RatingScale is deprecated and will be removed in v1.0.0.
        Use a tuple of (start, end) integers instead, e.g. (1, 5)
        instead of RatingScale(start=1, end=5).

    This class defines a numerical scale for rating concepts, with configurable
    start and end values that determine the valid range for ratings.

    :ivar start: The minimum value of the rating scale (inclusive).
                 Must be greater than or equal to 0.
    :vartype start: StrictInt
    :ivar end: The maximum value of the rating scale (inclusive).
              Must be greater than 0.
    :vartype end: StrictInt
    """

    start: StrictInt = Field(ge=0, default=0)
    end: StrictInt = Field(gt=0, default=10)

    model_config = ConfigDict(
        extra="forbid", frozen=True
    )  # make immutable once created

    def __init__(self, **data):
        """
        Initialize RatingScale with deprecation warning.
        """
        warnings.warn(
            "RatingScale is deprecated and will be removed in v1.0.0. "
            "Use a tuple of (start, end) integers instead, e.g. (1, 5) "
            "instead of RatingScale(start=1, end=5).",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(**data)

    @model_validator(mode="after")
    def _validate_rating_scale_post(self) -> Self:
        """
        Validates that the end value is greater than the start value.

        :return: The validated model instance.
        :rtype: Self
        :raises ValueError: If the end value is not greater than the start value.
        """
        if self.end <= self.start:
            raise ValueError(
                f"Invalid rating scale: end value ({self.end}) must be greater than start value ({self.start})"
            )
        return self
