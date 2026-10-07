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
Module defining base classes for data validation models.
"""

from __future__ import annotations

from pydantic import ConfigDict, Field, StrictFloat

from contextgem.internal.base.serialization import _InstanceSerializer
from contextgem.internal.decorators import _disable_direct_initialization


@_disable_direct_initialization
class _LLMPricing(_InstanceSerializer):
    """
    Internal implementation of the LLMPricing class.
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

    model_config = ConfigDict(
        extra="forbid", frozen=True
    )  # make immutable once created
