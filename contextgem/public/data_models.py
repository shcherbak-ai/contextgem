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

from __future__ import annotations

from contextgem.internal.base.data_models import _LLMPricing
from contextgem.internal.decorators import _expose_in_registry


@_expose_in_registry(additional_key=_LLMPricing)  # ty: ignore[too-many-positional-arguments]
class LLMPricing(_LLMPricing):
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

    pass
