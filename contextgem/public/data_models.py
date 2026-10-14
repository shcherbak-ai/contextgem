# SPDX-FileCopyrightText: Shcherbak AI AS
# SPDX-License-Identifier: Apache-2.0

"""
Module defining public data validation models.
"""

from __future__ import annotations

from contextgem.internal.base.data_models import _ModelPricing
from contextgem.internal.decorators import _expose_in_registry


@_expose_in_registry(additional_key=_ModelPricing)  # ty: ignore[too-many-positional-arguments]
class ModelPricing(_ModelPricing):
    """
    Represents the pricing details for a model.

    Defines the cost structure for processing input tokens and generating output tokens,
    with prices specified per million tokens. Optionally, defines separate prices for
    the input tokens that the model provider reads from, or writes to, its prompt cache.

    :ivar input_per_1m_tokens: The cost in currency units for processing 1M input tokens.
    :vartype input_per_1m_tokens: StrictFloat
    :ivar output_per_1m_tokens: The cost in currency units for generating 1M output tokens.
    :vartype output_per_1m_tokens: StrictFloat
    :ivar cache_read_per_1m_tokens: The cost in currency units for processing 1M input tokens
        read from the prompt cache (cached input tokens), which providers usually price lower
        than regular input tokens. Defaults to None, in which case such tokens are priced
        at ``input_per_1m_tokens``.
    :vartype cache_read_per_1m_tokens: StrictFloat | None
    :ivar cache_write_per_1m_tokens: The cost in currency units for processing 1M input tokens
        written to the prompt cache, which some providers price higher than regular input
        tokens. Defaults to None, in which case such tokens are priced at ``input_per_1m_tokens``.
    :vartype cache_write_per_1m_tokens: StrictFloat | None

    Example:
        .. literalinclude:: ../../../dev/usage_examples/docstrings/data_models/def_model_pricing.py
            :language: python
            :caption: Model pricing definition
    """

    pass
