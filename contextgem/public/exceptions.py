# SPDX-FileCopyrightText: Shcherbak AI AS
# SPDX-License-Identifier: Apache-2.0

"""
Module exposing the exceptions that the framework raises to the calling code,
the exception that a tool function raises for the LLM, and the category of
the warnings that the framework issues.

The exceptions are the classes that the framework itself raises, so they can be
used in ``except`` clauses: ``from contextgem import ModelAPIError``, or
``ContextGemError``, their base class, to catch any of them. ``ToolCallError`` is
raised by a tool function, to send its message to the LLM in a chat. Likewise,
the warning category is the one of the warnings that the framework issues, so it
can be used in warning filters: ``from contextgem import ContextGemWarning``, or
``UnknownPricesWarning``, its subclass, for the warning on a model whose costs
cannot be calculated.
"""

from __future__ import annotations

from contextgem.internal.exceptions import (
    ContextGemError,
    ContextGemWarning,
    LLMToolLoopLimitError,
    ModelAPIError,
    ModelExtractionError,
    ModelRefusalError,
    ToolCallError,
    UnknownPricesWarning,
)


__all__ = (  # noqa: RUF022 - the order of the API reference
    "ContextGemError",
    "ModelAPIError",
    "ModelExtractionError",
    "ModelRefusalError",
    "LLMToolLoopLimitError",
    "ToolCallError",
    "ContextGemWarning",
    "UnknownPricesWarning",
)
