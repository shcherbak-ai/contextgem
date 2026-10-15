# SPDX-FileCopyrightText: Shcherbak AI AS
# SPDX-License-Identifier: Apache-2.0

"""
Module exposing the exceptions that the framework raises to the calling code,
and the category of the warnings that it issues.

The exceptions are the classes that the framework itself raises, so they can be
used in ``except`` clauses: ``from contextgem import ModelAPIError``. Likewise,
the warning category is the one of the warnings that the framework issues, so it
can be used in warning filters: ``from contextgem import ContextGemWarning``.
"""

from __future__ import annotations

from contextgem.internal.exceptions import (
    ContextGemWarning,
    LLMToolLoopLimitError,
    ModelAPIError,
    ModelExtractionError,
)


__all__ = (  # noqa: RUF022 - the order of the API reference
    "ModelAPIError",
    "ModelExtractionError",
    "LLMToolLoopLimitError",
    "ContextGemWarning",
)
