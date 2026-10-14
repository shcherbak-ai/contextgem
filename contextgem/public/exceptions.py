# SPDX-FileCopyrightText: Shcherbak AI AS
# SPDX-License-Identifier: Apache-2.0

"""
Module exposing the exceptions that the framework raises to the calling code.

The exceptions are the classes that the framework itself raises, so they can be
used in ``except`` clauses: ``from contextgem import ModelAPIError``.
"""

from __future__ import annotations

from contextgem.internal.exceptions import (
    LLMToolLoopLimitError,
    ModelAPIError,
    ModelExtractionError,
)


__all__ = (
    "ModelAPIError",
    "ModelExtractionError",
    "LLMToolLoopLimitError",
)
