# SPDX-FileCopyrightText: Shcherbak AI AS
# SPDX-License-Identifier: Apache-2.0

"""
Module defining the base classes for example subclasses.
"""

from __future__ import annotations

from typing import Any

from pydantic import Field

from contextgem.internal.base.instances import _InstanceBase
from contextgem.internal.decorators import _disable_direct_initialization
from contextgem.internal.typings.types import JSONDictField, NonEmptyStr


class _Example(_InstanceBase):
    """
    Internal implementation of the Example class.
    """

    content: Any = Field(
        ..., description="Arbitrary content associated with the example."
    )


@_disable_direct_initialization
class _StringExample(_Example):
    """
    Internal implementation of the StringExample class.
    """

    content: NonEmptyStr = Field(
        ...,
        description="A non-empty string that holds the text content of the extracted item example.",
    )


@_disable_direct_initialization
class _JsonObjectExample(_Example):
    """
    Internal implementation of the JsonObjectExample class.
    """

    content: JSONDictField = Field(
        ...,
        min_length=1,
        description="A JSON-serializable dict that holds the content of the extracted item example.",
    )
