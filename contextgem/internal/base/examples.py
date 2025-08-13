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
Module defining the base classes for example subclasses.
"""

from __future__ import annotations

from typing import Any

from pydantic import Field, field_validator

from contextgem.internal.base.instances import _InstanceBase
from contextgem.internal.decorators import _disable_direct_initialization
from contextgem.internal.typings.aliases import NonEmptyStr
from contextgem.internal.utils import _is_json_serializable


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

    content: dict[str, Any] = Field(
        ...,
        min_length=1,
        description="A JSON-serializable dict that holds the content of the extracted item example.",
    )

    @field_validator("content")
    @classmethod
    def _validate_content_serializable(cls, value: dict[str, Any]) -> dict[str, Any]:
        """
        Validates that the `content` field is serializable to JSON.

        :param value: The value of the `content` field to validate.
        :type value: dict[str, Any]
        :return: The validated `content` value.
        :rtype: dict[str, Any]
        :raises ValueError: If the `content` value is not serializable.
        """
        if not _is_json_serializable(value):
            raise ValueError("`content` must be JSON serializable.")
        return value
