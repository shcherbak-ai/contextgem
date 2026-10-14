# SPDX-FileCopyrightText: Shcherbak AI AS
# SPDX-License-Identifier: Apache-2.0

"""
Module defining base classes for Image class.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from contextgem.internal.base.instances import _InstanceBase
from contextgem.internal.decorators import _disable_direct_initialization
from contextgem.internal.typings.types import NonEmptyStr


@_disable_direct_initialization
class _Image(_InstanceBase):
    """
    Internal implementation of the Image class.
    """

    mime_type: Literal["image/jpg", "image/jpeg", "image/png", "image/webp"] = Field(
        ..., description="The MIME type of the image."
    )
    base64_data: NonEmptyStr = Field(
        ...,
        description="The base64-encoded data of the image.",
        repr=False,  # do not show in repr due to the excessive base64 string length
    )
