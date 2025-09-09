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
