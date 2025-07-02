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
Module for handling document images.

This module provides the Image class, which represents visual content that can be attached to
or fully represent a document. Images are stored in base64-encoded format with specified MIME types
to ensure proper handling.

The module supports common image formats (JPEG, PNG, WebP) and integrates with the broader ContextGem
framework for document analysis that includes visual content alongside textual information.
"""

from __future__ import annotations

from typing import Literal

from contextgem.internal.base.instances import _InstanceBase
from contextgem.internal.typings.aliases import NonEmptyStr


class Image(_InstanceBase):
    """
    Represents an image with specified MIME type and base64-encoded data.
    An image is typically attached to a document, or fully represents a document.

    :ivar mime_type: The MIME type of the image. This must be one of the
        predefined valid types ("image/jpg", "image/jpeg", "image/png",
        "image/webp").
    :vartype mime_type: Literal["image/jpg", "image/jpeg", "image/png",
        "image/webp"]
    :ivar base64_data: The base64-encoded data of the image. The util function
        `image_to_base64()` from contextgem.public.utils can be used to encode images to base64.
    :vartype base64_data: NonEmptyStr

    Note:
        - Attached to documents:
            An image must be attached to a document. A document can have multiple images.

        - Extraction types:
            Only concept extraction is supported for images. Use LLM with role ``"extractor_vision"``
            or ``"reasoner_vision"`` to extract concepts from images.

    Example:
        .. literalinclude:: ../../../dev/usage_examples/docstrings/images/def_image.py
            :language: python
            :caption: Image definition
    """

    mime_type: Literal["image/jpg", "image/jpeg", "image/png", "image/webp"]
    base64_data: NonEmptyStr
