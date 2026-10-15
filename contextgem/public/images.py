# SPDX-FileCopyrightText: Shcherbak AI AS
# SPDX-License-Identifier: Apache-2.0

"""
Module for handling document images.

This module provides the Image class, which represents visual content that can be attached to
or fully represent a document. Images are stored in base64-encoded format with specified MIME types
to ensure proper handling.
"""

from __future__ import annotations

from contextgem.internal.base.images import _Image
from contextgem.internal.decorators import _expose_in_registry


@_expose_in_registry(additional_key=_Image)
class Image(_Image):
    """
    Represents an image with specified MIME type and base64-encoded data.
    An image is typically attached to a document, or fully represents a document.

    Util function ``create_image()`` from ``contextgem.public.utils`` can be used to create
    an Image instance from various sources: file paths, PIL Image objects, file-like objects,
    or raw bytes data.

    :ivar mime_type: The MIME type of the image. This must be one of the predefined valid types
        ("image/jpg", "image/jpeg", "image/png", "image/webp").
    :vartype mime_type: Literal["image/jpg", "image/jpeg", "image/png", "image/webp"]
    :ivar base64_data: The base64-encoded data of the image. The util function
        ``image_to_base64()`` from ``contextgem.public.utils`` can be used to
        encode images to base64.
    :vartype base64_data: str

    Note:
        - Attached to documents:
            An image must be attached to a document. A document can have multiple images.

        - Extraction types:
            Only document-level concept extraction is supported for images. Set
            ``modality="image"`` or ``modality="multimodal"`` on a concept to extract it
            from images, and use an LLM that supports vision.

    Example:
        .. literalinclude:: ../../../dev/usage_examples/docstrings/images/def_image.py
            :language: python
            :caption: Image definition
    """
