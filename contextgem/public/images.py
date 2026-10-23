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
    :ivar custom_data: A dictionary for data of your own about the image, e.g. the ID
        that your application has for it. ``Document.from_docling()`` stores where
        the image comes from under the key ``"docling"``. Must be JSON-serializable.
        Not sent to the models, and kept when the image is serialized or cloned.
        Defaults to an empty dict.
    :vartype custom_data: dict[str, Any]
    :ivar unique_id: The unique ID of the image, a ULID assigned when the image is
        created. Read-only. Kept when the image is serialized or cloned, i.e. a restored
        or cloned image has the ID of the original.
    :vartype unique_id: str

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

        .. The example above is not in the installed package. It is shown at
           https://contextgem.shcherbak.ai/api/images/index.html.md
    """
