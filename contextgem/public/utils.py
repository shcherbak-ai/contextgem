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
Module defining public utility functions and classes of the framework.
"""

from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import TYPE_CHECKING, BinaryIO

from PIL import Image as PILImage

from contextgem.internal.base.utils import _JsonObjectClassStruct
from contextgem.internal.loggers import _configure_logger_from_env


if TYPE_CHECKING:
    from contextgem.public.images import Image


def image_to_base64(source: str | Path | BinaryIO | bytes) -> str:
    """
    Converts an image to its Base64 encoded string representation.

    Helper function that can be used when constructing ``Image`` objects.

    :param source: The image source - can be a file path (str or Path),
        file-like object (BytesIO, file handle, etc.), or raw bytes data.
    :type source: str | Path | BinaryIO | bytes
    :return: A Base64 encoded string representation of the image.
    :rtype: str
    :raises FileNotFoundError: If the image file path does not exist.
    :raises OSError: If the image cannot be read.

    Example:
        >>> from pathlib import Path
        >>> import io
        >>>
        >>> # From file path
        >>> base64_str = image_to_base64("path/to/image.jpg")
        >>>
        >>> # From file handle
        >>> with open("image.png", "rb") as f:
        ...     base64_str = image_to_base64(f)
        >>>
        >>> # From bytes data
        >>> with open("image.webp", "rb") as f:
        ...     image_bytes = f.read()
        >>> base64_str = image_to_base64(image_bytes)
        >>>
        >>> # From BytesIO
        >>> buffer = io.BytesIO(image_bytes)
        >>> base64_str = image_to_base64(buffer)
    """
    if isinstance(source, str | Path):
        # File path
        image_path = Path(source)
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode("utf-8")
        except Exception as e:
            raise OSError(f"Cannot read image file {image_path}: {e}") from e
    elif isinstance(source, bytes):
        # Raw bytes data
        return base64.b64encode(source).decode("utf-8")
    else:
        # File-like object (BinaryIO, BytesIO, file handles, etc.)
        try:
            return base64.b64encode(source.read()).decode("utf-8")
        except Exception as e:
            raise OSError(f"Cannot read from file-like object: {e}") from e


def create_image(source: str | Path | PILImage.Image | BinaryIO | bytes) -> Image:
    """
    Creates an Image instance from various image sources.

    This function automatically determines the MIME type and converts the image to base64
    format using Pillow functionality. It supports common image formats including JPEG,
    PNG, and WebP.

    :param source: The image source - can be a file path (str or Path), PIL Image object,
        file-like object (BytesIO, file handle, etc.), or raw bytes data.
    :type source: str | Path | PILImage.Image | BinaryIO | bytes
    :return: An Image instance with the appropriate MIME type and base64 data.
    :rtype: Image
    :raises ValueError: If the image format is not supported or cannot be determined.
    :raises FileNotFoundError: If the image file path does not exist.
    :raises OSError: If the image cannot be opened or processed.

    Example:
        >>> from pathlib import Path
        >>> from PIL import Image as PILImage
        >>> import io
        >>>
        >>> # From file path
        >>> img = create_image("path/to/image.jpg")
        >>>
        >>> # From PIL Image object
        >>> pil_img = PILImage.open("path/to/image.png")
        >>> img = create_image(pil_img)
        >>>
        >>> # From file-like object
        >>> with open("image.jpg", "rb") as f:
        ...     img = create_image(f)
        >>>
        >>> # From bytes data
        >>> with open("image.png", "rb") as f:
        ...     image_bytes = f.read()
        >>> img = create_image(image_bytes)
        >>>
        >>> # From BytesIO
        >>> buffer = io.BytesIO(image_bytes)
        >>> img = create_image(buffer)
    """

    # Avoid circular imports
    from contextgem.public.images import Image

    # Format mapping from PIL format to MIME type
    format_to_mime = {
        "JPEG": "image/jpeg",
        "JPG": "image/jpg",
        "PNG": "image/png",
        "WEBP": "image/webp",
    }

    # Handle different input types and get original bytes when possible
    # Note: We avoid using PIL compression for non-PIL sources to prevent cross-platform
    # issues (different compression algorithms, library versions, platform-specific defaults)
    original_bytes = None

    if isinstance(source, PILImage.Image):
        # PIL Image object - we need to use PIL since we don't have original bytes
        pil_image = source
        image_format = pil_image.format
        if image_format is None:
            raise ValueError(
                "Cannot determine image format from PIL Image object. "
                "Please save the image with a specific format first."
            )
        # For PIL images, we'll need to save to buffer to get bytes

    elif isinstance(source, str | Path):
        # File path - read original bytes directly, use PIL only for format detection
        image_path = Path(source)
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")

        try:
            # Read original bytes
            original_bytes = image_path.read_bytes()
            # Use PIL only to detect format
            pil_image = PILImage.open(io.BytesIO(original_bytes))
            image_format = pil_image.format
        except Exception as e:
            raise OSError(f"Cannot open image file {image_path}: {e}") from e

    elif isinstance(source, bytes):
        # Raw bytes data - use directly
        original_bytes = source
        try:
            # Use PIL only to detect format
            pil_image = PILImage.open(io.BytesIO(original_bytes))
            image_format = pil_image.format
        except Exception as e:
            raise OSError(f"Cannot open image from bytes data: {e}") from e

    else:
        # File-like object - read original bytes directly
        try:
            # Read original bytes
            original_bytes = source.read()
            # Use PIL only to detect format
            pil_image = PILImage.open(io.BytesIO(original_bytes))
            image_format = pil_image.format
        except Exception as e:
            raise OSError(f"Cannot open image from file-like object: {e}") from e

    # Convert format to MIME type
    if image_format not in format_to_mime:
        raise ValueError(
            f"Unsupported image format: {image_format}. "
            f"Supported formats: {', '.join(format_to_mime.keys())}"
        )

    mime_type = format_to_mime[image_format]

    # Convert to base64 - use original bytes when available to avoid recompression
    if original_bytes is not None:
        # Use original bytes directly - no compression, no cross-platform issues
        base64_data = image_to_base64(original_bytes)
    else:
        # For PIL Image objects - save to buffer to get bytes
        buffer = io.BytesIO()
        try:
            pil_image.save(buffer, format=image_format, optimize=False)
            buffer.seek(0)
            base64_data = image_to_base64(buffer)
        except Exception as e:
            raise OSError(f"Cannot convert image to base64: {e}") from e
        finally:
            buffer.close()

    return Image(mime_type=mime_type, base64_data=base64_data)  # type: ignore[arg-type]


def reload_logger_settings():
    """
    Reloads logger settings from environment variables.

    This function should be called when environment variables related to logging
    have been changed after the module was imported. It re-reads the environment
    variables and reconfigures the logger accordingly.

    :return: None

    Example:
        .. literalinclude:: ../../../dev/usage_examples/docstrings/utils/reload_logger_settings.py
            :language: python
            :caption: Reload logger settings
    """
    _configure_logger_from_env()


class JsonObjectClassStruct(_JsonObjectClassStruct):
    """
    A base class that automatically converts class hierarchies to dictionary representations.

    This class enables the use of existing class hierarchies (such as dataclasses or Pydantic models)
    with nested type hints as a structure definition for JsonObjectConcept. When you need to use
    typed class hierarchies with JsonObjectConcept, inherit from this class in all parts of your
    class structure.

    Example:
        .. literalinclude:: ../../../dev/usage_examples/docstrings/utils/json_object_cls_struct.py
            :language: python
            :caption: Using JsonObjectClassStruct for class hierarchies
    """

    pass
