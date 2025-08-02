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
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, BinaryIO, get_type_hints

from PIL import Image as PILImage

from contextgem.internal.loggers import _configure_logger_from_env
from contextgem.internal.typings.strings_to_types import (
    PRIMITIVE_TYPES_STRING_MAP_REVERSED,
    _deserialize_type_hint,
)
from contextgem.internal.typings.typed_class_utils import _raise_dict_class_type_error
from contextgem.internal.typings.types_normalization import _normalize_type_annotation
from contextgem.internal.typings.types_to_strings import (
    JSON_PRIMITIVE_TYPES,
    _is_json_serializable_type,
    _raise_json_serializable_type_error,
)


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

    # Handle different input types
    if isinstance(source, PILImage.Image):
        # PIL Image object
        pil_image = source
        image_format = pil_image.format
        if image_format is None:
            raise ValueError(
                "Cannot determine image format from PIL Image object. "
                "Please save the image with a specific format first."
            )
    elif isinstance(source, str | Path):
        # File path
        image_path = Path(source)
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")

        try:
            pil_image = PILImage.open(image_path)
            image_format = pil_image.format
        except Exception as e:
            raise OSError(f"Cannot open image file {image_path}: {e}") from e
    elif isinstance(source, bytes):
        # Raw bytes data
        try:
            pil_image = PILImage.open(io.BytesIO(source))
            image_format = pil_image.format
        except Exception as e:
            raise OSError(f"Cannot open image from bytes data: {e}") from e
    else:
        # File-like object (BinaryIO, BytesIO, file handles, etc.)
        try:
            pil_image = PILImage.open(source)
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

    # Convert to base64 using the image_to_base64 function
    buffer = io.BytesIO()
    try:
        pil_image.save(buffer, format=image_format)
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


class JsonObjectClassStruct:
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

    # Registry to store all subclasses for type resolution during structure generation:
    # 1. Resolves string type annotations and forward references
    # 2. Enables lookup of already defined classes by name when traversing class hierarchies
    # 3. Provides a shared namespace for the entire type structure
    # 4. Critical for resolving classes used in generic containers like list[Class]
    _registry = {}

    def __init_subclass__(cls, **kwargs):
        """
        Registers each subclass in the registry
        """
        super().__init_subclass__(**kwargs)
        JsonObjectClassStruct._registry[cls.__name__] = cls

    @classmethod
    def _as_dict_structure(cls) -> dict[str, Any]:
        """
        Converts class hierarchy to a dictionary structure for JsonObjectConcept.

        :return: Dictionary representation of the class structure suitable for JsonObjectConcept.
        :rtype: dict[str, Any]
        """
        result = {}

        # Get annotations using get_type_hints to resolve forward references
        # This correctly handles Pydantic and other typed classes
        try:
            annotations = get_type_hints(cls)
        except Exception:
            # Fallback to raw annotations if get_type_hints fails
            annotations = cls.__annotations__ if hasattr(cls, "__annotations__") else {}

        # Create a combined namespace for resolving types
        namespace = {}
        namespace.update(globals())  # Start with global namespace
        namespace.update(JsonObjectClassStruct._registry)  # Add registered classes

        for field_name, field_type in annotations.items():
            # Normalize field type to handle generic types consistently
            field_type = _normalize_type_annotation(field_type)

            # Process based on field_type
            processed_field_type = cls._process_field_type(
                field_type, field_name, namespace
            )

            # Validate that the processed field type is JSON serializable
            if not _is_json_serializable_type(processed_field_type):
                _raise_json_serializable_type_error(
                    processed_field_type, field_name=field_name
                )

            result[field_name] = processed_field_type

        return result

    @classmethod
    def _process_field_type(
        cls, field_type: Any, field_name: str, namespace: dict[str, Any]
    ) -> Any:
        """
        Processes a field type and converts it to the appropriate dictionary structure.

        :param field_type: The type annotation to process
        :type field_type: Any
        :param field_name: The name of the field
        :type field_name: str
        :param namespace: The namespace for resolving types
        :type namespace: dict[str, Any]
        :return: Appropriate representation of the type for dictionary structure
        :rtype: Any
        """
        # Case 1: String type annotations (forward references or type strings)
        if isinstance(field_type, str):
            return cls._process_string_type(field_type, field_name, namespace)

        # Case 2: Types in registry (JsonObjectClassStruct subclasses)
        if (
            hasattr(field_type, "__name__")
            and field_type.__name__ in JsonObjectClassStruct._registry
        ):
            return JsonObjectClassStruct._registry[
                field_type.__name__
            ]._as_dict_structure()

        # Case 3: List types
        if hasattr(field_type, "__origin__") and field_type.__origin__ is list:
            return cls._process_list_type(field_type, field_name, namespace)

        # Case 4: Dictionary types
        if hasattr(field_type, "__origin__") and field_type.__origin__ is dict:
            return cls._process_dict_type(field_type, field_name, namespace)

        # Default case: Use the field type directly
        return field_type

    @classmethod
    def _process_string_type(
        cls, field_type: str, field_name: str, namespace: dict[str, Any]
    ) -> Any:
        """
        Processes a string type annotation

        :param field_type: The type annotation to process
        :type field_type: str
        :param field_name: The name of the field
        :type field_name: str
        :param namespace: The namespace for resolving types
        :type namespace: dict[str, Any]
        :return: Appropriate representation of the type for dictionary structure
        :rtype: Any
        """

        # First try using the general type parser for all type annotations
        try:
            return _deserialize_type_hint(field_type)
        except ValueError:
            # If that fails, continue with the rest of the logic
            pass

        # Handle list type syntax in strings with regex
        list_match = re.compile(r"list\[(.*)\]").match(field_type)
        if list_match:
            inner_type_name = list_match.group(1).strip()
            if inner_type_name in JsonObjectClassStruct._registry:
                inner_class = JsonObjectClassStruct._registry[inner_type_name]
                return [inner_class._as_dict_structure()]

        # Try to resolve the type from our namespace
        if field_type in namespace:
            resolved_type = namespace[field_type]
            if hasattr(resolved_type, "_as_dict_structure"):
                return resolved_type._as_dict_structure()
            return resolved_type

        # Resolve primitive type by name
        if field_type in PRIMITIVE_TYPES_STRING_MAP_REVERSED:
            return PRIMITIVE_TYPES_STRING_MAP_REVERSED[field_type]

        # If we can't resolve the type, raise an error
        raise ValueError(
            f"Could not resolve type '{field_type}' for field '{field_name}'. "
            f"Make sure the type is either a built-in type or registered "
            f"with JsonObjectClassStruct."
        )

    @classmethod
    def _process_list_type(
        cls, field_type: Any, field_name: str, namespace: dict[str, Any]
    ) -> Any:
        """
        Processes a list type annotation

        :param field_type: The type annotation to process
        :type field_type: Any
        :param field_name: The name of the field
        :type field_name: str
        :param namespace: The namespace for resolving types
        :type namespace: dict[str, Any]
        :return: Appropriate representation of the type for dictionary structure
        :rtype: Any
        """
        # Validate list type has exactly one type argument
        if len(field_type.__args__) != 1:
            raise ValueError(
                f"List type annotation for '{field_name}' must have "
                f"exactly one type argument, got {len(field_type.__args__)}"
            )

        # Get the item type
        item_type = field_type.__args__[0]

        # Check if item_type is registered by name in our registry
        if (
            hasattr(item_type, "__name__")
            and item_type.__name__ in JsonObjectClassStruct._registry
        ):
            resolved_class = JsonObjectClassStruct._registry[item_type.__name__]
            return [resolved_class._as_dict_structure()]

        # Handle string item types
        if isinstance(item_type, str):
            # Try to resolve from namespace
            if item_type in namespace:
                resolved_item_type = namespace[item_type]
                if hasattr(resolved_item_type, "_as_dict_structure"):
                    return [resolved_item_type._as_dict_structure()]
                return [resolved_item_type]
            # Try to resolve primitive type
            if item_type in PRIMITIVE_TYPES_STRING_MAP_REVERSED:
                return [PRIMITIVE_TYPES_STRING_MAP_REVERSED[item_type]]
            # Can't resolve - raise error
            raise ValueError(
                f"Could not resolve list item type '{item_type}' for field '{field_name}'. "
                f"Make sure the type is either a built-in type or registered "
                f"with JsonObjectClassStruct."
            )

        # Default case: keep the list type as is
        return field_type

    @classmethod
    def _process_dict_type(
        cls, field_type: Any, field_name: str, namespace: dict[str, Any]
    ) -> Any:
        """
        Processes a dictionary type annotation

        :param field_type: The type annotation to process
        :type field_type: Any
        :param field_name: The name of the field
        :type field_name: str
        :param namespace: The namespace for resolving types
        :type namespace: dict[str, Any]
        :return: Appropriate representation of the type for dictionary structure
        :rtype: Any
        """

        # Validate dictionary has key and value type arguments
        if len(field_type.__args__) != 2:
            return field_type

        key_type, value_type = field_type.__args__

        # Check if key type is str (required for dictionaries)
        if key_type is not str:
            raise TypeError(
                f"Invalid key type for dictionary field '{field_name}': {key_type}.\n"
                f"Dictionary keys must be strings."
            )

        # Check if key or value type is a class that should be disallowed
        for type_arg, type_name in [(key_type, "key"), (value_type, "value")]:
            if (
                hasattr(type_arg, "__module__")
                and hasattr(type_arg, "__name__")
                and not isinstance(type_arg, type)
            ):
                is_basic_type = type_arg in JSON_PRIMITIVE_TYPES
                if not is_basic_type and not (
                    hasattr(type_arg, "__origin__") or isinstance(type_arg, str)
                ):
                    _raise_dict_class_type_error(type_name, field_name, type_arg)
            elif isinstance(type_arg, str):
                if type_arg in namespace:
                    resolved_type = namespace[type_arg]
                    if hasattr(resolved_type, "_as_dict_structure"):
                        _raise_dict_class_type_error(type_name, field_name, type_arg)

        # Keep dictionary type as is - structure is defined by key/value types
        return field_type
