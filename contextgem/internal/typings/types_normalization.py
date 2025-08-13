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
Utilities for normalizing Python type annotations into consistent forms.

Provides functions to standardize type hints across different typing notations
(e.g., converting typing.List[str] to list[str]), while ensuring compatibility
with JSON serialization requirements.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union, get_args, get_origin  # noqa: UP035


def _normalize_type_annotation(tp: Any) -> Any:
    """
    Normalizes type annotations to use consistent forms, converting
    between typing module versions and built-in generics.

    For example, typing.List[str] becomes list[str], typing.Dict[str, int]
    becomes dict[str, int], etc.

    Note: Only JSON-serializable types are supported (int, float, str, bool,
    None, list, dict). Tuples and sets are not supported as they are not
    JSON-serializable.

    :param tp: Type annotation to normalize
    :return: Normalized type annotation
    :raises ValueError: If a non-JSON-serializable type is encountered
    """
    # Handle basic types
    if tp in (str, int, float, bool, type(None)):
        return tp

    # Handle direct typing module types (without args)
    if tp is List:  # noqa: UP006
        return list
    if tp is Dict:  # noqa: UP006
        return dict

    # Get origin and arguments of the type
    origin = get_origin(tp)
    args = get_args(tp)

    # No origin or args means it's a plain type
    if origin is None:
        return tp

    # Direct mapping for common roots
    origin_map = {
        List: list,  # noqa: UP006
        Dict: dict,  # noqa: UP006
    }

    # Unsupported types check
    if origin in (tuple, set) or origin.__name__ in ("Tuple", "Set"):
        raise ValueError(
            f"Type {tp} is not supported as it's not JSON-serializable. "
            f"Use list instead of tuple/set in JSON structures."
        )

    # Always standardize origin to built-in equivalent if possible
    normalized_origin = origin_map.get(origin, origin)

    # Handle specific collection types
    if normalized_origin is list:
        if not args:
            return list
        if len(args) != 1:
            raise ValueError(
                f"List type annotation must have exactly one type argument, got {len(args)}: {tp}"
            )
        return list[_normalize_type_annotation(args[0])]

    elif normalized_origin is dict:
        if not args or len(args) != 2:
            return dict
        return dict[
            _normalize_type_annotation(args[0]), _normalize_type_annotation(args[1])
        ]

    elif origin is Union or origin is Optional:
        if not args:
            return Union if origin is Union else Optional

        # Normalize all arguments
        normalized_args = tuple(_normalize_type_annotation(arg) for arg in args)
        if len(normalized_args) == 0:
            raise ValueError(
                f"Union type annotation must have at least one type argument, got {len(args)}: {tp}"
            )

        # Optional is just Union with NoneType, so standardize to Union
        if origin is Optional:
            if type(None) not in normalized_args:
                normalized_args = normalized_args + (type(None),)
            if len(normalized_args) == 1:
                return normalized_args[0]
            else:
                return Union[normalized_args]  # noqa: UP007

        # Handle Union
        if len(normalized_args) == 1:
            return normalized_args[0]
        if len(normalized_args) == 2:
            return Union[normalized_args[0], normalized_args[1]]  # noqa: UP007
        # For more than two types, we need to use __getitem__ with tuple
        # Type checker doesn't recognize Union.__getitem__ method, which works at runtime
        return Union.__getitem__(normalized_args)  # type: ignore

    # For other types with origin/args, keep the general structure
    # but normalize the arguments
    if args:
        normalized_args = tuple(_normalize_type_annotation(arg) for arg in args)
        try:
            return (
                normalized_origin[normalized_args]
                if len(normalized_args) == 1
                else normalized_origin.__getitem__(normalized_args)
            )
        except (TypeError, AttributeError):
            # If we can't create a new generic with normalized args, keep the original
            return tp

    # Fallback
    return tp
