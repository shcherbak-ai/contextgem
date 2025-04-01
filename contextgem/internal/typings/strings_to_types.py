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
Module for deserializing type annotations from string representations.

This module provides functionality to convert string representations of type hints
(either from prompts or serialized format) back into actual Python type annotations.
It works in conjunction with the types_to_strings module to provide bidirectional
conversion between type annotations and their string representations.
"""

from types import GenericAlias, UnionType
from typing import Union

from contextgem.internal.typings.types_to_strings import JSON_OBJECT_STRUCTURE_TYPES_MAP

JSON_OBJECT_STRUCTURE_TYPES_MAP_REVERSED = {
    v: k for k, v in JSON_OBJECT_STRUCTURE_TYPES_MAP.items()
}


def _skip_whitespace(s: str, i: int) -> int:
    """
    Skips over any whitespace characters in the provided string starting from the given index
    and returns the index of the first non-whitespace character. If no non-whitespace characters
    remain, it returns the length of the string.

    :param s: The input string to process.
    :type s: str
    :param i: The starting index within the string.
    :type i: int
    :return: The index of the first non-whitespace character after the starting position or the
        length of the string if no such character exists.
    :rtype: int
    """
    while i < len(s) and s[i].isspace():
        i += 1
    return i


def _parse_identifier(s: str, i: int) -> tuple[str, int]:
    """
    Parses an identifier from the given string starting from the specified index.

    This function checks for a valid identifier beginning at the given index within
    the input string. It increments the index as long as alphanumeric characters
    or underscores are encountered, thereby extracting the identifier. If no valid
    identifier is found at the starting index, a ValueError is raised.

    :param s: The input string containing the potential identifier.
    :type s: str
    :param i: The starting index in the string to search for the identifier.
    :type i: int
    :return: A tuple containing the parsed identifier and the updated index where
        parsing ended.
    :rtype: tuple
    :raises ValueError: If no valid identifier is found at the given index.
    """
    start = i
    while i < len(s) and (s[i].isalnum() or s[i] == "_"):
        i += 1
    if start == i:
        raise ValueError(f"Expected identifier at position {i} in {s!r}")
    return s[start:i], i


def _parse_type_hint(s: str, i: int = 0) -> tuple[type | GenericAlias | UnionType, int]:
    """
    Parses a serialized type hint string using a recursive descent parsing approach.

    The function takes a serialized string representing type hints and an optional starting
    index. It identifies and parses the type hint structure including basic types,
    generics such as `list` and `dict`, and unions. The function will validate the structure
    of the serialized input and return the corresponding type hint along with the new index
    position for further parsing.

    :param s: The serialized string representing the type hints.
    :param i: The starting index in the string for parsing the type hints.
    :return: A tuple, where the first element is the parsed type hint (which can be
        a basic type, a generic type, or a union type), and the second element is the
        updated index position.
    :raises ValueError: If the serialized string has an invalid structure, missing
        expected characters such as brackets or commas, or contains unknown type
        identifiers.
    """

    i = _skip_whitespace(s, i)
    ident, i = _parse_identifier(s, i)

    # If it is one of our base types, return it.
    if ident in JSON_OBJECT_STRUCTURE_TYPES_MAP_REVERSED:
        return JSON_OBJECT_STRUCTURE_TYPES_MAP_REVERSED[ident], i

    # Now check for generics (list, dict, union)
    i = _skip_whitespace(s, i)
    if i >= len(s) or s[i] != "[":
        raise ValueError(f"Expected '[' after {ident} at position {i} in {s!r}")
    i += 1  # skip '['

    if ident == "list":
        # list[<type>]
        inner_type, i = _parse_type_hint(s, i)
        i = _skip_whitespace(s, i)
        if i >= len(s) or s[i] != "]":
            raise ValueError(f"Expected ']' after list type at position {i} in {s!r}")
        i += 1  # skip ']'
        return list[inner_type], i

    elif ident == "dict":
        # dict[<key type>, <value type>]
        key_type, i = _parse_type_hint(s, i)
        i = _skip_whitespace(s, i)
        if i >= len(s) or s[i] != ",":
            raise ValueError(
                f"Expected ',' after dict key type at position {i} in {s!r}"
            )
        i += 1  # skip comma
        value_type, i = _parse_type_hint(s, i)
        i = _skip_whitespace(s, i)
        if i >= len(s) or s[i] != "]":
            raise ValueError(
                f"Expected ']' after dict value type at position {i} in {s!r}"
            )
        i += 1  # skip ']'
        return dict[key_type, value_type], i

    elif ident == "union":
        # union[<type1>, <type2>, ...]
        types_list = []
        while True:
            t, i = _parse_type_hint(s, i)
            types_list.append(t)
            i = _skip_whitespace(s, i)
            if i < len(s) and s[i] == ",":
                i += 1  # skip comma
                continue
            elif i < len(s) and s[i] == "]":
                i += 1  # skip ']'
                break
            else:
                raise ValueError(
                    f"Expected ',' or ']' in union at position {i} in {s!r}"
                )
        # If only one type is present, just return that type.
        if len(types_list) == 1:
            return types_list[0], i
        # Otherwise, build a Union.
        if len(types_list) == 2:
            return Union[types_list[0], types_list[1]], i
        else:
            # For more than two types, unpack them as arguments.
            args = (types_list[0], types_list[1]) + tuple(types_list[2:])
            union_type = Union.__getitem__(args)
            return union_type, i

    else:
        raise ValueError(f"Unknown type identifier: {ident}")


def _deserialize_type_hint(s: str) -> type | GenericAlias | UnionType:
    """
    Parses a string representation of a type hint and reconstructs the
    corresponding type hint. This function processes the entire string
    and validates whether the input string is properly formatted as a type hint.

    Once parsed, it ensures no additional characters exist after the type hint
    to guarantee the string is fully consumed.

    :param s: The string representation of a type hint to be deserialized.
    :type s: str

    :return: The reconstructed type hint as a type, a generic alias,
        or a union type.
    :rtype: type | GenericAlias | UnionType

    :raises ValueError: If extra characters are found after the type hint string.
    """
    result, index = _parse_type_hint(s, 0)
    index = _skip_whitespace(s, index)
    if index != len(s):
        raise ValueError(
            f"Extra characters found after type hint at position {index} in {s!r}"
        )
    return result
