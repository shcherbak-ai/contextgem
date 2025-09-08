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
Module defining custom validators.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from contextgem.internal.typings.types import JSONDict


def _validate_sequence_is_list(
    v: Sequence,
) -> list:
    """
    Validator function to ensure that the sequence value is a list.
    Necessary for type compliance of Sequence fields.

    Can be used either as a field validator (e.g. in a pydantic field -
    Annotated[Sequence[_Concept], BeforeValidator(_validate_sequence_is_list)])
    or as a standalone function.

    :param v: The sequence value to validate.
    :return: The validated list.
    :raises ValueError: If the sequence is not a list.
    """
    if not isinstance(v, list):
        raise ValueError("Sequence must be a list")
    return v


def _is_json_value(v: Any) -> bool:
    """
    Returns True if the value conforms to JSONValue
    (str|int|float|bool|None|list|dict[str,JSONValue]).

    :param v: The value to check.
    :type v: Any
    :return: True if the value conforms to JSONValue, False otherwise.
    :rtype: bool
    """
    if v is None or isinstance(v, str | int | float | bool):
        return True
    if isinstance(v, list):
        return all(_is_json_value(item) for item in v)
    if isinstance(v, dict):
        return all(isinstance(k, str) and _is_json_value(val) for k, val in v.items())
    return False


def _validate_is_json_dict(v: Any) -> JSONDict:
    """
    Validates that v is a dict[str, JSONValue]. Raises ValueError otherwise.

    :param v: The value to validate.
    :type v: Any
    :return: The validated value.
    :rtype: JSONDict
    :raises ValueError: If the value is not a dict or does not conform to JSONValue.
    """
    if not isinstance(v, dict):
        raise ValueError("Expected a JSON object (dict)")
    if not _is_json_value(v):
        raise ValueError("Schema must be JSON-serializable with string keys")
    # v matches JSONDict by construction
    return v


def _validate_tool_parameters_schema(v: Any) -> JSONDict:
    """
    Validates that a tool `function.parameters` schema is a proper JSON schema object
    with at least the following structure:

    - type == "object"
    - properties: dict[str, JSONValue]
    - required: list[str] whose items are keys in properties

    Note: This is a lightweight structural validator to catch common mistakes early.
    Detailed value-level validation is performed at runtime by fastjsonschema.

    :param v: The parameters object to validate.
    :type v: Any
    :return: The validated JSONDict.
    :rtype: JSONDict
    :raises ValueError: If the structure is invalid.
    """
    obj = _validate_is_json_dict(v)

    # `type` must be "object"
    t = obj.get("type")
    if t != "object":
        raise ValueError("Tool parameters must have type == 'object'")

    # `properties` must be a dict
    props = obj.get("properties")
    if not isinstance(props, dict):
        raise ValueError("Tool parameters must include a 'properties' object")

    # `required` must be a list of strings, and each must exist in properties
    req: list[str] = obj.get("required", [])  # type: ignore[assignment]
    if not isinstance(req, list) or not all(isinstance(i, str) for i in req):
        raise ValueError("Tool parameters 'required' must be a list of strings")
    missing: list[str] = [k for k in req if k not in props]
    if missing:
        missing_str = ", ".join(missing)
        raise ValueError(
            "Tool parameters 'required' keys missing in properties: " + missing_str
        )

    return obj
