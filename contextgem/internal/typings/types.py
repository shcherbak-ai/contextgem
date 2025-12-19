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
Module defining core types and aliases used throughout the ContextGem framework.

This module centralizes standardized type definitions (e.g., TypedDicts,
callable signatures) and lightweight aliases to ensure consistent typing across
the codebase. It includes specialized string types, literals for configuration
options, JSON-serializable type helpers, and tool-calling types.
"""

from __future__ import annotations

from collections.abc import Callable, Coroutine
from decimal import Decimal
from pathlib import Path
from typing import Annotated, Any, Literal, TypedDict

from pydantic import BeforeValidator, Field, StrictStr, StringConstraints

from contextgem.internal.typings.validators import _validate_is_json_dict


NonEmptyStr = Annotated[
    StrictStr, StringConstraints(strip_whitespace=True, min_length=1)
]

LLMRoleAny = Literal[
    "extractor_text",
    "reasoner_text",
    "extractor_vision",
    "reasoner_vision",
    "extractor_multimodal",
    "reasoner_multimodal",
]

LLMRoleAspect = Literal["extractor_text", "reasoner_text"]

AssignedInstancesAttrName = Literal["aspects", "concepts"]

DefaultPromptType = Literal["aspects", "concepts"]

ExtractedInstanceType = Literal["aspect", "concept"]

ReferenceDepth = Literal["paragraphs", "sentences"]

ClassificationType = Literal["multi_class", "multi_label"]

# Define standard SaT model IDs as a separate type
StandardSaTModelId = Literal[
    "sat-1l",
    "sat-1l-sm",
    "sat-3l",
    "sat-3l-sm",
    "sat-6l",
    "sat-6l-sm",
    "sat-9l",
    "sat-12l",
    "sat-12l-sm",
]

# Combined type for sat_model_id parameter
SaTModelId = StandardSaTModelId | str | Path


LanguageRequirement = Literal["en", "adapt"]

JustificationDepth = Literal["brief", "balanced", "comprehensive"]

AsyncCalsAndKwargs = list[
    tuple[Callable[..., Coroutine[Any, Any, Any]], dict[str, Any]]
]

DefaultDecimalField = Field(
    default_factory=lambda: Decimal("0.00000"), ge=Decimal("0.00000")
)

ReasoningEffort = Literal["minimal", "low", "medium", "high", "xhigh"]

TextMode = Literal["raw", "markdown"]

MessageRole = Literal["system", "user", "assistant", "tool"]

# JSON-serializable types
# A JSON value can be a primitive, a list of JSON values, or a dict of str->JSON value
JSONPrimitive = str | int | float | bool | None
JSONValue = JSONPrimitive | list["JSONValue"] | dict[str, "JSONValue"]
JSONDict = dict[str, JSONValue]
# For use as a field type in Pydantic models, avoid recursive forward references
# and use a validator instead.
JSONDictField = Annotated[dict[str, Any], BeforeValidator(_validate_is_json_dict)]


# Tool-calling related types
# A tool handler can be sync or async, but must return a string.
# The runtime will ensure the tool message content is a string.
ToolHandlerSync = Callable[..., str]
ToolHandlerAsync = Callable[..., Coroutine[Any, Any, str]]
ToolHandler = ToolHandlerSync | ToolHandlerAsync


class ToolRegistration(TypedDict):
    """
    Internal structure used to store tool data in registry.

    :param handler: Sync or async callable that must return a string.
    :type handler: ToolHandler
    :param schema: JSON schema (object) from `function.parameters` describing tool arguments.
    :type schema: JSONDict
    """

    handler: ToolHandler
    schema: JSONDict
