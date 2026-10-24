# SPDX-FileCopyrightText: Shcherbak AI AS
# SPDX-License-Identifier: Apache-2.0

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

from pydantic import (
    AfterValidator,
    BeforeValidator,
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
    StringConstraints,
)

from contextgem.internal.typings.validators import (
    _validate_is_json_dict,
    _validate_model_role,
)


NonEmptyStr = Annotated[
    StrictStr, StringConstraints(strip_whitespace=True, min_length=1)
]

# Free-form label that routes aspects/concepts to a model in a model group
ModelRole = Annotated[StrictStr, AfterValidator(_validate_model_role)]

Modality = Literal["text", "image", "multimodal"]

AssignedInstancesAttrName = Literal["aspects", "concepts"]

DefaultPromptType = Literal["aspect", "concept"]

ExtractedInstanceType = Literal["aspect", "concept"]

ReferenceDepth = Literal["paragraphs", "sentences"]

ClassificationType = Literal["multi_class", "multi_label"]

# How a model or its provider declined to answer a request: the model refused
# ("refusal", e.g. OpenAI's refusal field, or a refused answer of a decision model),
# the answer was stopped with the "content_filter" finish reason, which litellm
# reports for a refusal of a model as well as for a content filter of a provider,
# or the provider blocked the request or its answer ("blocked"), which litellm
# raises as ContentPolicyViolationError for both
ModelRefusal = Literal["refusal", "content_filter", "blocked"]

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

ReasoningEffort = Literal["none", "minimal", "low", "medium", "high", "xhigh", "max"]

MessageRole = Literal["system", "user", "assistant", "tool"]

# Probability of an option of a closed question (yes/no, one of the labels,
# a level of a rating scale). An integer is converted to a float. A boolean,
# a string and a NaN are rejected.
Probability = Annotated[StrictFloat, Field(ge=0.0, le=1.0, allow_inf_nan=False)]

# Probabilities of the options of the closed question answered by an extracted item,
# keyed by the options as they appear as item values. The types of the options are
# strict, as `True`, `1` and `"1"` are different options: without that, `1.0` would
# be validated as `True`.
ItemProbabilities = Annotated[
    dict[StrictBool | StrictInt | StrictStr, Probability], Field(min_length=1)
]

# JSON-serializable types
# A JSON value can be a primitive, a list of JSON values, or a dict of str->JSON value
JSONPrimitive = str | int | float | bool | None
JSONValue = JSONPrimitive | list["JSONValue"] | dict[str, "JSONValue"]
JSONDict = dict[str, JSONValue]
# For use as a field type in Pydantic models, avoid recursive forward references
# and use a validator instead.
JSONDictField = Annotated[dict[str, Any], BeforeValidator(_validate_is_json_dict)]

# The number of a page of a document, from 1
PageNumber = Annotated[StrictInt, Field(ge=1)]


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
    :param definition: Tool definition in OpenAI-compatible schema, generated from
        the handler and sent to the LLM.
    :type definition: JSONDict
    :param schema: JSON schema (object) from `function.parameters` of the tool
        definition, describing tool arguments.
    :type schema: JSONDict
    """

    handler: ToolHandler
    definition: JSONDict
    schema: JSONDict
