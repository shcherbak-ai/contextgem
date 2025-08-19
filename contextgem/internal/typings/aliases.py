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
Module defining type aliases used throughout the ContextGem framework.

This module provides standardized type definitions and aliases that ensure
consistent typing across the codebase. It includes specialized string types,
literal types for configuration options, and compatibility solutions for
different Python versions.
"""

from __future__ import annotations

from collections.abc import Callable, Coroutine
from decimal import Decimal
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import Field, StrictStr, StringConstraints


NonEmptyStr = Annotated[
    StrictStr, StringConstraints(strip_whitespace=True, min_length=1)
]

LLMRoleAny = Literal[
    "extractor_text", "reasoner_text", "extractor_vision", "reasoner_vision"
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

ReasoningEffort = Literal["minimal", "low", "medium", "high"]

TextMode = Literal["raw", "markdown"]
