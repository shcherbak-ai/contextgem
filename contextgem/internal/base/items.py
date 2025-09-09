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
Module defining the base classes for item subclasses.

This module provides the foundational class structure for items that can be extracted
from aspects or documents in the ContextGem framework. Items serve as the basic units of information
extracted from aspects or documents, providing a structured way to store and process extracted data.
"""

from __future__ import annotations

from typing import Any

from pydantic import Field, PrivateAttr

from contextgem.internal.base.attrs import _RefParasAndSentsAttrituteProcessor
from contextgem.internal.base.paras_and_sents import _Paragraph, _Sentence
from contextgem.internal.decorators import (
    _disable_direct_initialization,
)
from contextgem.internal.typings.types import NonEmptyStr


@_disable_direct_initialization
class _ExtractedItem(_RefParasAndSentsAttrituteProcessor):
    """
    Base class for extracted items.
    """

    value: Any = Field(..., frozen=True, description="Extracted value.")
    justification: NonEmptyStr | None = Field(
        default=None,
        frozen=True,
        description="Optional justification (explanation) for the extraction.",
    )

    _reference_paragraphs: list[_Paragraph] = PrivateAttr(default_factory=list)
    _reference_sentences: list[_Sentence] = PrivateAttr(default_factory=list)
