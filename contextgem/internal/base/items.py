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

from typing import Any, Optional

from pydantic import Field, PrivateAttr

from contextgem.internal.base.attrs import _RefParasAndSentsAttrituteProcessor
from contextgem.internal.decorators import _post_init_method
from contextgem.internal.typings.aliases import NonEmptyStr
from contextgem.public.paragraphs import Paragraph
from contextgem.public.sentences import Sentence


class _ExtractedItem(_RefParasAndSentsAttrituteProcessor):
    """
    Base class for items extracted from aspects or documents in the ContextGem framework.

    This class provides a structured way to store extracted information along with
    optional justification and reference data.

    :ivar value: The extracted information value.
    :vartype value: Any
    :ivar justification: Optional explanation providing context for the extraction.
        Defaults to None.
    :vartype justification: Optional[NonEmptyStr]
    :ivar reference_paragraphs: List of paragraphs referenced by this item.
    :vartype reference_paragraphs: list[Paragraph]
    :ivar reference_sentences: List of sentences referenced by this item.
    :vartype reference_sentences: list[Sentence]
    """

    value: Any = Field(..., frozen=True)
    justification: Optional[NonEmptyStr] = Field(default=None, frozen=True)

    _reference_paragraphs: list[Paragraph] = PrivateAttr(default_factory=list)
    _reference_sentences: list[Sentence] = PrivateAttr(default_factory=list)

    @_post_init_method
    def _post_init(self, __context):
        if self.__class__ == _ExtractedItem:
            raise TypeError("Cannot instantiate base class directly")
