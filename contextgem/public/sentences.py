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
Module for handling document sentences.

This module provides the Sentence class, which represents a structured unit of text
within a document paragraph. Sentences are the fundamental building blocks of text analysis,
containing the raw text content of individual statements.

The module supports validation to ensure data integrity and integrates with the paragraph
structure to maintain the hierarchical organization of document content.
"""

from __future__ import annotations

from pydantic import Field

from contextgem.internal.base.paras_and_sents import _ParasAndSentsBase
from contextgem.internal.typings.aliases import NonEmptyStr


class Sentence(_ParasAndSentsBase):
    """
    Represents a sentence within a document paragraph.

    Sentences are immutable text units that serve as the fundamental building blocks for
    document analysis. The raw text content is preserved and cannot be modified after
    initialization to maintain data integrity.

    :ivar raw_text: The complete text content of the sentence. This value is frozen after initialization.
    :vartype raw_text: NonEmptyStr

    Note:
        Normally, you do not need to construct sentences manually, as they are populated automatically
        from document's ``raw_text`` or ``paragraphs`` attributes. Only use this constructor for
        advanced use cases, such as when you have a custom paragraph/sentence segmentation tool.

    Example:
        .. literalinclude:: ../../../dev/usage_examples/docstrings/sentences/def_sentence.py
            :language: python
            :caption: Sentence definition
    """

    raw_text: NonEmptyStr = Field(..., frozen=True)
