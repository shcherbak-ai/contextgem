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
Module for handling document paragraphs.

This module provides the Paragraph class, which represents a structured segment of text
within a document. Paragraphs serve as containers for sentences and maintain the raw text
content of the segment they represent.

The module supports validation to ensure data integrity and provides mechanisms to prevent
inconsistencies during document analysis by restricting certain attribute modifications
after initial assignment.
"""

from __future__ import annotations

from contextgem.internal.base.paras_and_sents import _Paragraph
from contextgem.internal.decorators import _expose_in_registry


@_expose_in_registry(additional_key=_Paragraph)
class Paragraph(_Paragraph):
    """
    Represents a paragraph of a document with its raw text content and constituent sentences.

    Paragraphs are immutable text segments that can contain multiple sentences. Once sentences
    are assigned to a paragraph, they cannot be changed to maintain data integrity during analysis.

    :ivar raw_text: The complete text content of the paragraph. This value is frozen after initialization.
    :vartype raw_text: str
    :ivar sentences: The individual sentences contained within the paragraph. Defaults to an empty list.
        Cannot be reassigned once populated.
    :vartype sentences: list[Sentence]

    Note:
        Normally, you do not need to construct paragraphs manually, as they are populated automatically
        from document's ``raw_text`` attribute. Only use this constructor for advanced use cases,
        such as when you have a custom paragraph segmentation tool.

    Example:
        .. literalinclude:: ../../../dev/usage_examples/docstrings/paragraphs/def_paragraph.py
            :language: python
            :caption: Paragraph definition
    """

    pass
