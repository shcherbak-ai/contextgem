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
Module for handling documents.

This module provides the Document class, which represents a structured or unstructured file
containing written or visual content. Documents can be processed to extract information,
analyze content, and organize data into paragraphs, sentences, aspects, and concepts.
"""

from __future__ import annotations

from contextgem.internal.base.documents import _Document
from contextgem.internal.decorators import _expose_in_registry


@_expose_in_registry(additional_key=_Document)
class Document(_Document):
    """
    Represents a document containing textual and visual content for analysis.

    A document serves as the primary container for content analysis within the ContextGem framework,
    enabling complex document understanding and information extraction workflows.

    :ivar raw_text: The main text of the document as a single string.
        Defaults to None.
    :vartype raw_text: str | None
    :ivar paragraphs: List of Paragraph instances in consecutive order as they appear
        in the document. Defaults to an empty list.
    :vartype paragraphs: list[Paragraph]
    :ivar images: List of Image instances attached to or representing the document.
        Defaults to an empty list.
    :vartype images: list[Image]
    :ivar aspects: List of aspects associated with the document for focused analysis.
        Validated to ensure unique names and descriptions. Defaults to an empty list.
    :vartype aspects: list[Aspect]
    :ivar concepts: List of concepts associated with the document for information extraction.
        Validated to ensure unique names and descriptions. Defaults to an empty list.
    :vartype concepts: list[_Concept]
    :ivar paragraph_segmentation_mode: Mode for paragraph segmentation. When set to "sat",
        uses a SaT (Segment Any Text https://arxiv.org/abs/2406.16678) model. Defaults to "newlines".
    :vartype paragraph_segmentation_mode: Literal["newlines", "sat"]
    :ivar sat_model_id: SaT model ID for paragraph/sentence segmentation or a local path to a SaT model.
        For model IDs, defaults to "sat-3l-sm". See https://github.com/segment-any-text/wtpsplit
        for the list of available models. For local paths, provide either a string path or a Path
        object pointing to the directory containing the SaT model.
    :vartype sat_model_id: SaTModelId
    :ivar pre_segment_sentences: Whether to pre-segment sentences during Document initialization.
        When False (default), sentence segmentation is deferred until sentences are actually needed,
        improving initialization performance. When True, sentences are segmented immediately during
        Document creation using the SaT model.
    :vartype pre_segment_sentences: bool

    Note:
        Normally, you do not need to construct/populate paragraphs manually, as they are
        populated automatically from document's ``raw_text`` attribute. Only use this constructor
        for advanced use cases, such as when you have a custom paragraph segmentation tool.

    Example:
        .. literalinclude:: ../../../dev/usage_examples/docstrings/documents/def_document.py
            :language: python
            :caption: Document definition
    """

    pass
