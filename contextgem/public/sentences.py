# SPDX-FileCopyrightText: Shcherbak AI AS
# SPDX-License-Identifier: Apache-2.0

"""
Module for handling document sentences.

This module provides the Sentence class, which represents a structured unit of text
within a document paragraph. Sentences are the fundamental building blocks of text analysis,
containing the raw text content of individual statements.

The module supports validation to ensure data integrity and integrates with the paragraph
structure to maintain the hierarchical organization of document content.
"""

from __future__ import annotations

from contextgem.internal.base.paras_and_sents import _Sentence
from contextgem.internal.decorators import _expose_in_registry


@_expose_in_registry(additional_key=_Sentence)
class Sentence(_Sentence):
    """
    Represents a sentence within a document paragraph.

    Sentences are immutable text units that serve as the fundamental building blocks for
    document analysis. The raw text content is preserved and cannot be modified after
    initialization to maintain data integrity.

    :ivar raw_text: The complete text content of the sentence. This value is frozen after initialization.
    :vartype raw_text: str
    :ivar additional_context: Optional supplementary information about the sentence, as a
        single-line string. It is passed to the LLM together with the sentence text in the prompts
        that list sentences, i.e. when sentence-level references are used
        (``reference_depth="sentences"``). A decision model is sent it together with
        the sentence text when aspects are extracted with sentences as their items.
        Sentences that are segmented automatically inherit
        the additional context of their paragraph. Defaults to None.
    :vartype additional_context: str | None
    :ivar page_numbers: The numbers of the pages of the document that the sentence is on, from 1,
        in ascending order. Sentences that are segmented automatically inherit the page numbers of
        their paragraph. They are not sent to the models. Defaults to an empty list.
    :vartype page_numbers: list[int]

    Note:
        Normally, you do not need to construct sentences manually, as they are populated automatically
        from document's ``raw_text`` or ``paragraphs`` attributes. Only use this constructor for
        advanced use cases, such as when you have a custom paragraph/sentence segmentation tool.

    Example:
        .. literalinclude:: ../../../dev/usage_examples/docstrings/sentences/def_sentence.py
            :language: python
            :caption: Sentence definition
    """
