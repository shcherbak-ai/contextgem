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
    :ivar additional_context: Optional supplementary information about the sentence.
        It is passed to the LLM together with the sentence text in the prompts
        that list sentences, i.e. when sentence-level references are used
        (``reference_depth="sentences"``). A decision model is sent it together with
        the sentence text when aspects are extracted with sentences as their items.
        Its whitespace, line breaks included, is sent as single spaces, as the text of
        the sentence is. Sentences that are segmented automatically inherit
        the additional context of their paragraph. Defaults to None.
    :vartype additional_context: str | None
    :ivar page_numbers: The numbers of the pages of the document that the sentence is on, from 1,
        in ascending order. Sentences that are segmented automatically inherit the page numbers of
        their paragraph. They are not sent to the models. Defaults to an empty list.
    :vartype page_numbers: list[int]
    :ivar custom_data: A dictionary for data of your own about the sentence, e.g. the ID
        that your application has for its text. It reaches the extraction results: the
        ``reference_sentences`` of aspects and extracted items are the sentences of
        the document's own paragraphs, not copies (see
        https://contextgem.shcherbak.ai/documents/document_config/#locating-paras-and-sents),
        and an item that an aspect extracts as the sentence (``reference_depth="sentences"``)
        gets a copy of it. Sentences that are segmented automatically get a copy of
        the custom data of their paragraph. Must be JSON-serializable. Not sent to
        the models, and kept when the document is serialized or cloned. Defaults to an
        empty dict.
    :vartype custom_data: dict[str, Any]
    :ivar unique_id: The unique ID of the sentence, a ULID assigned when the sentence is
        created. Read-only. Kept when the document is serialized or cloned, i.e. a restored
        or cloned sentence has the ID of the original.
    :vartype unique_id: str

    Note:
        Normally, you do not need to construct sentences manually, as they are populated automatically
        from document's ``raw_text`` or ``paragraphs`` attributes. Only use this constructor for
        advanced use cases, such as when you have a custom paragraph/sentence segmentation tool.

    Example:
        .. literalinclude:: ../../../dev/usage_examples/docstrings/sentences/def_sentence.py
            :language: python
            :caption: Sentence definition

        .. The example above is not in the installed package. It is shown at
           https://contextgem.shcherbak.ai/api/sentences/index.html.md
    """
