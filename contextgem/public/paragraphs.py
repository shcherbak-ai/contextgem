# SPDX-FileCopyrightText: Shcherbak AI AS
# SPDX-License-Identifier: Apache-2.0

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
    :ivar additional_context: Optional supplementary information about the paragraph, as a
        single-line string, e.g. its structural role, formatting, or position in the document
        ("Heading, level 2" or "Table: Fees, row 2, column 3"). It is passed to the LLM
        together with the paragraph text in the prompts that list paragraphs, i.e. when
        aspects are extracted, or when concepts are extracted with references. A decision
        model is sent it together with the paragraph text when aspects are extracted.
        Sentences segmented from the paragraph inherit it. Defaults to None.
    :vartype additional_context: str | None
    :ivar page_numbers: The numbers of the pages of the document that the paragraph is on,
        from 1, in ascending order, e.g. ``[4, 5]`` for a paragraph that continues on the next
        page. Set by :meth:`~contextgem.public.documents.Document.from_docling` for the formats
        with pages, e.g. PDF, and for a presentation (the slides) or a spreadsheet (the sheets).
        The ``reference_page_numbers`` of aspects and extracted items are
        made from them, e.g. to cite the pages of an extracted item. They are not sent to the
        models. Sentences segmented from the paragraph inherit them. Defaults to an empty list.
    :vartype page_numbers: list[int]
    :ivar custom_data: A dictionary for data of your own about the paragraph, e.g. the ID
        that your application has for its text. It reaches the extraction results: the
        ``reference_paragraphs`` of aspects and extracted items are the document's own
        paragraphs, not copies (see
        https://contextgem.shcherbak.ai/documents/document_config/#locating-paras-and-sents),
        an item that an aspect extracts as the paragraph gets a copy of it, and so does each
        sentence segmented from the paragraph. ``Document.from_docling()`` stores where
        the paragraph comes from under the key ``"docling"``. Must be JSON-serializable.
        Not sent to the models, and kept when the document is serialized or cloned.
        Defaults to an empty dict.
    :vartype custom_data: dict[str, Any]
    :ivar unique_id: The unique ID of the paragraph, a ULID assigned when the paragraph is
        created. Read-only. Kept when the document is serialized or cloned, i.e. a restored
        or cloned paragraph has the ID of the original.
    :vartype unique_id: str

    Note:
        Normally, you do not need to construct paragraphs manually, as they are populated automatically
        from document's ``raw_text`` attribute. Only use this constructor for advanced use cases,
        such as when you have a custom paragraph segmentation tool, or when you want to provide
        additional context for the paragraphs.

    Example:
        .. literalinclude:: ../../../dev/usage_examples/docstrings/paragraphs/def_paragraph.py
            :language: python
            :caption: Paragraph definition

        .. The example above is not in the installed package. It is shown at
           https://contextgem.shcherbak.ai/api/paragraphs/index.html.md
    """
