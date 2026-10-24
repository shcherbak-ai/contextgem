# SPDX-FileCopyrightText: Shcherbak AI AS
# SPDX-License-Identifier: Apache-2.0

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
        The list holds copies of the aspects given, made when it is assigned or
        added to (``add_aspects()``), so that an aspect can be reused for other
        documents: read the extracted items from the document's own aspects,
        e.g. ``document.aspects`` or ``get_aspect_by_name()``.
    :vartype aspects: list[Aspect]
    :ivar concepts: List of concepts associated with the document for information extraction.
        Validated to ensure unique names and descriptions. Defaults to an empty list.
        The list holds copies of the concepts given, made when it is assigned or
        added to (``add_concepts()``), so that a concept can be reused for other
        documents: read the extracted items from the document's own concepts,
        e.g. ``document.concepts``, ``get_concept_by_name()`` or the list that
        an extraction method returns.
    :vartype concepts: list[_Concept]
    :ivar paragraph_segmentation_mode: Mode for paragraph segmentation. When set to "sat",
        uses a SaT (Segment Any Text https://arxiv.org/abs/2406.16678) model. Defaults to "newlines".
    :vartype paragraph_segmentation_mode: Literal["newlines", "sat"]
    :ivar sat_model_id: SaT model ID for paragraph/sentence segmentation or a local path to a SaT model.
        For model IDs, defaults to "sat-3l-sm". See https://github.com/segment-any-text/wtpsplit
        for the list of available models. For local paths, provide either a string path or a Path
        object pointing to the directory containing the SaT model. A model given by its ID is
        downloaded from Hugging Face when it is first used, and cached. A directory that also
        contains the tokenizer of the model (tokenizer.json) is loaded without network access;
        otherwise, the tokenizer is downloaded from Hugging Face, or read from its cache.
    :vartype sat_model_id: SaTModelId
    :ivar pre_segment_sentences: Whether to pre-segment sentences during Document initialization.
        When False (default), sentence segmentation is deferred until sentences are actually needed,
        improving initialization performance. When True, sentences are segmented immediately during
        Document creation using the SaT model.
    :vartype pre_segment_sentences: bool
    :ivar sentences: All the sentences of the document's paragraphs, in order. Read-only:
        the sentences are those of the paragraphs, i.e. the same objects.
    :vartype sentences: list[Sentence]
    :ivar model_roles: The ``model_role`` values of the document's aspects and concepts,
        and of the sub-aspects and concepts of its aspects, with None for those without
        a role. Read-only. A ``DocumentModelGroup`` that extracts them needs a model
        for each of these roles.
    :vartype model_roles: set[str | None]
    :ivar custom_data: A dictionary for data of your own about the document, e.g. the ID
        that your application has for it. Must be JSON-serializable. Not sent to the models,
        and kept when the document is serialized or cloned. Defaults to an empty dict.
    :vartype custom_data: dict[str, Any]
    :ivar unique_id: The unique ID of the document, a ULID assigned when the document is
        created. Read-only. Kept when the document is serialized or cloned, i.e. a restored
        or cloned document has the ID of the original.
    :vartype unique_id: str

    Note:
        Normally, you do not need to construct/populate paragraphs manually, as they are
        populated automatically from document's ``raw_text`` attribute. Pass paragraphs
        instead of ``raw_text`` only if you segment the text yourself, e.g. with a paragraph
        segmentation tool of your own.
        To create a document from a file converted with Docling, with the structure of the
        file kept in its paragraphs, use :meth:`from_docling`.

    Example:
        .. literalinclude:: ../../../dev/usage_examples/docstrings/documents/def_document.py
            :language: python
            :caption: Document definition

        .. The example above is not in the installed package. It is shown at
           https://contextgem.shcherbak.ai/api/documents/index.html.md
    """
