# SPDX-FileCopyrightText: Shcherbak AI AS
# SPDX-License-Identifier: Apache-2.0

"""
Module for handling document aspects.

This module provides the Aspect class, which represents a defined area or topic within a document
that requires focused attention. Aspects are used to identify and extract specific subjects or themes
from documents according to predefined criteria.
"""

from __future__ import annotations

from contextgem.internal.base.aspects import _Aspect
from contextgem.internal.decorators import _expose_in_registry


@_expose_in_registry(additional_key=_Aspect)
class Aspect(_Aspect):
    """
    Represents an aspect with associated metadata, sub-aspects, concepts, and logic for validation.

    An aspect is a defined area or topic within a document that requires focused attention.
    Each aspect corresponds to a specific subject or theme described in the task.

    :ivar name: The name of the aspect. Required, non-empty string.
    :vartype name: str
    :ivar description: A detailed description of the aspect. Required, non-empty string.
    :vartype description: str
    :ivar concepts: A list of concepts associated with the aspect. These concepts must be
        unique in both name and description, and must have the "text" modality.
    :vartype concepts: list[_Concept]
    :ivar model_role: The role label of the model that extracts the aspect when
        a ``DocumentModelGroup`` is used: the aspect is routed to the model in the group
        with the same ``role``. Defaults to None, which routes the aspect to the model
        without a role. Ignored by a standalone model, which extracts every
        aspect it is given.
    :vartype model_role: str | None
    :ivar reference_depth: The structural depth of references (paragraphs or sentences).
        Defaults to "paragraphs". Affects the structure of ``extracted_items``.
    :vartype reference_depth: ReferenceDepth
    :ivar decision_threshold: The lowest probability (above 0, up to 1) with which
        a decision model (``DocumentDecisionModel``) takes a paragraph as belonging to
        the aspect, and, with ``reference_depth="sentences"``, a sentence of such
        a paragraph. Defaults to None, in which case the ``decision_threshold`` of
        the decision model applies. Not passed on to the sub-aspects. Ignored when
        the aspect is extracted by an LLM.
    :vartype decision_threshold: float | None
    :ivar add_justifications: Whether the LLM will output justification for each extracted item.
        Inherited from base class. Defaults to False.
    :vartype add_justifications: bool
    :ivar justification_depth: The level of detail for justifications.
        Inherited from base class. Defaults to "brief".
    :vartype justification_depth: JustificationDepth
    :ivar justification_max_sents: Maximum number of sentences in a justification.
        Inherited from base class. Defaults to 2.
    :vartype justification_max_sents: int
    :ivar aspects: The sub-aspects of the aspect, extracted from the paragraphs of
        the aspect once it is extracted. Their names and descriptions must differ from
        those of the aspect, and a sub-aspect cannot have sub-aspects of its own.
        Defaults to an empty list.
    :vartype aspects: list[Aspect]
    :ivar extracted_items: The items extracted for the aspect, set by the extraction
        methods: each is a paragraph of the document, or a sentence with
        ``reference_depth="sentences"``, with its text as ``value``, the paragraph (and the
        sentence) in its ``reference_paragraphs`` (and ``reference_sentences``), and its
        ``justification`` and ``probabilities`` where these apply. For the attributes of
        the items, see
        https://contextgem.shcherbak.ai/aspects/aspects/. Defaults to an empty list.
    :vartype extracted_items: list[_StringItem]
    :ivar reference_paragraphs: The paragraphs of the extracted items, each once, in
        the order of the items, set by the extraction methods. They are the document's
        own paragraphs, not copies, and the sub-aspects and the concepts of the aspect
        are extracted from them. Defaults to an empty list.
    :vartype reference_paragraphs: list[Paragraph]
    :ivar reference_sentences: The sentences of the extracted items with
        ``reference_depth="sentences"``, each once, i.e. the extracted sentences, not all
        the sentences of the reference paragraphs. They are the sentences of
        the document's own paragraphs. Defaults to an empty list.
    :vartype reference_sentences: list[Sentence]
    :ivar reference_page_numbers: The numbers of the pages that the reference paragraphs
        and sentences are on, in ascending order, e.g. in a document created with
        ``Document.from_docling()``. Read-only, made from the ``page_numbers`` of
        the references. Empty if they have none.
    :vartype reference_page_numbers: list[int]
    :ivar model_roles: The ``model_role`` values of the sub-aspects and the concepts of
        the aspect, and of the concepts of its sub-aspects, with None for those without
        a role. The aspect's own ``model_role`` is not included. Read-only.
    :vartype model_roles: set[str | None]
    :ivar custom_data: A dictionary for data of your own about the aspect, e.g. the ID
        that your application has for it. Must be JSON-serializable. Not sent to the models,
        and kept when the aspect is serialized or cloned. Defaults to an empty dict.
    :vartype custom_data: dict[str, Any]
    :ivar unique_id: The unique ID of the aspect, a ULID assigned when the aspect is
        created. Read-only. Kept when the aspect is serialized or cloned, i.e. a restored
        or cloned aspect has the ID of the original.
    :vartype unique_id: str

    Example:
        .. literalinclude:: ../../../dev/usage_examples/docstrings/aspects/def_aspect.py
            :language: python
            :caption: Aspect definition
    """
