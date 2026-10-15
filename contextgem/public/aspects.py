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

    Example:
        .. literalinclude:: ../../../dev/usage_examples/docstrings/aspects/def_aspect.py
            :language: python
            :caption: Aspect definition
    """
