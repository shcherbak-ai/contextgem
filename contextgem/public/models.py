# SPDX-FileCopyrightText: Shcherbak AI AS
# SPDX-License-Identifier: Apache-2.0

"""
Module for handling processing logic using model groups.

This module provides the class for grouping models in document processing workflows.
Each model of a group extracts the aspects and concepts routed to it by its role,
e.g. to use a smaller model for simpler tasks and a more capable model for more
complex ones.
"""

from __future__ import annotations

from contextgem.internal.base.models import _DocumentModelGroup
from contextgem.internal.decorators import _expose_in_registry


@_expose_in_registry(additional_key=_DocumentModelGroup)
class DocumentModelGroup(_DocumentModelGroup):
    """
    Represents a group of models with unique roles for processing document content.

    This class manages multiple models, each of which extracts the aspects and concepts routed
    to it by its role, e.g. to use a smaller model for simpler tasks and a more capable
    model for more complex ones. A group can also have decision models
    (``DocumentDecisionModel``) next to its LLMs: a decision model extracts the aspects
    routed to it, and the concepts routed to it that have values from a closed set
    of options.

    :ivar models: A list of models (DocumentLLM instances, DocumentDecisionModel instances,
        or both), each with a unique ``role``. An aspect/concept
        is extracted by the model with the ``role`` equal to the ``model_role`` of the aspect/concept,
        and an aspect/concept without ``model_role`` is extracted by the model without ``role``.
        Only one model in the group can have no role. At least 2 instances are required.
    :vartype models: list[DocumentLLM | DocumentDecisionModel]
    :ivar output_language: Language for produced output text (justifications, explanations).
        Values: "en" (always English) or "adapt" (matches document/image language).
        All LLMs in the group must share the same output_language setting.
        Defaults to "en". Applies to the extraction prompts of the LLMs, and to their
        default system messages.
        Does not apply to the decision models in the group, which produce no text.
    :vartype output_language: LanguageRequirement
    :ivar is_group: Whether the object is a model group: always True for a model group.
        Read-only.
    :vartype is_group: bool
    :ivar list_roles: The roles of the models in the group, in the order of ``models``,
        with None for the model without a role. Read-only.
    :vartype list_roles: list[str | None]

    Note:
        Refer to the :class:`~contextgem.public.llms.DocumentLLM` class for more information on constructing LLMs
        for the group.

    Example:
        .. literalinclude:: ../../../dev/usage_examples/docstrings/llms/def_model_group.py
            :language: python
            :caption: Model group definition
    """
