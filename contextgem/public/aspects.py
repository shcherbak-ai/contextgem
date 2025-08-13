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
        unique in both name and description and cannot include concepts with vision LLM roles.
    :vartype concepts: list[_Concept]
    :ivar llm_role: The role of the LLM responsible for aspect extraction.
        Default is "extractor_text". Valid roles are "extractor_text" and "reasoner_text".
    :vartype llm_role: LLMRoleAspect
    :ivar reference_depth: The structural depth of references (paragraphs or sentences).
        Defaults to "paragraphs". Affects the structure of ``extracted_items``.
    :vartype reference_depth: ReferenceDepth
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

    pass
