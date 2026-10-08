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
Module for handling processing logic using model groups.

This module provides the class for grouping models in document processing workflows.
Each model of a group extracts the aspects and concepts routed to it by its role,
e.g. to use a smaller model for simpler tasks and a more capable model for more
complex ones.
"""

from __future__ import annotations

# `_DocumentLLM` is not used in this module, but has to be in its namespace when
# the `DocumentModelGroup` class is created: pydantic resolves the type of the inherited
# `models` field (`list[_DocumentLLM]`) in the namespace of this module, as the module
# of `_DocumentModelGroup` cannot import `_DocumentLLM` at runtime (circular import).
from contextgem.internal.base.llms import _DocumentLLM  # noqa: F401
from contextgem.internal.base.models import _DocumentModelGroup
from contextgem.internal.decorators import _expose_in_registry


@_expose_in_registry(additional_key=_DocumentModelGroup)  # ty: ignore[too-many-positional-arguments]
class DocumentModelGroup(_DocumentModelGroup):
    """
    Represents a group of DocumentLLMs with unique roles for processing document content.

    This class manages multiple LLMs, each of which extracts the aspects and concepts routed
    to it by its role, e.g. to use a smaller model for simpler tasks and a more capable
    model for more complex ones.

    :ivar models: A list of DocumentLLM instances, each with a unique ``role``. An aspect/concept
        is extracted by the LLM with the ``role`` equal to the ``model_role`` of the aspect/concept,
        and an aspect/concept without ``model_role`` is extracted by the LLM without ``role``.
        Only one LLM in the group can have no role. At least 2 instances are required.
    :vartype models: list[DocumentLLM]
    :ivar output_language: Language for produced output text (justifications, explanations).
        Values: "en" (always English) or "adapt" (matches document/image language).
        All LLMs in the group must share the same output_language setting.
        Defaults to "en". Applies only when DocumentLLMs' default system messages are used.
    :vartype output_language: LanguageRequirement

    Note:
        Refer to the :class:`~contextgem.public.llms.DocumentLLM` class for more information on constructing LLMs
        for the group.

    Example:
        .. literalinclude:: ../../../dev/usage_examples/docstrings/llms/def_model_group.py
            :language: python
            :caption: Model group definition
    """

    pass
