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
Module for handling example data in document processing.

This module provides classes for defining examples that can be used to guide LLM extraction tasks.
Examples serve as reference points for the model to understand the expected format and content
of extracted information. The module supports different types of examples including string-based
examples and structured JSON object examples.

Examples can be attached to concepts to provide concrete illustrations of the kind of information
to be extracted, improving the accuracy and consistency of LLM-based extraction processes.
"""

from __future__ import annotations

from contextgem.internal.base.examples import (
    _JsonObjectExample,
    _StringExample,
)
from contextgem.internal.decorators import _expose_in_registry


@_expose_in_registry(additional_key=_StringExample)
class StringExample(_StringExample):
    """
    Represents a string example that can be provided by users for certain extraction tasks.

    :ivar content: A non-empty string that holds the text content of the example.
    :vartype content: str

    Note:
        Examples are optional and can be used to guide LLM extraction tasks. They serve as reference
        points for the model to understand the expected format and content of extracted information.
        StringExample can be attached to a :class:`~contextgem.public.concepts.StringConcept`.

    Example:
        .. literalinclude:: ../../../dev/usage_examples/docstrings/examples/def_example_string.py
            :language: python
            :caption: String example definition
    """

    pass


@_expose_in_registry(additional_key=_JsonObjectExample)
class JsonObjectExample(_JsonObjectExample):
    """
    Represents a JSON object example that can be provided by users for certain extraction tasks.

    :ivar content: A JSON-serializable dict with the minimum length of 1 that holds
        the content of the example.
    :vartype content: dict[str, Any]

    Note:
        Examples are optional and can be used to guide LLM extraction tasks. They serve as reference
        points for the model to understand the expected format and content of extracted information.
        JsonObjectExample can be attached to a :class:`~contextgem.public.concepts.JsonObjectConcept`.

    Example:
        .. literalinclude:: ../../../dev/usage_examples/docstrings/examples/def_example_json_object.py
            :language: python
            :caption: JSON object example definition
    """

    pass
