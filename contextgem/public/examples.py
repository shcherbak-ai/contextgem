# SPDX-FileCopyrightText: Shcherbak AI AS
# SPDX-License-Identifier: Apache-2.0

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
    :ivar custom_data: A dictionary for data of your own about the example. Must be
        JSON-serializable. Not sent to the models, and kept when the example is
        serialized or cloned. Defaults to an empty dict.
    :vartype custom_data: dict[str, Any]
    :ivar unique_id: The unique ID of the example, a ULID assigned when the example is
        created. Read-only. Kept when the example is serialized or cloned.
    :vartype unique_id: str

    Note:
        Examples are optional and can be used to guide LLM extraction tasks. They serve as reference
        points for the model to understand the expected format and content of extracted information.
        StringExample can be attached to a :class:`~contextgem.public.concepts.StringConcept`.

    Example:
        .. literalinclude:: ../../../dev/usage_examples/docstrings/examples/def_example_string.py
            :language: python
            :caption: String example definition

        .. The example above is not in the installed package. It is shown at
           https://contextgem.shcherbak.ai/api/examples/index.html.md
    """


@_expose_in_registry(additional_key=_JsonObjectExample)
class JsonObjectExample(_JsonObjectExample):
    """
    Represents a JSON object example that can be provided by users for certain extraction tasks.

    :ivar content: A JSON-serializable dict with the minimum length of 1 that holds
        the content of the example.
    :vartype content: dict[str, Any]
    :ivar custom_data: A dictionary for data of your own about the example. Must be
        JSON-serializable. Not sent to the models, and kept when the example is
        serialized or cloned. Defaults to an empty dict.
    :vartype custom_data: dict[str, Any]
    :ivar unique_id: The unique ID of the example, a ULID assigned when the example is
        created. Read-only. Kept when the example is serialized or cloned.
    :vartype unique_id: str

    Note:
        Examples are optional and can be used to guide LLM extraction tasks. They serve as reference
        points for the model to understand the expected format and content of extracted information.
        JsonObjectExample can be attached to a :class:`~contextgem.public.concepts.JsonObjectConcept`.

    Example:
        .. literalinclude:: ../../../dev/usage_examples/docstrings/examples/def_example_json_object.py
            :language: python
            :caption: JSON object example definition

        .. The example above is not in the installed package. It is shown at
           https://contextgem.shcherbak.ai/api/examples/index.html.md
    """
