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
Module defining the base classes for item subclasses.

This module provides the foundational class structure for items that can be extracted
from aspects or documents in the ContextGem framework. Items serve as the basic units of information
extracted from aspects or documents, providing a structured way to store and process extracted data.
"""

from __future__ import annotations

from typing import Any

from pydantic import Field, PrivateAttr

from contextgem.internal.base.attrs import _RefParasAndSentsAttrituteProcessor
from contextgem.internal.base.paras_and_sents import _Paragraph, _Sentence
from contextgem.internal.decorators import (
    _disable_direct_initialization,
)
from contextgem.internal.typings.types import ItemProbabilities, NonEmptyStr


@_disable_direct_initialization
class _ExtractedItem(_RefParasAndSentsAttrituteProcessor):
    """
    Base class for extracted items.

    :ivar value: Extracted value.
    :vartype value: Any
    :ivar justification: Optional justification (explanation) for the extraction.
    :vartype justification: str | None
    :ivar probabilities: Probabilities of the options of the closed question that
        the item answers, as given by a decision model. The keys are the options as
        they appear as item values, each mapped to a probability within [0, 1]:

        - ``True`` and ``False`` for a yes/no question, e.g.
          ``{True: 0.83, False: 0.17}``. For an item of an aspect, the question
          is whether the paragraph belongs to the aspect;
        - the labels for a classification, e.g. ``{"written": 0.71, "email": 0.2,
          "any": 0.06, "not specified": 0.03}``. Where several labels can apply,
          each label has its own probability, i.e. the probabilities do not sum to 1,
          e.g. ``{"NDA": 0.97, "Confidential": 0.62, "Employment": 0.08}``;
        - the integers of the scale for a rating, e.g.
          ``{1: 0.0, 2: 0.0, 3: 0.15, 4: 0.55, 5: 0.3}``.

        The types and the order of the options are kept, e.g. ``True`` is not
        the same option as ``1``. None (default) if the item has no probabilities,
        which is the case for every item extracted by an LLM.
    :vartype probabilities: dict[bool | int | str, float] | None
    """

    value: Any = Field(..., frozen=True, description="Extracted value.")
    justification: NonEmptyStr | None = Field(
        default=None,
        frozen=True,
        description="Optional justification (explanation) for the extraction.",
    )
    probabilities: ItemProbabilities | None = Field(
        default=None,
        frozen=True,
        description="Optional probabilities of the options of the closed question "
        "that the item answers, keyed by the options as they appear as item values.",
    )

    _reference_paragraphs: list[_Paragraph] = PrivateAttr(default_factory=list)
    _reference_sentences: list[_Sentence] = PrivateAttr(default_factory=list)

    def __eq__(self, other: object) -> bool:
        """
        Checks whether the item is equal to another object.

        Adds the comparison of the types of the options in ``probabilities`` to
        the comparison of the instances: ``{True: 0.5}`` and ``{1: 0.5}`` are equal
        as dicts, while they hold the probabilities of different options.

        :param other: The object to compare the item with.
        :type other: object
        :return: True if the object is an item of the same class with the same data,
            False otherwise.
        :rtype: bool
        """
        is_equal = super().__eq__(other)
        if is_equal is not True or not isinstance(other, _ExtractedItem):
            # Not equal, or the comparison is left to the other object
            return is_equal
        return self._probabilities_option_types() == other._probabilities_option_types()

    def _probabilities_option_types(self) -> set[tuple[type, bool | int | str]] | None:
        """
        Returns the options in ``probabilities`` together with their types,
        which the comparison of dicts does not tell apart (``True == 1``).

        :return: The set of (type, option) pairs, or None if the item
            has no probabilities.
        :rtype: set[tuple[type, bool | int | str]] | None
        """
        if self.probabilities is None:
            return None
        return {(type(option), option) for option in self.probabilities}
