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
Module for handling document paragraphs.

This module provides the Paragraph class, which represents a structured segment of text
within a document. Paragraphs serve as containers for sentences and maintain the raw text
content of the segment they represent.

The module supports validation to ensure data integrity and provides mechanisms to prevent
inconsistencies during document analysis by restricting certain attribute modifications
after initial assignment.
"""

from __future__ import annotations

from typing import Any

from pydantic import Field, model_validator
from typing_extensions import Self

from contextgem.internal.base.md_text import _MarkdownTextAttributesProcessor
from contextgem.internal.base.paras_and_sents import _ParasAndSentsBase
from contextgem.internal.typings.aliases import NonEmptyStr
from contextgem.public.sentences import Sentence


class Paragraph(_ParasAndSentsBase, _MarkdownTextAttributesProcessor):
    """
    Represents a paragraph of a document with its raw text content and constituent sentences.

    Paragraphs are immutable text segments that can contain multiple sentences. Once sentences
    are assigned to a paragraph, they cannot be changed to maintain data integrity during analysis.

    :ivar raw_text: The complete text content of the paragraph. This value is frozen after initialization.
    :vartype raw_text: str
    :ivar sentences: The individual sentences contained within the paragraph. Defaults to an empty list.
        Cannot be reassigned once populated.
    :vartype sentences: list[Sentence]

    Note:
        Normally, you do not need to construct paragraphs manually, as they are populated automatically
        from document's ``raw_text`` attribute. Only use this constructor for advanced use cases,
        such as when you have a custom paragraph segmentation tool.

    Example:
        .. literalinclude:: ../../../dev/usage_examples/docstrings/paragraphs/def_paragraph.py
            :language: python
            :caption: Paragraph definition
    """

    raw_text: NonEmptyStr = Field(..., frozen=True)
    sentences: list[Sentence] = Field(default_factory=list)

    def __setattr__(self, name: str, value: Any) -> None:
        """
        Sets the attribute of an instance, with additional restrictions on specific attributes.

        :param name: The name of the attribute to set.
        :type name: str
        :param value: The value to assign to the attribute.
        :return: None
        :raises ValueError: If attempting to reassign a restricted attribute
            after it has already been assigned to a *truthy* value.
        """
        if name in ["sentences", "_md_text"] and getattr(self, name, None):
            # Prevent sentences and _md_text reassignment once populated,
            # to prevent inconsistencies in analysis.
            raise ValueError(
                f"The attribute `{name}` cannot be changed once populated."
            )
        if name == "_md_text":
            self._validate_md_text(value)
        super().__setattr__(name, value)

    @model_validator(mode="after")
    def _validate_paragraph_post(self) -> Self:
        """
        Verifies that:
        - all sentences within the `sentences` attribute, if they exist, have their
            raw text content found within the `raw_text` attribute of the paragraph.
        - when `_md_text` is populated, `raw_text` is also populated.

        :return: The validated Paragraph instance.
        :rtype: Self
        :raises ValueError: If any sentence's raw text is not matched within
            the paragraph's raw text, or if `_md_text` is provided without `raw_text`
            being set.
        """
        if self.sentences and not all(
            i.raw_text in self.raw_text for i in self.sentences
        ):
            raise ValueError("Not all sentences were matched in paragraph text.")

        if self._md_text and not self.raw_text:
            raise ValueError(
                "Paragraph's `_md_text` cannot be populated without `raw_text` being set."
            )

        return self
