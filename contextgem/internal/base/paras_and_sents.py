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
Module providing base classes for document paragraphs and sentences.
"""

from __future__ import annotations

from typing import Any

from pydantic import Field, field_validator, model_validator
from typing_extensions import Self

from contextgem.internal.base.instances import _InstanceBase
from contextgem.internal.base.md_text import _MarkdownTextAttributesProcessor
from contextgem.internal.decorators import _disable_direct_initialization
from contextgem.internal.loggers import logger
from contextgem.internal.typings.types import NonEmptyStr
from contextgem.internal.utils import _contains_linebreaks


class _ParasAndSentsBase(_InstanceBase):
    """
    Base class for paragraph and sentence models.
    """

    additional_context: NonEmptyStr | None = Field(
        default=None,
        description=(
            "Optional supplementary information without linebreaks; used to enrich prompts."
        ),
    )

    @field_validator("additional_context")
    @classmethod
    def _validate_additional_context(cls, additional_context: str | None) -> str | None:
        """
        Validates the optional 'additional_context' attribute by checking for line breaks
        in the string, if provided. If line breaks are detected, a warning is logged to inform
        the user that such input may lead to unexpected behavior, as the LLM may not be able
        to process such input correctly due to the structure of the prompt.

        :param additional_context: The optional string to be validated for line breaks.
        :type additional_context: str | None
        :return: The unmodified 'additional_context' value after validation.
        :rtype: str | None
        """
        if additional_context is not None and _contains_linebreaks(additional_context):
            logger.warning(
                f"Additional context of `{cls.__name__}` contains line breaks. "
                f"This may cause unexpected behavior."
            )
        return additional_context


@_disable_direct_initialization
class _Sentence(_ParasAndSentsBase):
    """
    Internal implementation of the Sentence class.
    """

    raw_text: NonEmptyStr = Field(
        ...,
        frozen=True,
        description="Raw text content of the sentence; immutable after initialization.",
    )


@_disable_direct_initialization
class _Paragraph(_ParasAndSentsBase, _MarkdownTextAttributesProcessor):
    """
    Internal implementation of the Paragraph class.
    """

    raw_text: NonEmptyStr = Field(
        ...,
        frozen=True,
        description="Raw text content of the paragraph; immutable after initialization.",
    )
    sentences: list[_Sentence] = Field(
        default_factory=list,
        description=(
            "Sentences contained in this paragraph; cannot be reassigned once populated."
        ),
    )

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
