# SPDX-FileCopyrightText: Shcherbak AI AS
# SPDX-License-Identifier: Apache-2.0

"""
Module providing base classes for document paragraphs and sentences.
"""

from __future__ import annotations

from typing import Any, Self

from pydantic import Field, field_validator, model_validator

from contextgem.internal.base.instances import _InstanceBase
from contextgem.internal.decorators import _disable_direct_initialization
from contextgem.internal.loggers import logger
from contextgem.internal.typings.types import NonEmptyStr, PageNumber
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
    page_numbers: list[PageNumber] = Field(
        default_factory=list,
        description=(
            "Numbers of the pages of the document that the text is on, from 1; "
            "not sent to the models."
        ),
    )

    @field_validator("page_numbers")
    @classmethod
    def _validate_page_numbers(cls, page_numbers: list[int]) -> list[int]:
        """
        Puts the page numbers in ascending order, without repetitions, e.g. for
        a text that is on two pages and was found in several parts.

        :param page_numbers: The page numbers.
        :type page_numbers: list[int]
        :return: The page numbers in ascending order, each once.
        :rtype: list[int]
        """
        return sorted(set(page_numbers))

    @field_validator("additional_context")
    @classmethod
    def _validate_additional_context(cls, additional_context: str | None) -> str | None:
        """
        Validates the optional 'additional_context' attribute by checking for line breaks
        in the string, if provided. If line breaks are detected, a warning is logged to inform
        the user that they are not kept: the additional context is cleaned before it is
        sent to a model, as the text of a paragraph is, i.e. its whitespace, line breaks
        included, is collapsed to single spaces.

        :param additional_context: The optional string to be validated for line breaks.
        :type additional_context: str | None
        :return: The unmodified 'additional_context' value after validation.
        :rtype: str | None
        """
        if additional_context is not None and _contains_linebreaks(additional_context):
            logger.warning(
                f"Additional context of `{cls.__name__}` contains line breaks, "
                f"which are replaced with spaces when it is sent to a model."
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
class _Paragraph(_ParasAndSentsBase):
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
        if name == "sentences" and getattr(self, name, None):
            # Prevent sentences reassignment once populated,
            # to prevent inconsistencies in analysis.
            raise ValueError(
                f"The attribute `{name}` cannot be changed once populated."
            )
        super().__setattr__(name, value)

    @model_validator(mode="after")
    def _validate_paragraph_post(self) -> Self:
        """
        Verifies that all sentences within the `sentences` attribute, if they exist,
        have their raw text content found within the `raw_text` attribute of the paragraph.

        :return: The validated Paragraph instance.
        :rtype: Self
        :raises ValueError: If any sentence's raw text is not matched within
            the paragraph's raw text.
        """
        if self.sentences and not all(
            i.raw_text in self.raw_text for i in self.sentences
        ):
            raise ValueError("Not all sentences were matched in paragraph text.")

        return self
