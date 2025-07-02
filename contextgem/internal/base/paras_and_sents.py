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

This module defines the foundational classes used for representing and managing
paragraphs and sentences within documents. It includes the _ParasAndSentsBase class
which serves as the common ancestor for paragraph and sentence implementations,
providing shared validation and context handling functionality.
"""

from __future__ import annotations

from typing import Optional

from pydantic import Field, field_validator

from contextgem.internal.base.instances import _InstanceBase
from contextgem.internal.loggers import logger
from contextgem.internal.typings.aliases import NonEmptyStr
from contextgem.internal.utils import _contains_linebreaks


class _ParasAndSentsBase(_InstanceBase):
    """
    Base class for document paragraphs and sentences.

    This class provides common functionality for representing textual elements within documents,
    serving as the foundation for paragraph and sentence implementations. It inherits from
    _InstanceBase and implements validation for contextual information.

    :ivar additional_context: Optional supplementary information for the paragraph or sentence.
        Should be a non-empty string without linebreaks. Defaults to None.
    :vartype additional_context: Optional[NonEmptyStr]
    """

    additional_context: Optional[NonEmptyStr] = Field(default=None)

    @field_validator("additional_context")
    @classmethod
    def _validate_additional_context(
        cls, additional_context: Optional[str]
    ) -> Optional[str]:
        """
        Validates the optional 'additional_context' attribute by checking for line breaks
        in the string, if provided. If line breaks are detected, a warning is logged to inform
        the user that such input may lead to unexpected behavior, as the LLM may not be able
        to process such input correctly due to the structure of the prompt.

        :param additional_context: The optional string to be validated for line breaks.
        :type additional_context: Optional[str]
        :return: The unmodified 'additional_context' value after validation.
        :rtype: Optional[str]
        """
        if additional_context is not None and _contains_linebreaks(additional_context):
            logger.warning(
                f"Additional context of `{cls.__name__}` contains line breaks. "
                f"This may cause unexpected behavior."
            )
        return additional_context
