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
Module for processing and managing markdown text attributes.
"""

from typing import Optional

from pydantic import BaseModel, PrivateAttr

from contextgem.internal.typings.aliases import NonEmptyStr


class _MarkdownTextAttributesProcessor(BaseModel):
    """
    Base class for processing and managing markdown text attributes.
    """

    _md_text: Optional[NonEmptyStr] = PrivateAttr(default=None)

    def _validate_md_text(self, value: Optional[NonEmptyStr]) -> None:
        """
        Validates the markdown text content.

        :param value: The markdown text content to validate. Must be a non-empty string or None.
        :type value: Optional[NonEmptyStr]
        :raises ValueError: If the value is an empty string or not a string.
        :return: None
        """
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise ValueError("Markdown text must be a non-empty string")
