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
Module defining the base classes for all instance-specific subclasses.

This module provides foundational classes that serve as the building blocks for
various instance types in the ContextGem framework. It includes the _InstanceBase class
which implements core functionality such as unique ID generation, serialization,
custom data storage, and instance cloning capabilities.
"""

from __future__ import annotations

from abc import ABC
from copy import deepcopy
from typing import TYPE_CHECKING

from pydantic import ConfigDict, Field, PrivateAttr, field_validator
from typing_extensions import Self
from ulid import ULID

from contextgem.internal.base.mixins import _PostInitCollectorMixin
from contextgem.internal.base.serialization import _InstanceSerializer
from contextgem.internal.utils import _is_text_content_empty


if TYPE_CHECKING:
    from contextgem.internal.base.aspects import _Aspect
    from contextgem.internal.base.concepts import _Concept


class _InstanceBase(_PostInitCollectorMixin, _InstanceSerializer, ABC):
    """
    Base class that provides reusable methods for all instance-specific subclasses.

    This class implements core functionality such as unique ID generation, serialization,
    custom data storage, and instance cloning capabilities. It serves as the foundation
    for various instance types in the ContextGem framework.
    """

    custom_data: dict = Field(
        default_factory=dict,
        description="A serializable dictionary for storing additional custom data "
        "related to the instance.",
    )

    _unique_id: str = PrivateAttr(default_factory=lambda: str(ULID()))

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    def clone(self) -> Self:
        """
        Creates and returns a deep copy of the current instance.

        :return: A deep copy of the current instance.
        """
        return deepcopy(self)

    def model_copy(self, *args, **kwargs):
        """
        Raises NotImplementedError to redirect users to use clone() method instead.

        This method is intentionally disabled in favor of the clone() method which
        provides clearer semantics for creating deep copies of instances.

        :param args: Positional arguments (ignored).
        :param kwargs: Keyword arguments (ignored).
        :raises NotImplementedError: Always raised to direct users to use clone().
        """
        raise NotImplementedError("Use `clone()` instead")

    @property
    def unique_id(self) -> str:
        """
        Returns the ULID of the instance.
        """
        return self._unique_id

    @field_validator("raw_text", check_fields=False)
    @classmethod
    def _validate_raw_text(cls, raw_text: str | None) -> str | None:
        """
        Validates that the raw text is not empty.

        This validation goes beyond Pydantic's standard string constraints by also
        checking for invisible Unicode characters and other control characters that
        would make the text effectively empty despite having a non-zero length.

        :param raw_text: The raw text to validate.
        :type raw_text: str | None. None is allowed for Document instances,
            e.g. when a document has only images.
        :return: The original raw text if it is not empty.
        :rtype: str | None
        :raises ValueError: If the raw text is empty or contains only whitespace/control characters.
        """
        if raw_text is not None and _is_text_content_empty(raw_text):
            raise ValueError(
                "`raw_text` cannot be empty or contain only whitespace/control characters. "
                "This includes text with only spaces, tabs, newlines, invisible Unicode characters, "
                "or other control characters."
            )
        return raw_text

    @field_validator(
        "aspects",
        "concepts",
        "paragraphs",
        "sentences",
        "reference_paragraphs",
        "reference_sentences",
        "images",
        "examples",
        check_fields=False,
    )
    @classmethod
    def _validate_list_uniqueness(
        cls, instances: list[_InstanceBase]
    ) -> list[_InstanceBase]:
        """
        Validates that all elements in the provided list have unique IDs.

        :param instances: List of `_InstanceBase` objects to validate.
        :type instances: list[_InstanceBase]
        :return: The original list if all elements have unique IDs.
        :rtype: list[_InstanceBase]
        :raises ValueError: If duplicate elements based on unique IDs are found in the list.
        """
        ids: list[str] = [i.unique_id for i in instances]
        if instances and len(set(ids)) < len(ids):
            raise ValueError(
                f"List elements of class {instances[0].__class__.__name__} contain duplicates."
            )
        return instances

    @field_validator("aspects", "concepts", check_fields=False)
    @classmethod
    def _validate_text_and_description_uniqueness(
        cls, instances: list[_Aspect] | list[_Concept]
    ) -> list[_Aspect] | list[_Concept]:
        """
        Validates the list field to ensure that the instances on the list have
        unique names and descriptions (case-insensitive).

        Outputs a deepcopy of the input list to prevent any modifications to the state of
        the original instances that may be reusable across multiple documents.

        :param instances: The list of `_Aspect` or `_Concept` objects to validate.
        :type instances: list[_Aspect] | list[_Concept]
        :return: The validated list of `_Aspect` or `_Concept` objects if all conditions pass.
        :rtype: list[_Aspect] | list[_Concept]
        :raises ValueError: If there are duplicate names or descriptions in the objects
            (case-insensitive).
        """

        if instances and (
            len(set([i.name.lower() for i in instances])) < len(instances)
            or len(set([i.description.lower() for i in instances])) < len(instances)
        ):
            raise ValueError(
                f"{instances[0].__class__.__name__}s of the document must have "
                f"unique names and unique descriptions (case-insensitive)."
            )
        return deepcopy(instances)
