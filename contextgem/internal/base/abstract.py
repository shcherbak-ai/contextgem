# SPDX-FileCopyrightText: Shcherbak AI AS
# SPDX-License-Identifier: Apache-2.0

"""
Abstract base layer for instance and model processor types.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from pydantic import ConfigDict, Field, PrivateAttr
from ulid import ULID

from contextgem.internal.base.mixins import (
    _ApiKeyRedactionMixin,
    _AssignmentRollbackMixin,
    _PostInitCollectorMixin,
)
from contextgem.internal.base.serialization import _InstanceSerializer
from contextgem.internal.typings.types import JSONDictField


if TYPE_CHECKING:
    from contextgem.internal.data_models import (
        _ModelCostOutputContainer,
        _ModelUsageOutputContainer,
    )


class _AbstractInstanceBase(
    _PostInitCollectorMixin, _AssignmentRollbackMixin, _InstanceSerializer, ABC
):
    """
    Abstract base for instance-like Pydantic models.
    """

    custom_data: JSONDictField = Field(
        default_factory=dict,
        description="A serializable dictionary for storing additional custom data "
        "related to the instance.",
    )

    _unique_id: str = PrivateAttr(default_factory=lambda: str(ULID()))

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    @property
    def unique_id(self) -> str:
        """
        Returns the ULID of the instance.
        """
        return self._unique_id


class _AbstractGenericModelProcessor(
    _ApiKeyRedactionMixin,
    _PostInitCollectorMixin,
    _AssignmentRollbackMixin,
    _InstanceSerializer,
    ABC,
):
    """
    Abstract base for model-backed processors (single or grouped).

    A validation error of a processor does not give the API key in the inputs
    that it reports (see ``_ApiKeyRedactionMixin``).
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    @property
    @abstractmethod
    def is_group(self) -> bool:
        """
        Abstract property, to be implemented by subclasses.

        Whether the model is a single instance or a group.
        """
        pass

    @property
    @abstractmethod
    def list_roles(self) -> list[str | None]:
        """
        Abstract property, to be implemented by subclasses.

        Returns the list of all model roles in the model group or single model.
        None on the list stands for a model without a role.
        """
        pass

    @abstractmethod
    def _set_private_attrs(self) -> None:
        """
        Abstract method, to be implemented by subclasses.

        Sets private attributes for the model group or single model, e.g. prompts, capabilities, etc.
        """
        pass

    @abstractmethod
    def get_usage(self, *args, **kwargs) -> list[_ModelUsageOutputContainer]:
        """
        Abstract method, to be implemented by subclasses.

        Returns the usage data for the model group or single model.
        """
        pass

    @abstractmethod
    def get_cost(self, *args, **kwargs) -> list[_ModelCostOutputContainer]:
        """
        Abstract method, to be implemented by subclasses.

        Returns the cost data for the model group or single model as a list of
        `_ModelCostOutputContainer` entries. Implementations may accept optional
        filter parameters (e.g., role) where applicable.
        """
        pass

    @abstractmethod
    def reset_usage_and_cost(self) -> None:
        """
        Abstract method, to be implemented by subclasses.

        Resets the usage and cost data for the model group or single model. Implementations
        may support optional filters (e.g., by role) where applicable.
        """
        pass

    @abstractmethod
    def _warn_tools_ignored_if_enabled(self) -> None:
        """
        Abstract method, to be implemented by subclasses.

        Should warn that tools are ignored during extraction
        workflows if tools are configured. Tools are supported only in ``llm.chat(...)``.
        """
        pass
