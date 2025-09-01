"""
Abstract base layer for instance and LLM processor types.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from pydantic import ConfigDict, Field, PrivateAttr
from ulid import ULID

from contextgem.internal.base.mixins import _PostInitCollectorMixin
from contextgem.internal.base.serialization import _InstanceSerializer


if TYPE_CHECKING:
    from contextgem.internal.data_models import (
        _LLMCostOutputContainer,
        _LLMUsageOutputContainer,
    )
    from contextgem.internal.typings.aliases import LLMRoleAny


class _AbstractInstanceBase(_PostInitCollectorMixin, _InstanceSerializer, ABC):
    """
    Abstract base for instance-like Pydantic models.
    """

    custom_data: dict = Field(
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


class _AbstractGenericLLMProcessor(_PostInitCollectorMixin, _InstanceSerializer, ABC):
    """
    Abstract base for LLM-backed processors (single or grouped).
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    @property
    @abstractmethod
    def is_group(self) -> bool:
        """
        Abstract property, to be implemented by subclasses.

        Whether the LLM is a single instance or a group.
        """
        pass

    @property
    @abstractmethod
    def list_roles(self) -> list[LLMRoleAny]:
        """
        Abstract property, to be implemented by subclasses.

        Returns the list of all LLM roles in the LLM group or LLM.
        """
        pass

    @abstractmethod
    def _set_private_attrs(self) -> None:
        """
        Abstract method, to be implemented by subclasses.

        Sets private attributes for the LLM group or LLM, e.g. prompts, capabilities, etc.
        """
        pass

    @abstractmethod
    def get_usage(self, *args, **kwargs) -> list[_LLMUsageOutputContainer]:
        """
        Abstract method, to be implemented by subclasses.

        Returns the usage data for the LLM group or LLM.
        """
        pass

    @abstractmethod
    def get_cost(self, *args, **kwargs) -> list[_LLMCostOutputContainer]:
        """
        Abstract method, to be implemented by subclasses.

        Returns the cost data for the LLM group or LLM as a list of
        `_LLMCostOutputContainer` entries. Implementations may accept optional
        filter parameters (e.g., role) where applicable.
        """
        pass

    @abstractmethod
    def reset_usage_and_cost(self) -> None:
        """
        Abstract method, to be implemented by subclasses.

        Resets the usage and cost data for the LLM group or LLM. Implementations
        may support optional filters (e.g., by role) where applicable.
        """
        pass
