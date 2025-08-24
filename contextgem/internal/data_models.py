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
Module defining internal data validation models.
"""

from __future__ import annotations

import time
from decimal import Decimal
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, StrictStr

from contextgem.internal.base.serialization import _InstanceSerializer
from contextgem.internal.typings.aliases import (
    DefaultDecimalField,
    LLMRoleAny,
    NonEmptyStr,
)


class _LLMCall(BaseModel):
    """
    Holds data on an individual call to the LLM.

    :ivar timestamp_sent: The epoch timestamp (in ms) when the prompt was sent.
    :vartype timestamp_sent: int
    :ivar prompt_kwargs: The kwargs passed for prompt rendering.
    :vartype prompt_kwargs: dict[str, Any]
    :ivar prompt: The fully rendered input prompt sent to the LLM.
    :vartype prompt: str
    :ivar response: The raw text response returned by the LLM for the given prompt.
        Defaults to None if the response is not yet received.
    :vartype response: str | None
    :ivar timestamp_received: The epoch timestamp (in ms) when the response was received.
                             Defaults to None if the response is not yet received.
    :vartype timestamp_received: int | None
    """

    timestamp_sent: StrictInt = Field(
        default_factory=lambda: int(time.time() * 1000), frozen=True
    )
    prompt_kwargs: dict[NonEmptyStr, Any] = Field(..., frozen=True)
    prompt: NonEmptyStr = Field(..., frozen=True)
    response: StrictStr | None = Field(default=None)
    timestamp_received: StrictInt | None = Field(default=None)

    def _record_response_timestamp(self) -> None:
        """
        Records the timestamp (in ms) when the response is received.

        Sets `timestamp_received` to the current epoch timestamp if not already set.
        """
        if self.timestamp_received is not None:  # Only set if not already set
            raise ValueError("Response already received.")
        self.timestamp_received = int(time.time() * 1000)

    def _get_time_spent(self):
        """
        Calculates the time spent on the LLM call in seconds.

        :return: Time difference between `timestamp_received` and `timestamp_sent` in seconds,
                 as a float limited to 2 decimal points, or None if `timestamp_received` is not set.
        """
        if self.timestamp_received is None:
            return None  # Response not received yet
        return round((self.timestamp_received - self.timestamp_sent) / 1000, 2)


class _LLMUsage(_InstanceSerializer):
    """
    Represents the input and output usage of a LLM.

    :ivar input: Represents the number of tokens used for the input of the LLM.
    :vartype input: int
    :ivar output: Represents the number of tokens generated as output by the LLM.
    :vartype output: int
    :ivar calls: A list of _LLMCall objects representing the data on the individual calls made to the LLM.
    :vartype calls: list[_LLMCall]
    """

    input: StrictInt = Field(
        default=0,
        ge=0,
    )
    output: StrictInt = Field(
        default=0,
        ge=0,
    )
    calls: list[_LLMCall] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class _LLMUsageOutputContainer(BaseModel):
    """
    Container class for storing the usage data storage unit for an LLM,
    i.e. part of the list returned by the LLM's get_usage() method.

    :ivar model: The name or identifier of the model being used.
    :vartype model: str
    :ivar role: The role of the model, which must be one of
        "extractor_text", "reasoner_text", "extractor_vision",
        "reasoner_vision", "extractor_multimodal", "reasoner_multimodal".
    :vartype role: LLMRoleAny
    :ivar is_fallback: Indicates whether the LLM is a fallback model.
    :vartype is_fallback: StrictBool
    :ivar usage: Detailed usage information encapsulated in a `_LLMUsage`
        object.
    :vartype usage: _LLMUsage
    """

    model: NonEmptyStr
    role: LLMRoleAny
    is_fallback: StrictBool
    usage: _LLMUsage

    model_config = ConfigDict(
        extra="forbid", frozen=True
    )  # make immutable once created


class _LLMCost(_InstanceSerializer):
    """
    Represents a structure for tracking the cost associated with an LLM's
    processing inputs, outputs, and the total processing cost.

    :ivar input: Cost associated with processing the input.
    :vartype input: Decimal
    :ivar output: Cost associated with generating the output.
    :vartype output: Decimal
    :ivar total: Total cost combining both input and output processing.
    :vartype total: Decimal
    """

    input: Decimal = DefaultDecimalField
    output: Decimal = DefaultDecimalField
    total: Decimal = DefaultDecimalField

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class _LLMCostOutputContainer(BaseModel):
    """
    Container class for storing the cost data storage unit for an LLM,
    i.e. part of the list returned by the LLM's get_cost() method.

    :ivar model: The name of the language model being used.
    :vartype model: str
    :ivar role: The role of the model in processing, which can be one of:
                "extractor_text", "reasoner_text", "extractor_vision",
                "reasoner_vision", "extractor_multimodal", "reasoner_multimodal".
    :vartype role: LLMRoleAny
    :ivar is_fallback: Indicates if the LLM is a fallback model.
    :vartype is_fallback: bool
    :ivar cost: The _LLMCost object associated with the LLM execution.
    :vartype cost: _LLMCost
    """

    model: NonEmptyStr
    role: LLMRoleAny
    is_fallback: StrictBool
    cost: _LLMCost

    model_config = ConfigDict(
        extra="forbid", frozen=True
    )  # make immutable once created
