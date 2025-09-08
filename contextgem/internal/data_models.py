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
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    StrictBool,
    StrictInt,
    StrictStr,
    field_validator,
    model_validator,
)

from contextgem.internal.base.abstract import _AbstractInstanceBase
from contextgem.internal.base.serialization import _InstanceSerializer
from contextgem.internal.decorators import _expose_in_registry
from contextgem.internal.typings.types import (
    DefaultDecimalField,
    LLMRoleAny,
    MessageRole,
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


@_expose_in_registry
class _Message(_AbstractInstanceBase):
    """
    Represents a single chat message sent to an LLM.

    The message is compatible with OpenAI-style chat format. Content can be either
    a plain string or a multimodal list containing text and image_url parts.

    Example multimodal content:
        [
            {"type": "text", "text": "describe this"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
        ]

    :ivar role: The role of the message, which must be one of "system", "user", "assistant", "tool".
    :vartype role: Literal["system", "user", "assistant", "tool"]
    :ivar content: The content of the message, which can be either a plain string or
        a multimodal list containing text and image_url parts.
    :vartype content: str | list[dict[str, Any]]
    :ivar tool_call_id: Identifier of the tool call this tool reply corresponds to. Must be set
        when ``role == 'tool'``; otherwise must be ``None``. Tool messages must have string content.
    :vartype tool_call_id: str | None
    :ivar tool_calls: A list of serialized tool call descriptors as returned by the model
        (OpenAI-compatible structure). Applicable only for ``assistant`` messages. When present,
        each item is expected to be a dict containing at least the function call payload.
    :vartype tool_calls: list[dict[str, Any]] | None
    :ivar name: The function name for a tool message. Required by providers for
        tool replies. Must be set when ``role == 'tool'``; otherwise must be ``None``.
    :vartype name: str | None
    """

    role: MessageRole = Field(..., frozen=True)
    content: NonEmptyStr | list[dict[NonEmptyStr, Any]] = Field(..., frozen=True)

    # Optional for tool messages: must be set when role == "tool"
    tool_call_id: NonEmptyStr | None = Field(default=None, frozen=True)
    # Optional for assistant messages: present when the assistant requested tool calls
    tool_calls: list[dict[NonEmptyStr, Any]] | None = Field(default=None, frozen=True)
    # Optional for tool messages: function name for the tool response
    name: NonEmptyStr | None = Field(default=None, frozen=True)

    _time_created: datetime = PrivateAttr(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    _response_succeeded: StrictBool | None = PrivateAttr(default=None)

    def __setattr__(self, name: str, value: Any) -> None:
        """
        Custom attribute setter that enforces the rule that `_response_succeeded`
        can only be assigned to `user` messages.

        :param name: The name of the attribute to set
        :type name: str
        :param value: The value to assign to the attribute
        :type value: Any
        :return: None
        :rtype: None
        :raises AttributeError: If the role is not `user` and the attribute
            is `_response_succeeded` (assignment not allowed for other roles).
        """
        if name == "_response_succeeded" and getattr(self, "role", None) != "user":
            raise AttributeError(
                "Assignment to `_response_succeeded` is only allowed "
                "for messages with role 'user'."
            )

        super().__setattr__(name, value)

    @property
    def time_created(self) -> datetime:
        """
        Returns the time created of the instance.
        """
        return self._time_created

    @field_validator("content")
    @classmethod
    def _validate_content(
        cls, content: str | list[dict[str, Any]]
    ) -> str | list[dict[str, Any]]:
        """
        Validates that list-based content follows the multimodal schema.

        :param content: The content to validate. Must be a non-empty string or
            a list of dicts with a 'type' field.
        :type content: str | list[dict[str, Any]]
        :return: The validated content.
        :rtype: str | list[dict[str, Any]]
        :raises ValueError: If the content does not follow the multimodal schema.
        """
        if isinstance(content, list):
            for part in content:
                if not isinstance(part, dict) or "type" not in part:
                    raise ValueError(
                        "Each content part must be a dict with a 'type' field"
                    )
                part_type = part["type"]
                if part_type == "text":
                    if not isinstance(part.get("text"), str):
                        raise ValueError(
                            "Text part must include a non-empty string 'text' field"
                        )
                    if part.get("text", "").strip() == "":
                        raise ValueError(
                            "Text part must include a non-empty string 'text' field"
                        )
                elif part_type == "image_url":
                    image_url = part.get("image_url")
                    if not (
                        isinstance(image_url, dict)
                        and isinstance(image_url.get("url"), str)
                    ):
                        raise ValueError(
                            "image_url part must include 'image_url': {'url': '<non-empty string URL>'}"
                        )
                    if image_url.get("url", "").strip() == "":
                        raise ValueError(
                            "image_url part must include 'image_url': {'url': '<non-empty string URL>'}"
                        )
                else:
                    raise ValueError(f"Unsupported content part type: {part_type}")
        return content

    @model_validator(mode="after")
    def _validate_message_post(self) -> _Message:
        """
        Ensures tool-specific constraints:
        - When role == "tool", tool_call_id must be provided and content must be a string.
        - When role == "tool", name should be provided.
        - When role != "tool", tool_call_id must be None.
        - When role == "assistant", tool_calls may be provided; otherwise, it must be None.
        """
        if self.role == "tool":
            if self.tool_call_id is None:
                raise ValueError(
                    "`tool_call_id` must be set for messages with role 'tool'."
                )
            if isinstance(self.content, list):
                raise ValueError("Tool message content must be a non-empty string.")
            if self.name is None:
                raise ValueError("`name` must be set for messages with role 'tool'.")
        else:
            if self.tool_call_id is not None:
                raise ValueError(
                    "`tool_call_id` is only allowed for messages with role 'tool'."
                )
            if self.name is not None:
                raise ValueError(
                    "`name` is only allowed for messages with role 'tool'."
                )
        if self.tool_calls is not None and self.role != "assistant":
            raise ValueError(
                "`tool_calls` is only allowed for messages with role 'assistant'."
            )
        return self

    def _to_message_dict(self) -> dict[str, Any]:
        """
        Render this message to an OpenAI-compatible dictionary.

        :return: A dict with keys 'role' and 'content' suitable for chat APIs.
        :rtype: dict[str, Any]
        """
        data = {"role": self.role, "content": self.content}
        if self.role == "tool":
            data["tool_call_id"] = self.tool_call_id
            data["name"] = self.name
        if self.role == "assistant" and self.tool_calls is not None:
            data["tool_calls"] = self.tool_calls
        return data
