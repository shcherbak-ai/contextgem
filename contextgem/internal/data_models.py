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
    MessageRole,
    ModelRole,
    NonEmptyStr,
)


class _ModelCall(BaseModel):
    """
    Holds data on an individual call to the model, i.e. on one request sent to
    an LLM or to a decision model.

    :ivar timestamp_sent: The epoch timestamp (in ms) when the request was sent.
    :vartype timestamp_sent: int
    :ivar prompt_kwargs: The kwargs passed for prompt rendering. Empty for a call
        whose prompt is not rendered from a template: a chat message, a request
        that returns the results of tool calls, or a request to a decision model.
    :vartype prompt_kwargs: dict[str, Any]
    :ivar prompt: What was sent to the model. For an LLM, the fully rendered input
        prompt, or ``"<tools_round_trip>"`` for a request that returns the results
        of tool calls. For a decision model, the request as JSON, i.e. the model,
        the text and the questions.
    :vartype prompt: str
    :ivar response: What the model returned. For an LLM, the raw text response for
        the given prompt. For a decision model, the answers to the questions as JSON.
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

    # Tokens used by the call, kept to price each call on its own.
    # Zero if the response is not received.
    _input_tokens: int = PrivateAttr(default=0)
    _output_tokens: int = PrivateAttr(default=0)
    _cache_read_tokens: int = PrivateAttr(default=0)
    _cache_write_tokens: int = PrivateAttr(default=0)

    # Whether the response was cut off at the max output tokens,
    # as reported by the model provider (finish reason "length")
    _output_truncated: bool = PrivateAttr(default=False)

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
        Calculates the time spent on the model call in seconds.

        :return: Time difference between `timestamp_received` and `timestamp_sent` in seconds,
                 as a float limited to 2 decimal points, or None if `timestamp_received` is not set.
        """
        if self.timestamp_received is None:
            return None  # Response not received yet
        return round((self.timestamp_received - self.timestamp_sent) / 1000, 2)


class _ModelUsage(_InstanceSerializer):
    """
    Represents the input and output usage of a model.

    :ivar input: Represents the number of tokens used for the input of the model,
        including the tokens read from and written to the prompt cache of the model provider.
    :vartype input: int
    :ivar output: Represents the number of tokens generated as output by the model.
    :vartype output: int
    :ivar cache_read: Represents the number of input tokens read from the prompt cache
        of the model provider, i.e. the part of ``input`` that the provider has served
        from the cache. Zero if the provider does not report cached tokens.
    :vartype cache_read: int
    :ivar cache_write: Represents the number of input tokens written to the prompt cache
        of the model provider, i.e. the part of ``input`` that the provider has stored
        in the cache. Zero if the provider does not report such tokens.
    :vartype cache_write: int
    :ivar calls: A list of _ModelCall objects representing the data on the individual calls made to the model.
    :vartype calls: list[_ModelCall]
    """

    input: StrictInt = Field(
        default=0,
        ge=0,
    )
    output: StrictInt = Field(
        default=0,
        ge=0,
    )
    cache_read: StrictInt = Field(
        default=0,
        ge=0,
    )
    cache_write: StrictInt = Field(
        default=0,
        ge=0,
    )
    calls: list[_ModelCall] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class _ModelUsageOutputContainer(BaseModel):
    """
    Container class for storing the usage data storage unit for a model,
    i.e. part of the list returned by the model's get_usage() method.

    :ivar model: The name or identifier of the model being used.
    :vartype model: str
    :ivar role: The role label of the model, or None if the model has no role.
    :vartype role: str | None
    :ivar is_fallback: Indicates whether the model is a fallback model.
    :vartype is_fallback: StrictBool
    :ivar usage: Detailed usage information encapsulated in a `_ModelUsage`
        object.
    :vartype usage: _ModelUsage
    """

    model: NonEmptyStr
    role: ModelRole | None
    is_fallback: StrictBool
    usage: _ModelUsage

    model_config = ConfigDict(
        extra="forbid", frozen=True
    )  # make immutable once created


class _ModelCost(_InstanceSerializer):
    """
    Represents a structure for tracking the cost associated with a model's
    processing inputs, outputs, and the total processing cost.

    :ivar input: Cost associated with processing the input, including the cost
        of the input tokens read from and written to the prompt cache of the model provider.
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


class _ModelCostOutputContainer(BaseModel):
    """
    Container class for storing the cost data storage unit for a model,
    i.e. part of the list returned by the model's get_cost() method.

    :ivar model: The name of the model being used.
    :vartype model: str
    :ivar role: The role label of the model, or None if the model has no role.
    :vartype role: str | None
    :ivar is_fallback: Indicates if the model is a fallback model.
    :vartype is_fallback: bool
    :ivar cost: The _ModelCost object associated with the model execution.
    :vartype cost: _ModelCost
    """

    model: NonEmptyStr
    role: ModelRole | None
    is_fallback: StrictBool
    cost: _ModelCost

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
        a multimodal list containing text and image_url parts. Can be ``None`` only for
        an ``assistant`` message that has ``tool_calls``, i.e. when the model requested
        tool calls without returning a text.
    :vartype content: str | list[dict[str, Any]] | None
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
    # None is allowed only for assistant messages that have tool calls
    content: NonEmptyStr | list[dict[NonEmptyStr, Any]] | None = Field(..., frozen=True)

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
        cls, content: str | list[dict[str, Any]] | None
    ) -> str | list[dict[str, Any]] | None:
        """
        Validates that list-based content follows the multimodal schema.

        :param content: The content to validate. Must be a non-empty string,
            a list of dicts with a 'type' field, or None.
        :type content: str | list[dict[str, Any]] | None
        :return: The validated content.
        :rtype: str | list[dict[str, Any]] | None
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
        - Content may be None only when role == "assistant" and tool_calls are provided.
        """
        if self.content is None and not (self.role == "assistant" and self.tool_calls):
            raise ValueError(
                "`content` can be None only for messages with role 'assistant' "
                "that have `tool_calls`."
            )
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
            The content is None for an assistant message that has tool calls and no text.
        :rtype: dict[str, Any]
        """
        data = {"role": self.role, "content": self.content}
        if self.role == "tool":
            data["tool_call_id"] = self.tool_call_id
            data["name"] = self.name
        if self.role == "assistant" and self.tool_calls is not None:
            data["tool_calls"] = self.tool_calls
        return data
