# SPDX-FileCopyrightText: Shcherbak AI AS
# SPDX-License-Identifier: Apache-2.0

"""
Module defining the internal implementations of LLMs and chat sessions.

This module provides the ``_DocumentLLM`` class, which holds the configuration of
a specific LLM and handles the requests to it (prompts, retries, fallbacks, chat
with tools, usage and cost tracking), and the ``_ChatSession`` class, which preserves
the message history of a chat across turns. What an LLM shares with the other kinds
of models (``_DocumentModel``), and the extraction logic that it shares with a model
group, are defined in the ``models`` module of this package.
"""

from __future__ import annotations

import asyncio
import functools
import hashlib
import inspect
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

from aiolimiter import AsyncLimiter
from fastjsonschema import validate as _jsonschema_validate
from fastjsonschema.exceptions import JsonSchemaException as _JSONSchemaValidationError
from jinja2 import Template
from pydantic import (
    Field,
    PrivateAttr,
    StrictBool,
    StrictFloat,
    StrictInt,
    field_validator,
)

from contextgem.internal.base.images import _Image
from contextgem.internal.base.instances import _InstanceBase
from contextgem.internal.base.models import _LOCAL_MODEL_PROVIDERS, _DocumentModel
from contextgem.internal.base.serialization import _is_deserialization_in_progress
from contextgem.internal.base.utils import _is_registered_tool
from contextgem.internal.data_models import (
    _Message,
    _ModelCall,
    _ModelUsage,
)
from contextgem.internal.decorators import _disable_direct_initialization
from contextgem.internal.exceptions import (
    LLMToolLoopLimitError,
    ModelAPIError,
    ModelRefusalError,
    _ToolReturnValueError,
)
from contextgem.internal.loggers import logger
from contextgem.internal.suppressions import _suppress_litellm_warnings_context
from contextgem.internal.typings.types import (
    AsyncCalsAndKwargs,
    DefaultPromptType,
    JSONDictField,
    LanguageRequirement,
    ModelRefusal,
    NonEmptyStr,
    ReasoningEffort,
    ToolHandler,
    ToolRegistration,
)
from contextgem.internal.utils import (
    _MODEL_API_RETRIES_ATTEMPTED_ATTR,
    _get_llm_cache_token_counts,
    _get_model_api_retry_delay,
    _get_model_refusal_msg,
    _get_template,
    _is_retryable_model_api_error,
    _model_call_result_is_api_failure,
    _run_async_calls,
    _run_sync,
    _setup_jinja2_template,
    _warn_from_caller_code,
)


with _suppress_litellm_warnings_context():
    import litellm


litellm.suppress_debug_info = True

# The params of an LLM that declare the capabilities of its model, with the function
# of litellm that detects a capability where its param is not set, and the name
# of the capability in the messages
_LLM_CAPABILITY_PARAMS: dict[str, tuple[str, str]] = {
    "supports_vision": ("supports_vision", "vision"),
    "supports_reasoning": ("supports_reasoning", "reasoning"),
    "supports_tools": ("supports_function_calling", "tool calls"),
    "supports_parallel_tool_calls": (
        "supports_parallel_function_calling",
        "parallel tool calls",
    ),
}

# The models whose input tokens could not be counted in this process for want of
# the token encodings, which tiktoken downloads where litellm has no valid copy of
# them, e.g. without network access: the input tokens of their requests are not
# counted again, as each count would try the download again
_MODELS_WITHOUT_TOKEN_COUNT: set[str] = set()

# Where the documentation says how to prepare the token encodings without
# network access
_OFFLINE_ENCODINGS_DOCS_URL = (
    "https://contextgem.shcherbak.ai/testing/#no-api-keys-in-tests"
)

# The advice on a model that litellm does not know, for the messages on the token
# limits of its requests that could not be checked
_REGISTER_MODEL_ADVICE = (
    "If litellm does not know the model, e.g. one served by your own server, "
    "register it with `litellm.register_model()`, with its limits: see "
    "https://contextgem.shcherbak.ai/llms/llm_config/#explicit-capability-declaration."
)


@_disable_direct_initialization
class _DocumentLLM(_DocumentModel):
    """
    Internal implementation of the ``DocumentLLM`` class.
    """

    # LLM config
    deployment_id: NonEmptyStr | None = Field(
        default=None,
        description="Deployment ID for the model (e.g., for Azure OpenAI).",
    )
    api_version: NonEmptyStr | None = Field(
        default=None,  # specific to Azure OpenAI
        description="API version; used by some providers (e.g., Azure OpenAI).",
    )
    system_message: str | None = Field(
        default=None,
        description="System prompt to prime the model; defaults to framework message if unset.",
    )
    max_tokens: StrictInt | None = Field(
        default=None,
        gt=0,
        description=(
            "Maximum output tokens per request, including the reasoning tokens "
            "of reasoning (CoT-capable) models. Defaults to None, i.e. no limit "
            "is sent, and the default of the model applies."
        ),
    )
    reasoning_effort: ReasoningEffort | None = Field(
        default=None,
        description=(
            "Reasoning effort for CoT-capable models: 'none' | 'minimal' | 'low' | "
            "'medium' | 'high' | 'xhigh' | 'max'. Support for specific values varies "
            "by model and provider."
        ),
    )  # for reasoning (CoT-capable) models
    max_retries_failed_request: StrictInt = Field(
        default=0,
        ge=0,  # provider-specific
        description=(
            "Provider SDK retry count for failed requests; "
            "applies to the first attempt of a request only."
        ),
    )
    max_retries_invalid_data: StrictInt = Field(
        default=3,
        ge=0,
        description=(
            "Retries when the model returns invalid or unparsable data; "
            "a fallback LLM has its own, after it is asked for the first time. "
            "A refusal of the model or its provider is retried as well, also "
            "a request or an answer that the provider blocks. In chat, a request "
            "whose answer is empty or refused without a text is sent again."
        ),
    )
    output_language: LanguageRequirement = Field(
        default="en",
        description=(
            "Language for generated textual output (justifications, explanations). "
            "'en' forces English; 'adapt' matches document/image language. "
            "Applies to the extraction prompts, and to the default system message: "
            "with a custom system message, to the extraction prompts only."
        ),
    )
    temperature: StrictFloat | None = Field(
        default=None,
        ge=0,
        description=(
            "Sampling temperature (0 or higher); higher values increase randomness. "
            "Defaults to None, i.e. the param is not sent, and the default "
            "of the model applies."
        ),
    )
    top_p: StrictFloat | None = Field(
        default=None,
        ge=0,
        description=(
            "Nucleus sampling [0..1]; alternative to temperature. "
            "Defaults to None, i.e. the param is not sent. Some providers "
            "(e.g. Claude 4.x) reject requests that set both."
        ),
    )
    seed: StrictInt | None = Field(
        default=None,
        description=(
            "Random seed for sampling. Support depends on the model provider. "
            "Defaults to None, i.e. the param is not sent."
        ),
    )
    # Tool calling params (to be used only in chat() calls)
    # Sequence type for type checking: a list or a tuple of functions is accepted.
    # The value is validated to a list of the functions.
    tools: Sequence[ToolHandler] | None = Field(
        default=None,
        description=(
            "Tools available to the model: functions decorated with `@register_tool`. "
            "The name, the description and the parameters of each tool are taken "
            "from the function. "
            "Passed to litellm during chat() calls only. "
            "Ignored when using .extract_*() methods."
        ),
    )
    tool_choice: str | JSONDictField | None = Field(
        default=None,
        description=(
            "Tool choice for the model. "
            "Passed to litellm during chat() calls only. "
            "Ignored when using .extract_*() methods."
        ),
    )
    parallel_tool_calls: bool | None = Field(
        default=None,
        description=(
            "Whether to enable parallel tool calls during tool usage. "
            "Passed to litellm during chat() calls only. "
            "Ignored when using .extract_*() methods."
        ),
    )
    tool_max_rounds: StrictInt = Field(
        default=10,
        ge=1,
        description=(
            "Maximum number of tool execution rounds per LLM request. "
            "Prevents infinite or excessively long tool chains."
        ),
    )
    # Capabilities of the model (see `_LLM_CAPABILITY_PARAMS`), to be set where
    # the detection of litellm is not accurate for the model
    supports_vision: StrictBool | None = Field(
        default=None,
        description=(
            "Whether the model takes images as input. Defaults to None, i.e. "
            "detected with litellm.supports_vision() for the model."
        ),
    )
    supports_reasoning: StrictBool | None = Field(
        default=None,
        description=(
            "Whether the model is a reasoning (CoT-capable) model. Defaults to None, "
            "i.e. detected with litellm.supports_reasoning() for the model."
        ),
    )
    supports_tools: StrictBool | None = Field(
        default=None,
        description=(
            "Whether the model supports tool calls. Defaults to None, i.e. "
            "detected with litellm.supports_function_calling() for the model."
        ),
    )
    supports_parallel_tool_calls: StrictBool | None = Field(
        default=None,
        description=(
            "Whether the model supports parallel tool calls. Defaults to None, i.e. "
            "detected with litellm.supports_parallel_function_calling() for the model."
        ),
    )

    # Prompts
    _extract_aspect_items_prompt: Template = PrivateAttr()
    _extract_concept_items_prompt: Template = PrivateAttr()

    # The capabilities of the model detected with litellm, by their params, for
    # the params that are not set, and the model they were detected for
    _detected_capabilities: dict[str, bool] = PrivateAttr(default_factory=dict)
    _capabilities_detected_for: str | None = PrivateAttr(default=None)

    # The checks of the token limits of the requests that could not be made, by
    # the model and the limit, which are logged once (see `_log_skipped_token_check`)
    _skipped_token_checks_logged: set[tuple[str, str]] = PrivateAttr(
        default_factory=set
    )

    # Tool registry: name -> ToolRegistration
    _tool_registry: dict[str, ToolRegistration] = PrivateAttr(default_factory=dict)

    @property
    def _supports_vision(self) -> bool:
        """
        Returns whether the model takes images as input, as set with
        ``supports_vision`` or detected with litellm.

        :return: Whether the model supports vision.
        :rtype: bool
        """
        return self._has_capability("supports_vision")

    @property
    def _supports_reasoning(self) -> bool:
        """
        Returns whether the model is a reasoning (CoT-capable) model, as set with
        ``supports_reasoning`` or detected with litellm.

        :return: Whether the model supports reasoning.
        :rtype: bool
        """
        return self._has_capability("supports_reasoning")

    @property
    def _supports_tools(self) -> bool:
        """
        Returns whether the model supports tool calls, as set with
        ``supports_tools`` or detected with litellm.

        :return: Whether the model supports tool calls.
        :rtype: bool
        """
        return self._has_capability("supports_tools")

    @property
    def _supports_parallel_tool_calls(self) -> bool:
        """
        Returns whether the model supports parallel tool calls, as set with
        ``supports_parallel_tool_calls`` or detected with litellm.

        :return: Whether the model supports parallel tool calls.
        :rtype: bool
        """
        return self._has_capability("supports_parallel_tool_calls")

    @property
    def _kind_name(self) -> str:
        """
        Returns the name of the kind of the model, as used in the messages of
        the logic that models of all kinds share.

        :return: The name of the kind of the model.
        :rtype: str
        """
        return "LLM"

    @property
    def _invalid_output_desc(self) -> str:
        """
        Returns what the LLM is said to have returned when its output cannot
        be used, as worded in the messages on the extraction that fails for it:
        an output that is not JSON, and one that is valid JSON but does not match
        the answer format or the values that the aspects and concepts accept.

        :return: The description of an output of the LLM that cannot be used.
        :rtype: str
        """
        return "invalid data"

    @property
    def _retries_invalid_data(self) -> int:
        """
        Returns the number of times that a call is made again when the LLM returns
        invalid data for it, as set for the LLM. A fallback LLM has the same number
        of retries, after the first call that is made with it.

        :return: The number of retries of a call with invalid data returned.
        :rtype: int
        """
        return self.max_retries_invalid_data

    @property
    def _provider_retries_failed_request(self) -> int:
        """
        Returns the number of retries of a failed request that are left to the SDK
        of the LLM provider, as set for the LLM.

        :return: The number of retries of a failed request by the provider's SDK.
        :rtype: int
        """
        return self.max_retries_failed_request

    @property
    def _fallback_llms(self) -> list[_DocumentLLM]:
        """
        Returns the fallback models of the LLM that are LLMs themselves, i.e. those
        that can replace the LLM in a chat. A fallback model of another kind, such
        as a decision model, generates no text: it replaces the LLM only in
        the extraction of what it can extract.

        :return: The fallback LLMs, in the order they are tried.
        :rtype: list[_DocumentLLM]
        """
        return [
            fallback_model
            for fallback_model in self.fallback_models
            if isinstance(fallback_model, _DocumentLLM)
        ]

    def _log_kind_specific_info(self) -> None:
        """
        Logs the information that applies to specific LLM providers, when the LLM
        is created.

        :return: None
        :rtype: None
        """
        # Log helpful message for local model providers that may use smaller models
        if any(self.model.startswith(provider) for provider in _LOCAL_MODEL_PROVIDERS):
            logger.info(
                "Using local model provider. If you experience issues like JSON validation errors "
                "with smaller models, see our troubleshooting guide: "
                "https://contextgem.shcherbak.ai/optimizations/optimization_small_llm_troubleshooting/"
            )

        # Recommend `ollama_chat` prefix for better responses for Ollama models
        # (text-only processing, as the ollama_chat/ does not yet support image inputs)
        if self.model.startswith("ollama/"):
            logger.info(
                "For better responses with Ollama models when processing text only, "
                "consider using 'ollama_chat/' prefix instead of 'ollama/', as recommended "
                "by LiteLLM: https://docs.litellm.ai/docs/providers/ollama. "
                "Keep the 'ollama/' prefix for processing images."
            )

    def _set_kind_specific_private_attrs(self) -> None:
        """
        Initializes and configures what only an LLM has: the default system message,
        unless a system message is given, the prompts and the capabilities that
        are not set.

        :return: None
        :rtype: None
        """
        if self.system_message is None:
            self._set_system_message()
        self._set_prompts()
        self._update_detected_capabilities()

    def _check_request_params(self, *, use_tools: bool = False) -> None:
        """
        Checks the params that the requests of the LLM are sent with, as they are
        checked when a request is built, without sending a request: the max output
        tokens against the max output of the model, and the request config.

        :param use_tools: Whether the requests are sent with the tools of the LLM,
            i.e. in a chat. Defaults to False.
        :type use_tools: bool
        :return: None
        :rtype: None
        :raises ValueError: If a param of the requests is invalid, e.g. ``max_tokens``
            above the max output of the model.
        """
        self._validate_output_tokens()
        self._build_request_config(use_tools=use_tools)

    def __setattr__(self, name: str, value: Any) -> None:
        """
        Assigns the value to the attribute, keeping the default system message
        in line with the output language once the LLM is created.

        ``None`` assigned to ``system_message`` stands for the default system message,
        as when the LLM is created without a system message. When ``output_language``
        is assigned to an LLM that has the default system message, the default system
        message is rendered for the new output language in the same assignment, as it
        depends on the output language. A custom system message, including an empty
        one, is kept as it is.

        :param name: The name of the attribute to set.
        :type name: str
        :param value: The value to assign to the attribute.
        :type value: Any
        :return: None
        :rtype: None
        """
        if not self._private_attrs_initialized:
            # The LLM is being created: its default system message is set once
            # its params are validated
            super().__setattr__(name, value)
            return

        if name == "system_message" and value is None:
            value = self._get_default_system_message()
        elif (
            name == "output_language"
            and value != self.output_language
            and self.system_message == self._get_default_system_message()
        ):
            # Both params are changed in one assignment, so that the checks that
            # involve both, e.g. the comparison with the fallback LLMs, see them
            # in line, and both are restored if the new output language is rejected
            fields_snapshot = self._get_fields_snapshot()
            # Validated together with the output language, by the assignment below
            self.__dict__["system_message"] = self._get_default_system_message(value)
            try:
                super().__setattr__(name, value)
            except Exception:
                self._restore_fields(fields_snapshot)
                raise
            return

        super().__setattr__(name, value)

    def chat(
        self,
        prompt: str,
        *,
        images: list[_Image] | None = None,
        chat_session: _ChatSession | None = None,
    ) -> str:
        """
        Synchronously sends a prompt to the LLM and gets a response.
        For models supporting vision, attach images to the prompt if needed.

        This method allows direct interaction with the LLM by submitting your own prompt.

        :param prompt: The input prompt to send to the LLM
        :type prompt: str
        :param images: Optional list of Image instances for vision queries
        :type images: list[Image] | None
        :param chat_session: Optional stateful chat session to preserve and use history.
        :type chat_session: _ChatSession | None
        :return: The LLM's response. A response cut off at the max output tokens
            is returned incomplete, with a warning logged. A refused response
            that has a text is returned as it is, with a warning logged.
        :rtype: str
        :raises ValueError: If the prompt is empty or not a string
        :raises ValueError: If images parameter is not a list of Image instances
        :raises ValueError: If images are provided but the model doesn't support vision
        :raises ValueError: If the request params of a fallback LLM are invalid, e.g.
            parallel tool calls are set for a fallback LLM that does not support them
        :raises ValueError: If the prompt exceeds the context window of the LLM, and
            no fallback LLM is left to try
        :raises ModelAPIError: If the request to the LLM fails, or the LLM returns
            an empty response, and no fallback LLM is left to try
        :raises ModelRefusalError: If the LLM or its provider declines to answer
            without a text, or the provider blocks the request for its content,
            and no fallback LLM is left to try
        :raises LLMToolLoopLimitError: If the LLM keeps calling tools after
            ``tool_max_rounds`` rounds of tool calls
        """
        return _run_sync(
            self.chat_async(prompt=prompt, images=images, chat_session=chat_session)
        )

    async def chat_async(
        self,
        prompt: str,
        *,
        images: list[_Image] | None = None,
        chat_session: _ChatSession | None = None,
    ) -> str:
        """
        Asynchronously sends a prompt to the LLM and gets a response.
        For models supporting vision, attach images to the prompt if needed.

        This method allows direct interaction with the LLM by submitting your own prompt.

        :param prompt: The input prompt to send to the LLM
        :type prompt: str
        :param images: Optional list of Image instances for vision queries
        :type images: list[Image] | None
        :param chat_session: Optional stateful chat session to preserve and use history.
        :type chat_session: _ChatSession | None
        :return: The LLM's response. A response cut off at the max output tokens
            is returned incomplete, with a warning logged. A refused response
            that has a text is returned as it is, with a warning logged.
        :rtype: str
        :raises ValueError: If the prompt is empty or not a string
        :raises ValueError: If images parameter is not a list of Image instances
        :raises ValueError: If images are provided but the model doesn't support vision
        :raises ValueError: If the request params of a fallback LLM are invalid, e.g.
            parallel tool calls are set for a fallback LLM that does not support them
        :raises ValueError: If the prompt exceeds the context window of the LLM, and
            no fallback LLM is left to try
        :raises ModelAPIError: If the request to the LLM fails, or the LLM returns
            an empty response, and no fallback LLM is left to try
        :raises ModelRefusalError: If the LLM or its provider declines to answer
            without a text, or the provider blocks the request for its content,
            and no fallback LLM is left to try
        :raises LLMToolLoopLimitError: If the LLM keeps calling tools after
            ``tool_max_rounds`` rounds of tool calls
        """

        # Only LLMs can chat. A fallback model of another kind, such as a decision
        # model, is skipped: it is a fallback only for the extraction.
        fallback_llms = self._fallback_llms
        skipped_fallback_models = [
            fallback_model
            for fallback_model in self.fallback_models
            if not isinstance(fallback_model, _DocumentLLM)
        ]
        if skipped_fallback_models:
            skipped_models_list = ", ".join(
                f"{fallback_model._kind_name} `{fallback_model.model}`"
                for fallback_model in skipped_fallback_models
            )
            logger.info(
                f"Fallback models that cannot chat are skipped for chat: "
                f"{skipped_models_list}"
            )

        # Check the request params of the fallback LLMs before any request is sent.
        # A fallback LLM is used only if the LLM it is a fallback for fails, i.e. an invalid
        # param of a fallback LLM would otherwise go unnoticed until then. The request
        # params of this LLM are checked when its request is built.
        for fallback_llm in fallback_llms:
            fallback_llm._check_request_params(use_tools=True)

        # Try this LLM first, then each fallback LLM in the given order
        llms_to_try = [self, *fallback_llms]
        response: str | None = None
        for i, llm in enumerate(llms_to_try):
            if i > 0:
                logger.info(f"Using fallback LLM {llm.model} for chat")
            response = await llm._chat_async(
                prompt=prompt,
                images=images,
                chat_session=chat_session,
                has_fallback_pending=i < len(llms_to_try) - 1,
            )
            if response is not None:
                break

        if response is None:
            # This should never happen: a failed request or an empty response of an LLM
            # is skipped while a fallback LLM is yet to be tried, but an exception
            # is raised for it when no fallback LLM is left to try.
            raise RuntimeError(
                f"Failed to get response from LLM {self.model} and no fallback is available"
            )

        return response

    async def _chat_async(
        self,
        prompt: str,
        *,
        images: list[_Image] | None = None,
        chat_session: _ChatSession | None = None,
        has_fallback_pending: bool = False,
    ) -> str | None:
        """
        Asynchronously sends a prompt to this LLM only (without trying its fallback LLMs)
        and gets a response.

        :param prompt: The input prompt to send to the LLM
        :type prompt: str
        :param images: Optional list of Image instances for vision queries
        :type images: list[Image] | None
        :param chat_session: Optional stateful chat session to preserve and use history.
        :type chat_session: _ChatSession | None
        :param has_fallback_pending: Whether a fallback LLM will be tried if the call to
            this LLM fails. If True, a model API error or an empty response is logged
            without an exception being raised, and None is returned, as is
            a refused answer without a text. Defaults to False.
        :type has_fallback_pending: bool
        :return: The LLM's response, or None if the LLM call fails, or the LLM returns
            an empty response, while a fallback LLM is yet to be tried
        :rtype: str | None
        :raises ValueError: If the prompt is empty or not a string
        :raises ValueError: If images parameter is not a list of Image instances
        :raises ValueError: If images are provided but the model doesn't support vision
        :raises ValueError: If the prompt exceeds the context window of the LLM, and
            no fallback LLM is yet to be tried
        :raises ModelAPIError: If the request to the LLM fails, or the LLM returns
            an empty response, also when the request is sent again
            ``max_retries_invalid_data`` times, and no fallback LLM is yet to be tried
        :raises ModelRefusalError: If the LLM or its provider declines to answer
            without a text, or the provider blocks the request or its answer for
            their content, also when the request is sent again
            ``max_retries_invalid_data`` times, and no fallback LLM is yet to be tried
        :raises LLMToolLoopLimitError: If the LLM keeps calling tools after
            ``tool_max_rounds`` rounds of tool calls
        """

        # Validate prompt
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Prompt must be a non-empty string")

        # Validate images
        if images and (
            not isinstance(images, list)
            or not all(isinstance(image, _Image) for image in images)
        ):
            raise ValueError("Images must be a list of Image instances")

        # Check for vision support of the LLM and of its fallback LLMs, as a fallback
        # LLM is sent the same images, if the LLM it is a fallback for fails
        if images:
            for llm in [self, *self._fallback_llms]:
                if not llm._supports_vision:
                    source, advice = llm._get_missing_capability_msgs("supports_vision")
                    raise ValueError(
                        f"{'Fallback model' if llm.is_fallback else 'Model'} "
                        f"`{llm.model}` does not support vision {source}. {advice}".rstrip()
                    )

        # If a chat session is provided, reuse chat logic with preserved history
        if chat_session is not None:
            # Ensure system message from this LLM is the first message in the session if absent
            if not any(m.role == "system" for m in chat_session._messages):
                system_msg = self._check_and_build_system_message()
                if system_msg is not None:
                    chat_session._set_messages([system_msg, *chat_session._messages])

            # Build candidate user message for comparison
            candidate_user_message = self._build_user_message(prompt, images)

            # Try to reuse existing pending user message (to avoid duplicates across retries)
            user_message_obj = None
            if chat_session._messages:
                last_msg = chat_session._messages[-1]
                if (
                    last_msg.role == "user"
                    and getattr(last_msg, "_response_succeeded", None) is False
                    and last_msg.content == candidate_user_message.content
                ):
                    user_message_obj = last_msg

            # If no pending matching user message is found, append a new one once
            if user_message_obj is None:
                user_message_obj = candidate_user_message
                user_message_obj._response_succeeded = False  # set as not yet answered
                chat_session._append_message(user_message_obj)

            # Use the history of the chat session as the context for the LLM call,
            # with the tool calls of the previous turns and their results
            messages = [*chat_session._messages]
        else:
            # Build system + user messages explicitly
            messages = []
            system_msg = self._check_and_build_system_message()
            if system_msg is not None:
                messages.append(system_msg)
            messages.append(self._build_user_message(prompt, images))

        # Create model call object to track the interaction
        model_call = _ModelCall(prompt_kwargs={}, prompt=prompt)

        # Warn if using default system message
        if self.system_message == self._get_default_system_message():
            _warn_from_caller_code(
                "You are using the default system message optimized for extraction tasks. "
                "For simple chat interactions, consider setting system_message='' to disable it, "
                "or provide your own custom system message."
            )

        # Send message to LLM
        result = await self._query_llm(
            messages=messages,
            model_call_obj=model_call,
            num_retries_failed_request=self.num_retries_failed_request,
            max_retries_failed_request=self.max_retries_failed_request,
            raise_exception_on_model_api_error=True,  # always True for chat
            chat_session=chat_session,
            has_fallback_pending=has_fallback_pending,
            use_tools=True,  # tools are used only in chat
            # A request whose answer cannot be used, e.g. an empty or a blocked
            # one, is sent again, as the same data is asked for again in extraction
            max_retries_unusable_answer=self.max_retries_invalid_data,
        )

        # Update usage and cost statistics
        await self._update_usage_and_cost(result)

        response, usage = result

        # Whether the response was cut off at the max output tokens
        output_truncated = bool(usage.calls) and usage.calls[-1]._output_truncated
        # The request of an answer that cannot be used is sent again, unless
        # the answer was cut off at the max output tokens, as the same request
        # would be cut off again
        n_retries_unusable_answer = (
            0 if output_truncated else self.max_retries_invalid_data
        )

        # An empty response is handled as a failed one. It is told apart from a failed
        # request, which is already reported at this point.
        if response is not None and not response.strip():
            response = None

        # A refused answer, also of a request or an answer that the provider blocked.
        # Without a text, it is asked for again, and then tried with the next
        # fallback LLM, as an empty response is, and raised if none is left.
        refusal = usage.calls[-1]._refusal if usage.calls else None
        if refusal is not None and not _model_call_result_is_api_failure(result):
            refusal_msg = _get_model_refusal_msg(
                kind_name=self._kind_name, model=self.model, refusal=refusal
            )
            if response is None:
                logger.error(refusal_msg)
                if not has_fallback_pending:
                    raise ModelRefusalError(
                        refusal_msg,
                        retry_count=n_retries_unusable_answer,
                        original_error=usage.calls[-1]._original_error_of_refusal,
                    )
                logger.info("The call will be retried with a fallback LLM.")
            else:
                # An answer with a text is returned as it is, e.g. the model's own
                # words on why it does not answer
                logger.warning(f"{refusal_msg} The response is returned as it is.")
        elif response is None and not _model_call_result_is_api_failure(result):
            error_msg = f"LLM {self.model} returned an empty response."
            if output_truncated:
                # e.g. all the tokens were used up by the reasoning of the model
                error_msg += f" {self._get_output_truncated_msg()}"
            logger.error(error_msg)
            if not has_fallback_pending:
                raise ModelAPIError(error_msg, retry_count=n_retries_unusable_answer)
            logger.info("The call will be retried with a fallback LLM.")
        elif response is not None and output_truncated:
            # A response with a text is returned as it is: it can be cut off
            # on purpose, e.g. by a low `max_tokens` set to get a short answer
            warning_msg = (
                f"{self._get_output_truncated_msg()} "
                f"The response is returned incomplete."
            )
            logger.warning(warning_msg)

        # If response is None, the caller will try with the next fallback LLM
        if response is not None and chat_session is not None:
            user_message_obj._response_succeeded = True  # set as answered
            # The tool calls of the turn, if any, are kept in the history before
            # the answer, i.e. they are sent to the LLM with the next turns
            chat_session._add_answer(response)

        return response

    def _warn_tools_ignored_if_enabled(self) -> None:
        """
        Warns if tools are configured for this LLM, since tools are ignored
        during extraction workflows. Tools are only supported in ``llm.chat(...)``.

        :return: None
        :rtype: None
        """
        if self.tools:
            _warn_from_caller_code(
                f"Tool calling for model `{self.model}` is ignored during extraction workflows. "
                f"Tools are only supported in `llm.chat(...)`."
            )

    def _build_user_message(
        self, message_text: str, images: list[_Image] | None = None
    ) -> _Message:
        """
        Builds a 'user' role message with optional images.

        :param message_text: The text content of the message.
        :type message_text: str
        :param images: Optional list of _Image instances for vision queries.
        :type images: list[_Image] | None
        :return: A `_Message` instance representing the user message with role and content.
        :rtype: _Message
        """

        if images:
            user_message_content: list[dict[str, str | dict[str, str]]] = [
                {"type": "text", "text": message_text}
            ]
            for image in images:
                user_message_content.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{image.mime_type};base64,{image.base64_data}",
                        },
                    }
                )
            content = user_message_content
        else:
            content = message_text
        return _Message(role="user", content=content)

    @staticmethod
    def _find_registered_tool_in_scope(name: str) -> ToolHandler:
        """
        Finds a function decorated with ``@register_tool`` by its name in the scope
        of the code that calls the framework.

        Used only when an LLM is deserialized: the serialized data stores the tool
        definitions, while the functions themselves cannot be stored. The functions
        have to be defined or imported under their own names where the LLM is loaded.

        :param name: The name of the tool, i.e. the name of the function.
        :type name: str
        :return: The registered function.
        :rtype: ToolHandler
        :raises ValueError: If no registered function with this name is found.
        """
        # Look up the frames of the calling code, i.e. the frames outside the framework
        for frame_info in inspect.stack():
            frame = frame_info.frame
            mod = frame.f_globals.get("__name__", "")
            if isinstance(mod, str) and mod.startswith("contextgem"):
                continue
            for candidate in (frame.f_locals.get(name), frame.f_globals.get(name)):
                if (
                    _is_registered_tool(candidate)
                    and getattr(candidate, "__contextgem_tool_name__", None) == name
                ):
                    return cast(ToolHandler, candidate)

        raise ValueError(
            f"No registered function found for tool '{name}'. "
            "To load an LLM with tools, the functions of the tools have to be decorated "
            "with @register_tool and be in scope (imported/defined) under their names "
            "where the LLM is loaded."
        )

    def _sync_tool_registry_from_tools(self) -> None:
        """
        Rebuilds the internal tool registry from the validated `self.tools`.

        For each tool, the registry holds the function, the tool definition generated
        from it (as sent to the LLM), and the JSON schema of the tool parameters
        (used to validate the arguments of a tool call).

        :return: None
        :rtype: None
        :raises TypeError: If the definition of a tool cannot be generated, e.g. for
            an unsupported type hint.
        """
        from contextgem.internal.tools.docstring_parser import _parse_docstring
        from contextgem.internal.tools.schema_generator import _generate_tool_schema

        new_registry: dict[str, ToolRegistration] = {}
        for handler in self.tools or []:
            definition = _generate_tool_schema(
                handler, _parse_docstring(inspect.getdoc(handler))
            )
            new_registry[definition["function"]["name"]] = {
                "handler": handler,
                "definition": definition,
                "schema": definition["function"]["parameters"],
            }
        self._tool_registry = new_registry

    def _check_and_build_system_message(self) -> _Message | None:
        """
        Optionally builds a 'system' role message based on the configured message.
        Returns None if no system message should be added.

        :return: A `_Message` representing the system message with role and content,
            or None if no system message should be added.
        :rtype: _Message | None
        """
        sys_msg = self.system_message
        if sys_msg and sys_msg.strip():
            return _Message(role="system", content=sys_msg)
        return None

    def _update_default_prompt(
        self, prompt_path: str | Path, prompt_type: DefaultPromptType
    ) -> None:
        """
        For advanced users only!

        Update the default Jinja2 prompt template for the LLM.

        This method allows you to replace the built-in prompt templates with custom ones
        for specific extraction types. The framework uses these templates to guide the LLM
        in extracting structured information from documents.

        The custom prompt must be a valid Jinja2 template and include all the necessary
        variables that are present in the default prompt. Otherwise, the extraction may fail.
        Default prompts are located under ``contextgem/internal/prompts/``

        IMPORTANT NOTES:

        The default prompts are complex and specifically designed for
        various steps of LLM extraction with the framework. Such prompts include the
        necessary instructions, template variables, nested structures and loops, etc.

        Only use custom prompts if you MUST have a deeper customization and adaptation of the
        default prompts to your specific use case. Otherwise, the default prompts should be
        sufficient for most use cases.

        Use at your own risk!

        :param prompt_path: Path to the Jinja2 template file (.j2 extension required)
        :type prompt_path: str | Path
        :param prompt_type: Type of prompt to update ("aspect" or "concept")
        :type prompt_type: DefaultPromptType
        :return: None
        :rtype: None
        """
        # Convert to string if Path object
        prompt_path_str = str(prompt_path)

        if not prompt_path_str.endswith(".j2"):
            raise ValueError("Prompt path must end with `.j2`.")

        with open(prompt_path, encoding="utf-8") as file:
            template_text = file.read().strip()
            if not template_text:
                raise ValueError("Prompt template is empty.")

        template = _setup_jinja2_template(template_text)

        if prompt_type == "aspect":
            self._extract_aspect_items_prompt = template
        elif prompt_type == "concept":
            self._extract_concept_items_prompt = template
        else:
            raise ValueError(f"Invalid prompt type: {prompt_type}")
        logger.info(
            f"Default prompt for {prompt_type} extraction updated with a custom template."
        )

    def _eq_deserialized_llm_config(
        self,
        other: _DocumentLLM,
    ) -> bool:
        """
        Custom config equality method to compare this _DocumentLLM with a deserialized instance.

        Compares the __dict__ of both instances and performs specific checks for
        certain attributes that require special handling.

        Note that, by default, the reconstructed deserialized _DocumentLLM will be only partially
        equal (==) to the original one, as the api credentials are redacted, and the async limiter
        and the async lock are not serialized and point to different objects in memory
        post-initialization. The attached prompt templates are not serialized either:
        the deserialized _DocumentLLM has the built-in ones, which the LLMs share, i.e. a custom
        prompt has to be set again. Also, usage and cost are reset by default pre-serialization.

        :param other: Another _DocumentLLM instance to compare with
        :type other: _DocumentLLM
        :return: True if the instances are equal, False otherwise
        :rtype: bool
        """

        # What models of all kinds have is compared first, then what only
        # an LLM has (see `_eq_deserialized_kind_specific_config` below)
        return self._eq_deserialized_model_config(other)

    def _eq_deserialized_kind_specific_config(self, other: _DocumentModel) -> bool:
        """
        Compares what only an LLM has with the same of a deserialized LLM:
        the prompts and the capabilities.

        :param other: Another _DocumentLLM instance to compare with
        :type other: _DocumentModel
        :return: True if the instances are equal in what is compared, False otherwise
        :rtype: bool
        """

        # Safe cast: the instances are checked to be of the same class
        # before this method is called
        other = cast(_DocumentLLM, other)

        # Check _extract_aspect_items_prompt
        if (
            self._extract_aspect_items_prompt.render()
            != other._extract_aspect_items_prompt.render()
        ):
            logger.debug(
                "Extract aspect items prompt of deserialized LLM is different."
            )
            return False

        # Check _extract_concept_items_prompt
        if (
            self._extract_concept_items_prompt.render()
            != other._extract_concept_items_prompt.render()
        ):
            logger.debug(
                "Extract concept items prompt of deserialized LLM is different."
            )
            return False

        return True

    @field_validator("tools", mode="before")
    @classmethod
    def _resolve_tools(cls, tools: Any) -> Any:
        """
        Validates that the tools are functions decorated with ``@register_tool``.

        When an LLM is deserialized, the serialized tool definitions are replaced
        with the registered functions of the same names, found in the calling scope.

        :param tools: The value passed for `tools`.
        :type tools: Any
        :return: A list of the registered functions. A value that is not a sequence
            is returned unchanged, to be rejected by the validation of the field.
        :rtype: Any
        :raises ValueError: If a tool is not a function decorated with
            ``@register_tool``, or if the function of a deserialized tool is not found.
        """
        if not isinstance(tools, Sequence) or isinstance(tools, str | bytes):
            return tools

        resolved: list[Any] = []
        for tool in tools:
            if _is_registered_tool(tool):
                resolved.append(tool)
            elif isinstance(tool, dict) and _is_deserialization_in_progress():
                function_data = tool.get("function")
                name = (
                    function_data.get("name")
                    if isinstance(function_data, dict)
                    else None
                )
                if not isinstance(name, str) or not name.strip():
                    raise ValueError(
                        "A serialized tool definition must have the name of the tool."
                    )
                resolved.append(cls._find_registered_tool_in_scope(name))
            elif isinstance(tool, dict):
                raise ValueError(
                    "Tool definitions are not accepted as dictionaries. Define each tool "
                    "as a function decorated with `@register_tool`, and pass the function."
                )
            elif callable(tool):
                raise ValueError(
                    "A function passed as a tool must be decorated with `@register_tool`."
                )
            else:
                raise ValueError(
                    "Each tool must be a function decorated with `@register_tool`."
                )
        return resolved

    @field_validator("tools")
    @classmethod
    def _validate_tools(
        cls,
        tools: Sequence[ToolHandler] | None,
    ) -> Sequence[ToolHandler] | None:
        """
        Validates that the names of the tools are unique.

        :param tools: The functions passed as tools, or None if no tools are provided.
        :type tools: Sequence[ToolHandler] | None
        :return: The validated tools, or None if no tools are provided.
        :rtype: Sequence[ToolHandler] | None
        :raises ValueError: If two or more tools have the same name.
        """
        if tools is None:
            return None

        names = [getattr(tool, "__contextgem_tool_name__", "") for tool in tools]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(
                f"Tool names must be unique. Duplicate tool names: {', '.join(duplicates)}"
            )
        return tools

    def _validate_kind_specific_fallback(self, fallback_model: _DocumentModel) -> None:
        """
        Validates that a fallback LLM has the same output language, system message
        and tool-calling configuration as the current LLM.

        :param fallback_model: The fallback model to validate.
        :type fallback_model: _DocumentModel
        :return: None
        :rtype: None
        :raises ValueError: If the fallback LLM differs from the current LLM
            in any of these params.
        """
        # These params exist only for LLMs, i.e. are compared with a fallback
        # that is an LLM itself
        if isinstance(fallback_model, _DocumentLLM):
            fallback_llm = fallback_model
            # Check for the consistency of the fallback LLM output language
            if fallback_llm.output_language != self.output_language:
                raise ValueError(
                    f"The fallback LLM must have the same output language `{self.output_language}` as the main one."
                )
            # A fallback LLM is given the same instructions as the LLM it replaces.
            # With another system message, e.g. the default one of a fallback LLM
            # created without it, the fallback LLM would follow other instructions,
            # while in a chat session, which keeps a single system message,
            # it would be sent the system message of the main LLM anyway.
            elif fallback_llm.system_message != self.system_message:
                raise ValueError(
                    "The fallback LLM must have the same `system_message` as the main one."
                )

            # Validate tool-calling configuration parity. The tools have to be
            # the same functions, in any order.
            if (self.tools is None) != (fallback_llm.tools is None) or set(
                self.tools or []
            ) != set(fallback_llm.tools or []):
                raise ValueError(
                    "Fallback LLM must have the same `tools` configuration as the main LLM"
                )
            if self.tools and self._supports_tools != fallback_llm._supports_tools:
                raise ValueError(
                    "Fallback LLM must have the same support for tool calls as the main "
                    "LLM, as set with `supports_tools` or detected with litellm"
                )
            if self.tool_choice != fallback_llm.tool_choice:
                raise ValueError(
                    "Fallback LLM must have the same `tool_choice` configuration as the main LLM"
                )
            if self.parallel_tool_calls != fallback_llm.parallel_tool_calls:
                raise ValueError(
                    "Fallback LLM must have the same `parallel_tool_calls` configuration as the main LLM"
                )

    def _finish_kind_specific_validation(self) -> None:
        """
        Updates the state derived from the params of the LLM, and emits the warnings
        on its configuration, after all the params of the LLM are validated.

        The warnings are attributed to the code that triggered the validation, e.g.
        to the line that creates the LLM or assigns a value to its param (see
        ``_warn_from_caller_code()``).

        :return: None
        :rtype: None
        """

        # The state derived from the params is updated after all the checks pass:
        # if an assigned value is rejected, only the params are restored
        self._sync_tool_registry_from_tools()
        self._update_detected_capabilities()

        # Note: model-specific support for `reasoning_effort` values is not
        # validated here; validation is delegated to litellm / the provider API,
        # which raises an error at request time for unsupported combinations.
        # Only whether the param is sent at all is warned about below.

        # Note: the model role is a free-form routing label that implies no capabilities
        # of the model, i.e. vision and reasoning support is not validated against it.
        # Vision support is validated when extraction of concepts from images starts,
        # as well as at request time, when images are sent to the model.

        # Emit relevant warnings

        if self.tools is not None and not self._supports_tools:
            source, advice = self._get_missing_capability_msgs("supports_tools")
            _warn_from_caller_code(
                f"Model `{self.model}` is assigned tools, but does not support "
                f"tool calls {source}. The tools are sent with the requests of "
                f"a chat, which the model or its provider may reject or answer "
                f"without calling the tools. {advice}".rstrip(),
            )
        if (
            self.parallel_tool_calls is not None
            and not self._supports_parallel_tool_calls
        ):
            source, advice = self._get_missing_capability_msgs(
                "supports_parallel_tool_calls"
            )
            _warn_from_caller_code(
                f"Model `{self.model}` has `parallel_tool_calls` set, but does not "
                f"support parallel tool calls {source}. A chat with the LLM raises "
                f"ValueError before any request is sent. {advice}".rstrip(),
            )

        # `reasoning_effort` is sent only to a reasoning model, and only if litellm
        # takes it for the model (see `_build_request_config()`)
        if self.reasoning_effort is not None:
            if not self._supports_reasoning:
                source, advice = self._get_missing_capability_msgs("supports_reasoning")
                _warn_from_caller_code(
                    f"`reasoning_effort` is set for model `{self.model}`, but is not "
                    f"sent with its requests: the model does not support reasoning "
                    f"{source}. {advice}".rstrip(),
                )
            elif "reasoning_effort" not in (
                litellm.get_supported_openai_params(self.model) or []  # type: ignore[attr-defined]
            ):
                _warn_from_caller_code(
                    f"`reasoning_effort` is set for model `{self.model}`, but is not "
                    f"sent with its requests: litellm does not list it among the params "
                    f"of the model (`litellm.get_supported_openai_params()`).",
                )

        # Warn that the output language applies to the extraction prompts only if
        # the system message is not the default one, which is what has the instruction
        # on the language of the output, e.g. in a chat. With the default output
        # language, nothing that is set is left without effect, e.g. when the system
        # message is disabled for chat.
        if (
            self.output_language != type(self).model_fields["output_language"].default
            and self.system_message != self._get_default_system_message()
        ):
            _warn_from_caller_code(
                "`output_language` applies to the extraction prompts only, as the system message is not the default "
                "one: the system message gives no instruction on the language of the "
                "output, and in a chat, where no extraction prompt is sent, "
                "`output_language` has no effect. To apply it to the system message as "
                "well, use the default system message (`system_message=None`), or give "
                "such an instruction in your own system message.",
            )

    def _validate_input_tokens(self, messages: list[dict[str, str]]) -> None:
        """
        Validates that the input messages do not exceed the model's maximum input tokens.

        :param messages: List of message dictionaries to validate
        :type messages: list[dict[str, str]]
        :raises ValueError: If the messages exceed the model's maximum input tokens
        :return: None
        :rtype: None
        """

        context_exceeded = False
        try:
            # Get model information to check context window
            model_info = litellm.get_model_info(self.model)  # type: ignore[attr-defined]
            max_input_tokens = model_info.get("max_input_tokens")

            # If max_input_tokens is not available, skip validation
            if max_input_tokens is None:
                self._log_skipped_token_check(
                    "input",
                    f"Could not determine max_input_tokens for model `{self.model}`. "
                    f"Skipping validation.",
                )
                return

            if self.model in _MODELS_WITHOUT_TOKEN_COUNT:
                return

            # Count tokens in the messages
            try:
                token_count = litellm.token_counter(model=self.model, messages=messages)  # type: ignore[attr-defined]
            except OSError as e:
                # The token encodings could not be downloaded, e.g. without network
                # access: each count would try the download again, which can take
                # seconds, e.g. behind a proxy that refuses it
                _MODELS_WITHOUT_TOKEN_COUNT.add(self.model)
                logger.warning(
                    f"Could not count tokens for model `{self.model}`: {e}. Skipping "
                    f"input token validation for the requests of this model until "
                    f"the process is restarted. To count the tokens without network "
                    f"access, see {_OFFLINE_ENCODINGS_DOCS_URL}"
                )
                return
            except Exception as e:  # noqa: BLE001 - token counting must not fail the call
                logger.warning(
                    f"Could not count tokens for model `{self.model}`: {e}. Skipping input token validation."
                )
                return

            # Check if we exceed the context window
            if token_count > max_input_tokens:
                context_exceeded = True
                raise ValueError(
                    f"Input messages contain {token_count} tokens, which exceeds the model's "
                    f"maximum input tokens of {max_input_tokens} for model `{self.model}`. "
                    f"For long documents, consider setting `max_paragraphs_to_analyze_per_call` "
                    f"(for text) or `max_images_to_analyze_per_call` (for images) to process the "
                    f"document in smaller chunks. "
                    f"See the optimization guide for long documents: "
                    f"https://contextgem.shcherbak.ai/optimizations/optimization_long_docs/"
                )

            logger.debug(
                f"Input token validation passed: {token_count}/{max_input_tokens} tokens used"
            )

        except ValueError as e:
            if context_exceeded:
                # Re-raise our own ValueError
                raise
            # If it's a different ValueError, e.g. for a model that litellm
            # does not know, log and continue
            self._log_skipped_token_check(
                "input",
                f"Could not validate max input tokens for model `{self.model}`: {e} "
                f"{_REGISTER_MODEL_ADVICE}",
            )
        except Exception as e:  # noqa: BLE001 - model info must not fail the call
            # If we can't get model info, log a warning but don't fail
            self._log_skipped_token_check(
                "input",
                f"Could not validate max input tokens for model `{self.model}`: {e} "
                f"{_REGISTER_MODEL_ADVICE}",
            )

    def _validate_output_tokens(self) -> None:
        """
        Validates that the configured max_tokens do not exceed the model's maximum
        output tokens. Skipped if max_tokens is not set, as no limit is then sent
        with the requests, and the default of the model applies.

        :raises ValueError: If the configured tokens exceed the model's maximum output tokens
        :return: None
        :rtype: None
        """

        if self.max_tokens is None:
            return

        output_exceeded = False
        try:
            # Get model information to check output token limits
            model_info = litellm.get_model_info(self.model)  # type: ignore[attr-defined]
            max_output_tokens = model_info.get("max_output_tokens")

            # If max_output_tokens is not available, fall back to max_tokens
            if max_output_tokens is None:
                max_output_tokens = model_info.get("max_tokens")

            # If we still don't have a limit, skip validation
            if max_output_tokens is None:
                self._log_skipped_token_check(
                    "output",
                    f"Could not determine max_output_tokens for model `{self.model}`. "
                    f"Skipping max output token validation.",
                )
                return

            # Check if configured tokens exceed the model's limit
            if self.max_tokens > max_output_tokens:
                output_exceeded = True
                raise ValueError(
                    f"Configured max_tokens ({self.max_tokens}) exceeds the model's "
                    f"maximum output tokens of {max_output_tokens} for model `{self.model}`. "
                    f"Lower `max_tokens`, or leave it unset to use the default of the model. "
                    f"For long documents, consider setting `max_paragraphs_to_analyze_per_call` "
                    f"(for text) or `max_images_to_analyze_per_call` (for images) to process the "
                    f"document in smaller chunks. "
                    f"See the optimization guide for long documents: "
                    f"https://contextgem.shcherbak.ai/optimizations/optimization_long_docs/"
                )

            logger.debug(
                f"Output token validation passed: {self.max_tokens}/{max_output_tokens} "
                f"max_tokens configured"
            )

        except ValueError as e:
            if output_exceeded:
                # Re-raise our own ValueError
                raise
            # If it's a different ValueError, e.g. for a model that litellm
            # does not know, log and continue
            self._log_skipped_token_check(
                "output",
                f"Could not validate max output tokens for model `{self.model}`: {e} "
                f"{_REGISTER_MODEL_ADVICE}",
            )
        except Exception as e:  # noqa: BLE001 - model info must not fail the call
            # If we can't get model info, log a warning but don't fail
            self._log_skipped_token_check(
                "output",
                f"Could not validate max output tokens for model `{self.model}`: {e} "
                f"{_REGISTER_MODEL_ADVICE}",
            )

    def _log_skipped_token_check(self, limit: str, msg: str) -> None:
        """
        Logs at the WARNING level that a token limit of the requests of the LLM
        could not be checked, once for the LLM and its model: the check is skipped
        for each request, e.g. for a model that litellm does not know, which would
        otherwise log the same message for each request.

        :param limit: The limit that could not be checked: ``"input"`` or
            ``"output"``.
        :type limit: str
        :param msg: The message to log.
        :type msg: str
        :return: None
        :rtype: None
        """
        key = (self.model, limit)
        if key in self._skipped_token_checks_logged:
            return
        self._skipped_token_checks_logged.add(key)
        logger.warning(f"{msg} Not logged again for this LLM and model.")

    def _get_output_truncated_msg(self) -> str:
        """
        Builds the message for a response of the LLM that was cut off at the max
        output tokens, with the ways to get a complete response that apply to
        the configuration of the LLM.

        :return: The message describing the cut-off response.
        :rtype: str
        """
        msg = f"The response of LLM `{self.model}` was cut off at the max output tokens"
        if self.max_tokens is not None:
            msg += (
                f" set by `max_tokens` ({self.max_tokens}). Increase `max_tokens`, "
                f"or leave it unset to use the default of the model."
            )
        else:
            msg += " of the model."
        if self._supports_reasoning:
            msg += (
                " The reasoning tokens of the model count towards this limit, "
                "i.e. a lower `reasoning_effort` leaves more tokens for the response."
            )
        return msg

    def _ensure_vision_support_for_messages(self, messages: list[_Message]) -> None:
        """
        Validates that messages do not contain vision content when the model
        does not support vision.

        :param messages: List of message objects
        :type messages: list[_Message]
        :return: None
        :rtype: None
        :raises ValueError: If image content is found but the model lacks vision support
        """

        contains_image_content = any(
            isinstance(m.content, list)
            and any(
                isinstance(part, dict) and part.get("type") == "image_url"
                for part in m.content
            )
            for m in messages
        )
        if contains_image_content and not self._supports_vision:
            source, advice = self._get_missing_capability_msgs("supports_vision")
            raise ValueError(
                f"Model `{self.model}` does not support vision {source}. "
                f"{advice}".rstrip()
            )

        # Warn to use `ollama/` prefix for image processing when using local models,
        # as the ollama_chat/ does not yet support image inputs
        if contains_image_content and self.model.startswith("ollama_chat/"):
            _warn_from_caller_code(
                "Using `ollama_chat/` prefix for processing images with local models is "
                "not recommended, as it does not yet support image inputs. Please use "
                "`ollama/` prefix instead. "
                "See https://github.com/ollama/ollama/issues/10255 and "
                "https://github.com/ollama/ollama/issues/6451 for more details."
            )

    def _messages_to_request(
        self, messages: list[_Message], thinking_blocks_key: str | None = None
    ) -> list[dict[str, Any]]:
        """
        Converts internal message objects to provider-ready dictionaries.

        :param messages: List of message objects
        :type messages: list[_Message]
        :param thinking_blocks_key: Identifies the request that the messages are sent in
            (see `_get_thinking_blocks_key()`). The thinking blocks of an assistant
            message are included only if they were returned for a request with the same
            key. Defaults to None (no thinking blocks are included).
        :type thinking_blocks_key: str | None
        :return: List of request-compatible messages as dictionaries
        :rtype: list[dict[str, Any]]
        """
        return [m._to_message_dict(thinking_blocks_key) for m in messages]

    def _get_tool_definitions(self) -> list[dict[str, Any]]:
        """
        Returns the tool definitions generated from the functions passed as tools.

        :return: The tool definitions, in the order of the tools.
        :rtype: list[dict[str, Any]]
        """
        return [
            registration["definition"] for registration in self._tool_registry.values()
        ]

    def _get_thinking_blocks_key(
        self, messages: list[_Message], *, use_tools: bool = False
    ) -> str:
        """
        Returns what identifies a request to the LLM for the thinking blocks returned
        with its responses: the model, the API endpoint, the tools and the system message
        of the request.

        A model may return thinking blocks with its responses (e.g. Claude), which are
        to be sent back unchanged with the messages they came with. The provider may
        reject a thinking block in a request whose model, tools or system message differ
        from those of the request that the block was returned for, e.g. when a chat
        session is continued by another LLM, or after the tools of the LLM are changed.
        A thinking block is therefore sent back only in requests with the same key
        as the request it was returned for. The order of the tools does not matter.

        :param messages: The messages of the request.
        :type messages: list[_Message]
        :param use_tools: Whether the tools of the LLM are sent with the request.
            Defaults to False.
        :type use_tools: bool
        :return: The key, a SHA-256 hash.
        :rtype: str
        """
        tools = (
            sorted(
                json.dumps(definition, sort_keys=True, ensure_ascii=False, default=str)
                for definition in self._get_tool_definitions()
            )
            if use_tools and self.tools is not None
            else []
        )
        identity = {
            "model": self.model,
            "api_base": self.api_base,
            "deployment_id": self.deployment_id,
            "tools": tools,
            "system_message": next(
                (m.content for m in messages if m.role == "system"), None
            ),
        }
        return hashlib.sha256(
            json.dumps(
                identity, sort_keys=True, ensure_ascii=False, default=str
            ).encode("utf-8")
        ).hexdigest()

    def _build_request_config(self, *, use_tools: bool = False) -> dict[str, Any]:
        """
        Builds the common request configuration for a non-streaming completion call,
        including model-specific parameters. Messages are supplied separately per call.

        :param use_tools: Whether to include the tools of the LLM and the related params
            (tool choice, parallel tool calls). Tools are used only in chat, i.e. they
            are not sent with the requests of extraction workflows. Defaults to False.
        :type use_tools: bool
        :return: Base request configuration
        :rtype: dict[str, Any]
        :raises ValueError: If parallel tool calls are set for the LLM that does not
            support them
        """

        request_config: dict[str, Any] = {
            "model": self.model,
        }

        # Name of the request param that limits the output tokens. Reasoning
        # (CoT-capable) models take the limit as `max_completion_tokens`, where supported.
        max_tokens_param = "max_tokens"  # nosec B105 - not a password

        if self._supports_reasoning:
            model_params: list[str] | None = litellm.get_supported_openai_params(  # type: ignore[attr-defined]
                self.model
            )
            if model_params is not None:
                if "max_completion_tokens" in model_params:
                    max_tokens_param = "max_completion_tokens"  # nosec B105 - not a password
                if "reasoning_effort" in model_params and self.reasoning_effort:
                    request_config["reasoning_effort"] = self.reasoning_effort

        # The sampling params, the seed and the output token limit are sent only
        # if set, i.e. the defaults of the model apply otherwise. Whether a value
        # is supported for the model is left to litellm and the model provider: a param
        # reported by litellm as unsupported is dropped when the request is sent.
        if self.temperature is not None:
            request_config["temperature"] = self.temperature
        if self.top_p is not None:
            request_config["top_p"] = self.top_p
        if self.seed is not None:
            request_config["seed"] = self.seed
        if self.max_tokens is not None:
            request_config[max_tokens_param] = self.max_tokens

        if self.deployment_id:
            request_config["deployment_id"] = self.deployment_id

        if not use_tools:
            return request_config

        if self.tools is not None:
            # The tool definitions generated from the functions passed as tools
            request_config["tools"] = self._get_tool_definitions()
        if self.tool_choice is not None:
            request_config["tool_choice"] = self.tool_choice
        if self.parallel_tool_calls is not None:
            if not self._supports_parallel_tool_calls:
                source, advice = self._get_missing_capability_msgs(
                    "supports_parallel_tool_calls"
                )
                raise ValueError(
                    f"{'Fallback model' if self.is_fallback else 'Model'} "
                    f"`{self.model}` does not support parallel tool calls {source}. "
                    f"{advice}".rstrip()
                )
            request_config["parallel_tool_calls"] = self.parallel_tool_calls

        return request_config

    async def _send_non_streaming_completion(
        self,
        *,
        request_config: dict[str, Any],
        messages_payload: list[dict[str, Any]],
        num_retries_failed_request: int = 3,
        max_retries_failed_request: int = 0,
        drop_params: bool = False,
        drop_tool_choice: bool = False,
    ) -> Any:
        """
        Sends a single non-streaming chat completion request, retrying it if it fails
        with a transient error.

        Only transient errors (request timeouts, rate limits, server and connection errors)
        are retried, with capped exponential backoff and jitter between the retries.
        Non-transient errors (e.g. an invalid request or a failed authentication) are
        raised right away. The number of retries attempted is recorded on the raised
        exception.

        :param request_config: Base request configuration
        :type request_config: dict[str, Any]
        :param messages_payload: Provider-formatted messages to send
        :type messages_payload: list[dict[str, Any]]
        :param num_retries_failed_request: Optional number of retries when LLM request fails
            with a transient error. Defaults to 3.
        :type num_retries_failed_request: int
        :param max_retries_failed_request: Specific to certain provider APIs (e.g. OpenAI). Optional number of
            retries performed by the provider's SDK when LLM request fails. Applies to the first
            attempt only. Defaults to 0.
        :type max_retries_failed_request: int
        :param drop_params: Whether to drop unsupported parameters when calling the LLM API.
            Used internally for automatic retry when UnsupportedParamsError occurs.
        :type drop_params: bool
        :param drop_tool_choice: Whether to remove tool_choice in this call
        :type drop_tool_choice: bool
        :return: Raw completion object
        :rtype: Any
        """

        payload = dict(request_config)
        payload["messages"] = messages_payload
        if drop_tool_choice and "tool_choice" in payload:
            payload["tool_choice"] = None

        # The request is retried here rather than by litellm (`num_retries`), which
        # retries any error, including non-transient ones, and applies backoff only
        # to rate limit errors, with no jitter.
        retries_attempted = 0
        while True:
            try:
                return await litellm.acompletion(
                    **payload,
                    api_key=self.api_key,
                    api_base=self.api_base,
                    api_version=self.api_version,
                    num_retries=0,
                    # Provider-specific retries apply to the first attempt only,
                    # to prevent accumulation of retries
                    max_retries=(
                        max_retries_failed_request if retries_attempted == 0 else 0
                    ),
                    timeout=self.timeout,
                    stream=False,  # always disabled in contextgem
                    drop_params=drop_params,
                )
            except Exception as e:
                if (
                    retries_attempted >= num_retries_failed_request
                    or not _is_retryable_model_api_error(e)
                ):
                    setattr(e, _MODEL_API_RETRIES_ATTEMPTED_ATTR, retries_attempted)
                    raise
                retries_attempted += 1
                delay = _get_model_api_retry_delay(retries_attempted)
                logger.info(
                    f"LLM API request failed with a transient error "
                    f"({type(e).__name__}). Retrying in {delay:.1f}s "
                    f"({retries_attempted}/{num_retries_failed_request})"
                )
                await asyncio.sleep(delay)

    def _record_usage_call(
        self,
        *,
        usage: _ModelUsage,
        answer_text: str | None,
        call_obj: _ModelCall,
        completion_obj: Any,
    ) -> None:
        """
        Accumulates token usage and records metadata for a call into the given usage object.

        :param usage: The _ModelUsage object to accumulate token usage and metadata.
        :type usage: _ModelUsage
        :param answer_text: The text of the answer from the LLM.
        :type answer_text: str | None
        :param call_obj: The _ModelCall object to record the response timestamp and response.
        :type call_obj: _ModelCall
        :param completion_obj: The raw completion object from the LLM.
        :type completion_obj: Any
        """
        call_obj._input_tokens = completion_obj.usage.prompt_tokens
        call_obj._output_tokens = completion_obj.usage.completion_tokens
        call_obj._cache_read_tokens, call_obj._cache_write_tokens = (
            _get_llm_cache_token_counts(completion_obj.usage)
        )
        usage.input += call_obj._input_tokens
        usage.output += call_obj._output_tokens
        usage.cache_read += call_obj._cache_read_tokens
        usage.cache_write += call_obj._cache_write_tokens
        # litellm reports a response cut off at the max output tokens with
        # the "length" finish reason for all LLM providers
        call_obj._output_truncated = any(
            getattr(choice, "finish_reason", None) == "length"
            for choice in completion_obj.choices
        )
        call_obj._refusal = self._get_refusal_of_completion(completion_obj)
        call_obj._record_response_timestamp()
        call_obj.response = answer_text
        usage.calls.append(call_obj)

    @staticmethod
    def _get_refusal_of_completion(completion_obj: Any) -> ModelRefusal | None:
        """
        Returns whether, and how, the model or its provider declined to answer,
        as the response of litellm reports it:

        - "refusal" if the message has the refusal of the model in OpenAI's
          ``refusal`` field, which litellm keeps in the ``provider_specific_fields``
          of the message, as a non-empty text;
        - "content_filter" if the response has the ``content_filter`` finish
          reason, which litellm gives for a refusal of a model (e.g. the ``refusal``
          stop reason of Anthropic) as well as for the content filter of a provider
          (e.g. the safety filters of Gemini, of Azure OpenAI, and the guardrails
          of Amazon Bedrock).

        :param completion_obj: The raw completion object from the LLM.
        :type completion_obj: Any
        :return: How the answer was declined, or None if it was not.
        :rtype: ModelRefusal | None
        """
        choices = getattr(completion_obj, "choices", None) or []
        for choice in choices:
            message = getattr(choice, "message", None)
            provider_fields = getattr(message, "provider_specific_fields", None)
            refusal_text = (
                provider_fields.get("refusal")
                if isinstance(provider_fields, dict)
                else None
            )
            if isinstance(refusal_text, str) and refusal_text.strip():
                return "refusal"
        if any(
            getattr(choice, "finish_reason", None) == "content_filter"
            for choice in choices
        ):
            return "content_filter"
        return None

    async def _execute_tool_call(self, tc_obj: Any) -> _Message:
        """
        Executes a single tool call requested by the model and returns a tool message.

        :param tc_obj: The tool call object to execute.
        :type tc_obj: Any
        :return: The tool message object.
        :rtype: _Message
        :raises _ToolReturnValueError: If the tool returns a value other than
            a non-empty string. It carries the error to be raised to the calling code:
            ``TypeError`` for a value other than a string, or ``ValueError`` for
            an empty string.
        """

        fn = tc_obj.get("function")
        name = fn.get("name")
        args_raw = fn.get("arguments", "{}")
        call_id = tc_obj.get("id")

        try:
            args_dict = json.loads(args_raw)
        except Exception as e:  # noqa: BLE001 - returned to the model as a tool error
            content_text = json.dumps(
                {
                    "error": "InvalidToolArguments",
                    "message": f"Invalid JSON arguments: {e}",
                }
            )
            # Return a tool message with the error for the model to retry in another round
            return _Message(
                role="tool", content=content_text, tool_call_id=call_id, name=name
            )

        # Lookup tool registration
        reg = self._tool_registry.get(name)
        if not reg:
            # The model requested a tool that it was not given, e.g. made up its name.
            # Return a tool message with the error for the model to retry in another round
            content_text = json.dumps(
                {
                    "error": "UnknownTool",
                    "message": f"Tool '{name}' does not exist. "
                    f"Available tools: {', '.join(self._tool_registry)}",
                }
            )
            return _Message(
                role="tool", content=content_text, tool_call_id=call_id, name=name
            )

        schema = reg.get("schema")
        if not schema:
            # This should never happen as we build/validate the tool registry during the LLM init
            raise RuntimeError("Tool schema is required")
        try:
            _jsonschema_validate(schema, args_dict)
        except _JSONSchemaValidationError as ve:
            # Return a tool message with the error for the model to retry in another round
            result_payload = {"error": "SchemaValidationError", "message": str(ve)}
            content_text = json.dumps(result_payload)
            return _Message(
                role="tool", content=content_text, tool_call_id=call_id, name=name
            )

        handler = reg["handler"]
        try:
            if inspect.iscoroutinefunction(handler):
                res = await handler(**args_dict)
            else:
                res = handler(**args_dict)
        except Exception as ex:  # noqa: BLE001 - a tool's errors go back to the model
            content_text = json.dumps(
                {"error": "ToolExecutionError", "message": str(ex)}
            )
            return _Message(
                role="tool", content=content_text, tool_call_id=call_id, name=name
            )

        # Abort immediately for developer error to prevent extra LLM roundtrips.
        # Such an error is told apart from an error of the LLM API, e.g. tool calls
        # of the LLM that cannot be processed, i.e. is not a reason to try a fallback LLM.
        if not isinstance(res, str):
            raise _ToolReturnValueError(
                TypeError(
                    f"Tool handler {name} must return a string. "
                    "Use `json.dumps` for objects."
                )
            )
        if not res.strip():
            raise _ToolReturnValueError(
                ValueError(f"Tool handler {name} must return a non-empty string.")
            )

        content_text = res
        logger.debug(f"Tool {name} returned: {content_text}")

        # Return a tool message with the result
        return _Message(
            role="tool", content=content_text, tool_call_id=call_id, name=name
        )

    async def _execute_tool_calls(self, tool_calls: Any) -> list[_Message]:
        """
        Executes tool calls either in parallel or sequentially depending on configuration.

        :param tool_calls: The tool calls to execute.
        :type tool_calls: Any
        :return: The tool messages.
        :rtype: list[_Message]
        """
        if not tool_calls:
            return []
        # Safe cast: List comprehension creates tuples of (async_method, kwargs)
        cals_and_kwargs = cast(
            AsyncCalsAndKwargs,
            [(self._execute_tool_call, {"tc_obj": tc}) for tc in tool_calls],
        )
        messages = await _run_async_calls(
            cals_and_kwargs=cals_and_kwargs,
            use_concurrency=bool(self.parallel_tool_calls),
        )
        return cast(list[_Message], messages)

    @staticmethod
    def _get_answer_and_tool_calls(
        chat_completion: Any,
    ) -> tuple[str | None, list[Any] | None]:
        """
        Returns the answer text and the tool calls of a completion.

        A completion can have more than one choice, although a single completion
        is requested. For the requests that litellm sends to the Responses API
        (e.g. requests with function tools to OpenAI reasoning models), a text message
        and the tool calls returned in the same response are converted into separate
        choices: the text first, the tool calls last. Therefore, the tool calls are
        collected from all choices, and the answer text is taken from the last choice
        that has a text.

        :param chat_completion: The raw completion object from the LLM.
        :type chat_completion: Any
        :return: The tuple containing the answer text and the tool calls. If no choice
            has a text, the answer is the content of the first choice. If no choice
            has tool calls, the tool calls are None.
        :rtype: tuple[str | None, list[Any] | None]
        """
        choices = chat_completion.choices
        answer = choices[0].message.content
        tool_calls: list[Any] = []
        for choice in choices:
            content = getattr(choice.message, "content", None)
            if isinstance(content, str) and content.strip():
                answer = content
            tool_calls.extend(getattr(choice.message, "tool_calls", None) or [])
        return answer, tool_calls or None

    @staticmethod
    def _get_thinking_blocks(chat_completion: Any) -> list[dict[str, Any]] | None:
        """
        Returns the thinking blocks of a completion, as litellm returns them, from all
        its choices (see `_get_answer_and_tool_calls()`), e.g. the thinking blocks
        of Claude. They are to be sent back unchanged with the message they came with.

        :param chat_completion: The raw completion object from the LLM.
        :type chat_completion: Any
        :return: The thinking blocks, or None if the completion has none.
        :rtype: list[dict[str, Any]] | None
        """
        thinking_blocks: list[dict[str, Any]] = []
        for choice in chat_completion.choices:
            for block in getattr(choice.message, "thinking_blocks", None) or []:
                if isinstance(block, dict):
                    thinking_blocks.append(dict(block))
        return thinking_blocks or None

    async def _send_messages_with_tools(
        self,
        *,
        request_config: dict[str, Any],
        request_messages: list[dict[str, Any]],
        model_call_obj: _ModelCall,
        usage: _ModelUsage,
        num_retries_failed_request: int = 3,
        max_retries_failed_request: int = 0,
        async_limiter: AsyncLimiter | None = None,
        drop_params: bool = False,
        chat_session: _ChatSession | None = None,
        use_tools: bool = False,
        thinking_blocks_key: str | None = None,
        max_retries_unusable_answer: int = 0,
    ) -> tuple[str | None, bool]:
        """
        Sends the initial request, processes any tool-calling loops up to the
        configured max rounds, and returns the final answer along with a flag
        indicating whether tool calls were still requested after the loop ended.

        The thinking blocks that the LLM returns with a tool call are sent back with it
        in the next requests (see `_get_thinking_blocks_key()`).

        Each request is recorded in the usage object: with the tokens used and
        the response if it succeeds, or as an unfinished call if it fails.
        A request that the provider blocked, or whose answer it blocked, is recorded
        as a finished call without a response, with its refusal.

        In chat, a request whose answer cannot be used is sent again, up to
        ``max_retries_unusable_answer`` times, with the same messages, i.e. with
        the results of the tool calls made before it: the tools are not called
        again. Such an answer is one that the provider blocked, or one with neither
        a text nor tool calls, unless it was cut off at the max output tokens.
        In extraction, the call is made again for an answer that cannot be used
        instead (see ``_GenericModelProcessor._extract_instances``), as such
        an answer can be one with a text.

        :param request_config: The request configuration.
        :type request_config: dict[str, Any]
        :param request_messages: The request messages.
        :type request_messages: list[dict[str, Any]]
        :param model_call_obj: The model call object.
        :type model_call_obj: _ModelCall
        :param usage: The usage object.
        :type usage: _ModelUsage
        :param num_retries_failed_request: Optional number of retries when LLM request fails
            with a transient error. Defaults to 3.
        :type num_retries_failed_request: int
        :param max_retries_failed_request: Specific to certain provider APIs (e.g. OpenAI). Optional number of
            retries performed by the provider's SDK when LLM request fails. Defaults to 0.
        :type max_retries_failed_request: int
        :param async_limiter: The async limiter.
        :type async_limiter: AsyncLimiter | None
        :param drop_params: Whether to drop unsupported parameters when calling the LLM API.
            Used internally for automatic retry when UnsupportedParamsError occurs.
        :type drop_params: bool
        :param chat_session: Optional chat session of the turn being answered. If provided,
            the tool calls of the LLM and their results are recorded on the session, to be
            kept in its history when the turn is answered.
        :type chat_session: _ChatSession | None
        :param use_tools: Whether the tools of the LLM are sent with the requests, i.e.
            whether the tool calls requested by the LLM are executed. Defaults to False.
        :type use_tools: bool
        :param thinking_blocks_key: Identifies the requests that are sent (see
            `_get_thinking_blocks_key()`). The thinking blocks returned with the responses
            are kept only if it is provided. Defaults to None.
        :type thinking_blocks_key: str | None
        :param max_retries_unusable_answer: How many times a request whose answer
            cannot be used is sent again. Defaults to 0, as in extraction.
        :type max_retries_unusable_answer: int
        :return: The tuple containing the answer and whether tool calls were still requested.
        :rtype: tuple[str | None, bool]
        :raises _ToolReturnValueError: If a tool returns a value other than a non-empty
            string. It carries the error to be raised to the calling code.
        :raises litellm.ContentPolicyViolationError: If the provider blocks the last
            request that is sent, or its answer.
        """

        async def send_and_record(
            call_obj: _ModelCall, drop_tool_choice: bool = False
        ) -> tuple[str | None, list[Any] | None, list[dict[str, Any]] | None]:
            """
            Sends a request with the current messages and records the call
            in the usage object.

            :param call_obj: The model call object of the request.
            :type call_obj: _ModelCall
            :param drop_tool_choice: Whether to remove tool_choice in this request.
            :type drop_tool_choice: bool
            :return: The tuple containing the answer text, the tool calls, and
                the thinking blocks of the response.
            :rtype: tuple[str | None, list[Any] | None, list[dict[str, Any]] | None]
            """
            # The coroutine of the request is created only when it is awaited, i.e.
            # when the limiter lets the request through: a coroutine created before
            # that is never awaited if the call is cancelled while it waits for
            # the limiter, for which Python issues a warning.
            send_completion = functools.partial(
                self._send_non_streaming_completion,
                request_config=request_config,
                # A copy: the messages of the next requests are added to the list,
                # which the request is not to see changed, e.g. in a stand-in for
                # the API that keeps the requests
                messages_payload=list(request_messages),
                num_retries_failed_request=num_retries_failed_request,
                max_retries_failed_request=max_retries_failed_request,
                drop_params=drop_params,
                drop_tool_choice=drop_tool_choice,
            )
            try:
                if async_limiter:
                    async with async_limiter:
                        chat_completion = await send_completion()
                else:
                    chat_completion = await send_completion()
                answer, tool_calls = self._get_answer_and_tool_calls(chat_completion)
                thinking_blocks = (
                    self._get_thinking_blocks(chat_completion)
                    if thinking_blocks_key is not None
                    else None
                )
                self._record_usage_call(
                    usage=usage,
                    answer_text=answer,
                    call_obj=call_obj,
                    completion_obj=chat_completion,
                )
            except litellm.ContentPolicyViolationError as e:  # type: ignore[attr-defined]
                # The provider blocked the request or its answer for their content:
                # a refusal without an answer (call finished)
                call_obj._record_blocked(e)
                usage.calls.append(call_obj)
                raise
            except Exception:
                # Record the call details (call unfinished)
                usage.calls.append(call_obj)
                raise
            return answer, tool_calls, thinking_blocks

        async def send_until_answered(
            call_obj: _ModelCall, drop_tool_choice: bool = False
        ) -> tuple[str | None, list[Any] | None, list[dict[str, Any]] | None]:
            """
            Sends a request with the current messages (see ``send_and_record``), and
            sends it again while its answer cannot be used, up to
            ``max_retries_unusable_answer`` times: when the provider blocks
            the request or its answer, or when the answer has neither a text nor
            tool calls, e.g. a refused answer without a text, unless the answer was
            cut off at the max output tokens, as the same request would be cut off
            again. Each request is recorded as a call of its own, and each answer
            that is not used is logged.

            :param call_obj: The model call object of the first request.
            :type call_obj: _ModelCall
            :param drop_tool_choice: Whether to remove tool_choice in the requests.
            :type drop_tool_choice: bool
            :return: The tuple containing the answer text, the tool calls, and
                the thinking blocks of the response of the last request.
            :rtype: tuple[str | None, list[Any] | None, list[dict[str, Any]] | None]
            :raises litellm.ContentPolicyViolationError: If the provider blocks
                the last request, or its answer.
            """
            for retry in range(max_retries_unusable_answer + 1):
                if retry:
                    call_obj = _ModelCall(
                        prompt_kwargs=call_obj.prompt_kwargs, prompt=call_obj.prompt
                    )
                is_last_request = retry == max_retries_unusable_answer
                try:
                    answer, tool_calls, thinking_blocks = await send_and_record(
                        call_obj, drop_tool_choice=drop_tool_choice
                    )
                except litellm.ContentPolicyViolationError:  # type: ignore[attr-defined]
                    if is_last_request:
                        raise
                else:
                    has_tool_calls = bool(tool_calls) and use_tools and bool(self.tools)
                    if (
                        is_last_request
                        or has_tool_calls
                        or (isinstance(answer, str) and answer.strip())
                        or call_obj._output_truncated
                    ):
                        return answer, tool_calls, thinking_blocks
                if call_obj._refusal is not None:
                    logger.error(
                        _get_model_refusal_msg(
                            kind_name=self._kind_name,
                            model=self.model,
                            refusal=call_obj._refusal,
                        )
                    )
                    reason = "after a refusal"
                else:
                    logger.error(f"LLM {self.model} returned an empty response.")
                    reason = "after an empty response"
                logger.info(
                    f"Retrying the request {reason} "
                    f"({retry + 1}/{max_retries_unusable_answer})"
                )
            # This should never happen: the last request returns or raises
            raise RuntimeError("No request was sent.")

        # The tool calls of the turn are recorded from the start of the request,
        # i.e. not those of a previous attempt to answer the turn
        if chat_session is not None:
            chat_session._discard_pending_turn()

        # Initial send
        answer, tool_calls, thinking_blocks = await send_until_answered(model_call_obj)
        if not (use_tools and self.tools):
            tool_calls = None

        # Tools loop
        rounds = 0
        while tool_calls and rounds < self.tool_max_rounds:
            rounds += 1

            tool_calls_serialized: list[dict[str, Any]] = []
            for tc in tool_calls:
                fn = tc.get("function")
                name = fn.get("name")
                arguments = fn.get("arguments", "{}")
                call_id = tc.get("id")
                # A tool call that cannot be answered is an invalid response of the LLM.
                # This is checked for all the tool calls before any tool is executed.
                if not all(isinstance(i, str) and i.strip() for i in (name, call_id)):
                    raise ValueError(
                        "The LLM returned a tool call without the name of the tool "
                        "or without the ID of the call."
                    )
                tool_calls_serialized.append(
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {"name": name, "arguments": arguments},
                    }
                )

            # The assistant message has no text if the LLM returned only tool calls.
            # A placeholder text is not used: an LLM may repeat it in its responses,
            # and with some chat templates a text replaces the tool calls
            # in the conversation rendered for the LLM.
            # The thinking blocks returned with the tool calls are sent back with them
            assistant_msg_obj = _Message(
                role="assistant",
                content=answer
                if (isinstance(answer, str) and answer.strip())
                else None,
                tool_calls=tool_calls_serialized,
                thinking_blocks=thinking_blocks,
                thinking_blocks_key=thinking_blocks_key if thinking_blocks else None,
            )
            request_messages.append(
                assistant_msg_obj._to_message_dict(thinking_blocks_key)
            )

            tool_messages = await self._execute_tool_calls(tool_calls)

            # Append tool results
            request_messages.extend([tm._to_message_dict() for tm in tool_messages])
            if chat_session is not None:
                chat_session._add_pending_tool_messages(
                    [assistant_msg_obj, *tool_messages]
                )

            followup_call = _ModelCall(prompt_kwargs={}, prompt="<tools_round_trip>")
            answer, tool_calls, thinking_blocks = await send_until_answered(
                followup_call, drop_tool_choice=True
            )

        # The thinking blocks returned with the answer are kept with it in the session
        if chat_session is not None and thinking_blocks and thinking_blocks_key:
            chat_session._set_pending_answer_thinking_blocks(
                thinking_blocks, thinking_blocks_key
            )

        return answer, bool(tool_calls)

    async def _query_llm(
        self,
        *,
        messages: list[_Message],
        model_call_obj: _ModelCall,
        num_retries_failed_request: int = 3,
        max_retries_failed_request: int = 0,
        async_limiter: AsyncLimiter | None = None,
        drop_params: bool = False,
        raise_exception_on_model_api_error: bool = True,
        chat_session: _ChatSession | None = None,
        has_fallback_pending: bool = False,
        use_tools: bool = False,
        max_retries_unusable_answer: int = 0,
    ) -> tuple[str | None, _ModelUsage]:
        """
        Generates a response from an LLM based on the provided messages and system
        configuration.

        If an exception is raised after a request was sent, the usage and cost of the LLM
        are updated before that with the requests already made, as no usage statistics
        are returned in such case.

        A request that the provider blocked for its content, or whose answer it
        blocked, is a refusal, which is not raised here: the response is None, and
        the call is recorded as finished, with the refusal. The caller handles it
        as a refused answer without a text.

        :param messages: Full chat messages to send to the LLM. Must be a list
            of `_Message` objects including any applicable system message and the user turn.
        :type messages: list[_Message]
        :param model_call_obj: The _ModelCall object holding data on the initiated LLM call.
        :type model_call_obj: _ModelCall
        :param num_retries_failed_request: Optional number of retries when LLM request fails
            with a transient error. Defaults to 3.
        :type num_retries_failed_request: int
        :param max_retries_failed_request: Specific to certain provider APIs (e.g. OpenAI). Optional number of
            retries performed by the provider's SDK when LLM request fails. Defaults to 0.
        :type max_retries_failed_request: int
        :param async_limiter: An optional aiolimiter.AsyncLimiter instance that controls the frequency of
            async LLM API requests, when concurrency is enabled for certain tasks. If not provided,
            such requests will be sent synchronously.
        :type async_limiter: AsyncLimiter | None
        :param drop_params: Whether to drop unsupported parameters when calling the LLM API.
            Used internally for automatic retry when UnsupportedParamsError occurs.
        :type drop_params: bool
        :param raise_exception_on_model_api_error: Whether to raise an exception if the LLM call fails
            due to an error in the LLM API. If False, a warning is logged instead, and no data
            will be returned. Defaults to True.
        :type raise_exception_on_model_api_error: bool, optional
        :param chat_session: Optional chat session of the turn being answered. If provided,
            the tool calls of the LLM and their results are recorded on the session, to be
            kept in its history when the turn is answered.
        :type chat_session: _ChatSession | None
        :param has_fallback_pending: Whether a fallback model will be tried if this call fails.
            If True, an error in the LLM API is logged without being raised or warned about,
            as it is not final. The same applies to the messages that exceed the context
            window of the model. Defaults to False.
        :type has_fallback_pending: bool
        :param use_tools: Whether to send the tools of the LLM with the request and to execute
            the tool calls requested by the LLM. Tools are used only in chat, i.e. they are
            ignored in extraction workflows. Defaults to False.
        :type use_tools: bool
        :param max_retries_unusable_answer: How many times a request whose answer
            cannot be used is sent again (see ``_send_messages_with_tools``), as in
            chat. Defaults to 0, as in extraction, in which the call is made again.
        :type max_retries_unusable_answer: int
        :return: A tuple containing the LLM response and usage statistics.
            The LLM response is None if the LLM call fails.
        :rtype: tuple[str | None, _ModelUsage]
        :raises ValueError: If the messages exceed the context window of the model,
            and no fallback model is yet to be tried.
        :raises ModelAPIError: If the LLM call fails due to an error in the LLM API,
            no fallback model is yet to be tried, and exceptions are to be raised.
        :raises LLMToolLoopLimitError: If the LLM keeps requesting tool calls after
            the configured max rounds.
        """

        # Validate vision support and convert messages
        self._ensure_vision_support_for_messages(messages)
        # What identifies the request for the thinking blocks of the model: those
        # of earlier responses are sent back only to the same model, with the same
        # tools and system message
        thinking_blocks_key = self._get_thinking_blocks_key(
            messages, use_tools=use_tools
        )
        request_messages = self._messages_to_request(messages, thinking_blocks_key)

        # Create an empty usage dict in case the call fails without the possibility to retrieve usage tokens
        usage = _ModelUsage()

        # Validate max input / output tokens before making the API call
        try:
            self._validate_input_tokens(request_messages)
        except ValueError as e:
            # The messages exceed the context window of the model, i.e. the request
            # would fail with a non-transient error. As any such error, it is not final
            # while a fallback model, e.g. an LLM with a larger context window, is yet to be tried.
            # The call is recorded as unfinished, although no request is sent.
            usage.calls.append(model_call_obj)
            if not has_fallback_pending:
                await self._update_usage_and_cost((None, usage))
                raise
            logger.error(f"The request is not sent to the LLM: {e}")
            # The fallback model that is tried next is not necessarily an LLM
            logger.info("The call will be retried with a fallback model.")
            return None, usage
        self._validate_output_tokens()

        # Prepare request configuration with common parameters
        request_config = self._build_request_config(use_tools=use_tools)

        def record_failed_call() -> None:
            """
            Makes sure that the failed call is the last call recorded (call unfinished).
            A request that fails is recorded when it is sent. A call that fails
            otherwise, e.g. when the tool calls returned by the LLM cannot be
            processed, is recorded here.

            :return: None
            :rtype: None
            """
            if usage.calls and usage.calls[-1].timestamp_received is None:
                return
            if any(call is model_call_obj for call in usage.calls):
                # The call failed in a round of tool calls, after the initial request
                usage.calls.append(
                    _ModelCall(prompt_kwargs={}, prompt="<tools_round_trip>")
                )
            else:
                usage.calls.append(model_call_obj)

        # Make API call and process response (with tool loop if enabled)
        try:
            answer, tool_calls_remaining = await self._send_messages_with_tools(
                request_config=request_config,
                request_messages=request_messages,
                model_call_obj=model_call_obj,
                usage=usage,
                num_retries_failed_request=num_retries_failed_request,
                max_retries_failed_request=max_retries_failed_request,
                async_limiter=async_limiter,
                drop_params=drop_params,
                chat_session=chat_session,
                use_tools=use_tools,
                thinking_blocks_key=thinking_blocks_key,
                max_retries_unusable_answer=max_retries_unusable_answer,
            )

            if tool_calls_remaining:
                # Stop immediately to avoid returning incomplete answers
                raise LLMToolLoopLimitError(
                    (
                        f"Tool execution stopped after reaching tool_max_rounds={self.tool_max_rounds}. "
                        "Model continued requesting tools. "
                        "Consider increasing `tool_max_rounds` or refining tool instructions."
                    ),
                    retry_count=0,
                )

            return answer, usage

        except litellm.UnsupportedParamsError as e:  # type: ignore[attr-defined]
            # Handle unsupported model parameters error
            if (
                not drop_params
            ):  # only retry if we haven't already tried with drop_params
                # litellm raises this error before the request is sent, e.g. for
                # a temperature set for a model that accepts only its default one
                logger.warning(
                    f"The request to LLM `{self.model}` has params that are not "
                    f"supported for the model, and is sent without them: {e}"
                )

                # Recursively call with drop_params=True
                return await self._query_llm(
                    messages=messages,
                    model_call_obj=model_call_obj,
                    num_retries_failed_request=num_retries_failed_request,
                    max_retries_failed_request=max_retries_failed_request,
                    async_limiter=async_limiter,
                    drop_params=True,
                    raise_exception_on_model_api_error=raise_exception_on_model_api_error,
                    chat_session=chat_session,
                    has_fallback_pending=has_fallback_pending,
                    use_tools=use_tools,
                    max_retries_unusable_answer=max_retries_unusable_answer,
                )
            else:
                # If drop_params was already True and we still got UnsupportedParamsError,
                # fall through to regular error handling
                record_failed_call()
                # Logged as any other failed request is: the log would
                # otherwise say only that the request is sent without
                # the unsupported params, and a warning is logged only
                # instead of the exception, i.e. not when a fallback model is
                # yet to be tried and not when the exception is raised
                logger.error(
                    f"Exception occurred while calling LLM API with drop_params=True: {e}"
                )
                if has_fallback_pending:
                    logger.info("The call will be retried with a fallback model.")
                else:
                    n_retries = getattr(e, _MODEL_API_RETRIES_ATTEMPTED_ATTR, 0)
                    if raise_exception_on_model_api_error:
                        await self._update_usage_and_cost((None, usage))
                        raise ModelAPIError(
                            "Exception occurred while calling LLM API",
                            retry_count=n_retries,
                            original_error=e,
                        ) from e
                    else:
                        warning_msg = (
                            f"Exception occurred while calling LLM API with drop_params=True: {e}"
                            + f" ({n_retries} retries)."
                        )
                        logger.warning(warning_msg)
        except litellm.ContentPolicyViolationError as e:  # type: ignore[attr-defined]
            # The provider blocked the request or its answer for their content:
            # a refused answer without a text, which the call is recorded with
            # (call finished). The caller reports it, as a refused answer: the call
            # is made again in extraction, as the verdict of a content filter can
            # differ on another request, and the fallback models are tried next.
            # The error of litellm, the message of the provider
            logger.debug(f"The request or its answer was blocked by the provider: {e}")
        except LLMToolLoopLimitError:
            # Propagate tool loop limit errors without wrapping
            await self._update_usage_and_cost((None, usage))
            raise
        except _ToolReturnValueError as e:
            # Raise the error for the invalid value returned by a tool as it is:
            # it is not an error of the LLM API, i.e. no fallback LLM is to be tried
            await self._update_usage_and_cost((None, usage))
            raise e.error from None
        except asyncio.CancelledError:
            # The call is cancelled, e.g. on a timeout set by the calling code
            await self._update_usage_and_cost((None, usage))
            raise
        except Exception as e:
            # e.g. rate limit error, after the retries of the request are exhausted,
            # or a non-transient error such as failed authentication
            record_failed_call()
            logger.error(f"Exception occurred while calling LLM API: {e}")
            if has_fallback_pending:
                logger.info("The call will be retried with a fallback model.")
            else:
                n_retries = getattr(e, _MODEL_API_RETRIES_ATTEMPTED_ATTR, 0)
                if raise_exception_on_model_api_error:
                    await self._update_usage_and_cost((None, usage))
                    raise ModelAPIError(
                        "Exception occurred while calling LLM API",
                        retry_count=n_retries,
                        original_error=e,
                    ) from e
                else:
                    warning_msg = (
                        f"Exception occurred while calling LLM API: {e}"
                        + f" ({n_retries} retries)."
                    )
                    logger.warning(warning_msg)

        # The failed call is recorded in the usage as unfinished at this point,
        # or as refused if the provider blocked it
        return None, usage

    def _set_prompts(self) -> None:
        """
        Sets up prompt templates for various extraction tasks.

        The templates of the built-in prompts are compiled once and shared by
        the LLMs (see ``_get_template``): they are only rendered. A custom prompt
        replaces the template of the LLM it is set for, and of no other LLM
        (see ``_update_default_prompt``).

        :return: None
        :rtype: None
        """

        # Templates with placeholders
        # Extraction
        self._extract_aspect_items_prompt = _get_template("extract_aspect_items")
        self._extract_concept_items_prompt = _get_template("extract_concept_items")

    def _update_detected_capabilities(self) -> None:
        """
        Detects with litellm the capabilities of the model whose params are not set,
        when the LLM is created, and after a value is assigned to its param: for
        the new model if ``model`` is changed, and for a capability whose param
        is no longer set. A capability is detected once for a model, as a detection
        can take a request, e.g. to a local server of the model.

        :return: None
        :rtype: None
        """
        if self._capabilities_detected_for != self.model:
            self._detected_capabilities = {}
            self._capabilities_detected_for = self.model
        for param in _LLM_CAPABILITY_PARAMS:
            if (
                getattr(self, param) is None
                and param not in self._detected_capabilities
            ):
                self._detected_capabilities[param] = self._detect_capability(param)

    def _detect_capability(self, param: str) -> bool:
        """
        Detects a capability of the model with the function of litellm for it.

        :param param: The param of the LLM that declares the capability, e.g.
            ``supports_vision``.
        :type param: str
        :return: Whether litellm detects the capability for the model.
        :rtype: bool
        """
        detector_name, _ = _LLM_CAPABILITY_PARAMS[param]
        return bool(getattr(litellm, detector_name)(self.model))

    def _has_capability(self, param: str) -> bool:
        """
        Returns whether the model has a capability: as set with its param, or as
        detected with litellm if the param is not set.

        :param param: The param of the LLM that declares the capability, e.g.
            ``supports_vision``.
        :type param: str
        :return: Whether the model has the capability.
        :rtype: bool
        """
        declared = getattr(self, param)
        if declared is not None:
            return declared
        if (
            self._capabilities_detected_for == self.model
            and param in self._detected_capabilities
        ):
            return self._detected_capabilities[param]
        # Not detected yet, e.g. while a new model is validated
        return self._detect_capability(param)

    def _get_missing_capability_msgs(self, param: str) -> tuple[str, str]:
        """
        Returns the parts of a message on a capability that the model is taken not
        to have: where this comes from, to follow "does not support <capability>",
        and how to declare the capability, which is empty if its param is set
        to False.

        :param param: The param of the LLM that declares the capability, e.g.
            ``supports_vision``.
        :type param: str
        :return: Where the absence of the capability comes from, and how to declare
            the capability.
        :rtype: tuple[str, str]
        """
        if getattr(self, param) is False:
            return f"as set with `{param}=False`", ""
        detector_name, capability_name = _LLM_CAPABILITY_PARAMS[param]
        return (
            f"according to `litellm.{detector_name}()`",
            f"If the model supports {capability_name}, set `{param}=True` on the LLM.",
        )

    def _set_system_message(self) -> None:
        """
        Sets the default system message for the LLM, when it is created without
        a system message. It is set without the validation of an assignment, which
        would validate the whole LLM again while it is being validated, and issue
        the warnings on its params twice.

        :return: None
        :rtype: None
        """
        self.__dict__["system_message"] = self._get_default_system_message()

    def _get_default_system_message(
        self, output_language: LanguageRequirement | None = None
    ) -> str:
        """
        Retrieves the default system message for the LLM, rendered for its output
        language or for the given one.

        :param output_language: The output language to render the default system
            message for. Defaults to None (the output language of the LLM).
        :type output_language: LanguageRequirement | None
        :return: The default system message.
        :rtype: str
        """
        return _get_template("default_system_message", template_type="system").render(
            {"output_language": output_language or self.output_language}
        )


def _is_tool_exchange_message(message: _Message) -> bool:
    """
    Returns whether a message of a chat is a part of the tool calls of a turn:
    a message of the LLM that requests tool calls, or the result of a tool call.

    :param message: The message.
    :type message: _Message
    :return: True if the message is a part of the tool calls of a turn.
    :rtype: bool
    """
    return message.role == "tool" or (
        message.role == "assistant" and bool(message.tool_calls)
    )


@_disable_direct_initialization
class _ChatSession(_InstanceBase):
    """
    Internal implementation of the ``ChatSession`` class.
    """

    # The history of the chat, as sent to the LLM: with the tool calls of each
    # answered turn and their results, between the user message and the answer
    _messages: list[_Message] = PrivateAttr(default_factory=list)
    # The tool calls and their results of the turn being answered, which are added
    # to the history only when the turn is answered (see `_add_answer()`)
    _pending_tool_messages: list[_Message] = PrivateAttr(default_factory=list)
    # The thinking blocks returned with the answer of the turn being answered, with
    # the key of the request they were returned for (see `_add_answer()`)
    _pending_answer_thinking_blocks: list[dict[str, Any]] | None = PrivateAttr(
        default=None
    )
    _pending_answer_thinking_blocks_key: str | None = PrivateAttr(default=None)

    @property
    def messages(self) -> list[_Message]:
        """
        Returns the list of messages in the session: the system message, if any,
        and the messages of the user and the answers of the LLM. The tool calls
        of the LLM and their results are not listed, although they are kept in
        the history of the chat that is sent to the LLM.

        :return: The list of messages in the session.
        :rtype: list[_Message]
        """
        return [m for m in self._messages if not _is_tool_exchange_message(m)]

    def _validate_system_message_constraints(self) -> None:
        """
        Validate that if a 'system' message exists, it is the first and unique.

        :return: None
        :rtype: None
        :raises ValueError: If multiple 'system' messages exist or if the 'system'
            message is not the first in the conversation.
        """
        if not self._messages:
            return
        system_indices = [
            idx for idx, m in enumerate(self._messages) if m.role == "system"
        ]
        if len(system_indices) > 1:
            raise ValueError(
                f"Multiple 'system' messages are not allowed; found {len(system_indices)}."
            )
        if len(system_indices) == 1 and system_indices[0] != 0:
            raise ValueError(
                "The 'system' message must be the first message in the conversation."
            )

    def _set_messages(self, messages: list[_Message]) -> None:
        """
        Replace the current message history and validate uniqueness.

        :param messages: New list of messages to set.
        :type messages: list[_Message]
        :return: None
        :rtype: None
        :raises ValueError: If duplicate messages by unique_id are present.
        """
        self._messages = list(messages)
        self._validate_system_message_constraints()
        self._validate_list_uniqueness(cast(list, self._messages))

    def _append_message(self, message: _Message) -> None:
        """
        Append a message to the history and validate uniqueness.

        :param message: The message to append.
        :type message: _Message
        :return: None
        :rtype: None
        :raises ValueError: If this introduces a duplicate by unique_id.
        """
        self._messages.append(message)
        self._validate_system_message_constraints()
        self._validate_list_uniqueness(cast(list, self._messages))

    def _add_pending_tool_messages(self, messages: list[_Message]) -> None:
        """
        Adds messages of the tool calls of the turn being answered: an assistant
        message with tool calls, or the results of the tool calls. They are added
        to the history when the turn is answered (see `_add_answer()`).

        :param messages: The messages to add.
        :type messages: list[_Message]
        :return: None
        :rtype: None
        """
        self._pending_tool_messages.extend(messages)

    def _set_pending_answer_thinking_blocks(
        self, thinking_blocks: list[dict[str, Any]], thinking_blocks_key: str
    ) -> None:
        """
        Sets the thinking blocks returned with the answer of the turn being answered,
        which are kept with the answer when it is added (see `_add_answer()`).

        :param thinking_blocks: The thinking blocks.
        :type thinking_blocks: list[dict[str, Any]]
        :param thinking_blocks_key: Identifies the request that the thinking blocks
            were returned for.
        :type thinking_blocks_key: str
        :return: None
        :rtype: None
        """
        self._pending_answer_thinking_blocks = thinking_blocks
        self._pending_answer_thinking_blocks_key = thinking_blocks_key

    def _discard_pending_turn(self) -> None:
        """
        Discards the messages of the tool calls of the turn being answered, and
        the thinking blocks of its answer, e.g. when the turn is answered again from
        its start, by another LLM or after a failed attempt. The tools are then called
        again in the new attempt.

        :return: None
        :rtype: None
        """
        self._pending_tool_messages = []
        self._pending_answer_thinking_blocks = None
        self._pending_answer_thinking_blocks_key = None

    def _add_answer(self, answer: str) -> None:
        """
        Adds the answer of the LLM to the turn being answered, after the tool calls
        that the LLM made and their results, if any, which are thus kept
        in the history of the chat. The thinking blocks returned with the tool calls
        and the answer are kept with them.

        :param answer: The answer of the LLM.
        :type answer: str
        :return: None
        :rtype: None
        """
        answer_message = _Message(
            role="assistant",
            content=answer,
            thinking_blocks=self._pending_answer_thinking_blocks,
            thinking_blocks_key=self._pending_answer_thinking_blocks_key,
        )
        self._extend_messages([*self._pending_tool_messages, answer_message])
        self._discard_pending_turn()

    def _extend_messages(self, messages: list[_Message]) -> None:
        """
        Extend the message history with multiple messages and validate uniqueness.

        :param messages: Messages to add to the history.
        :type messages: list[_Message]
        :return: None
        :rtype: None
        :raises ValueError: If duplicates by unique_id are introduced.
        """
        if messages:
            self._messages.extend(messages)
            self._validate_system_message_constraints()
            self._validate_list_uniqueness(cast(list, self._messages))

    def reset(self) -> None:
        """
        Clears conversation history by removing all messages.

        :return: None
        :rtype: None
        """
        self._set_messages([])
        self._discard_pending_turn()
