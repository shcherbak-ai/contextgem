# SPDX-FileCopyrightText: Shcherbak AI AS
# SPDX-License-Identifier: Apache-2.0

"""
Module for handling processing logic using LLMs.

This module provides classes and utilities for interacting with LLMs in document processing
workflows. It includes functionality for managing LLM configurations, handling API calls,
processing text and image inputs, tracking token usage and costs, and managing rate limits
for LLM requests.

The module supports various LLM providers through the litellm library, enabling both
text-only and multimodal (vision) capabilities. It implements efficient asynchronous
processing patterns and provides detailed usage statistics for monitoring and cost management.
"""

from __future__ import annotations

from contextgem.internal.base.llms import _ChatSession, _DocumentLLM
from contextgem.internal.decorators import _expose_in_registry


@_expose_in_registry(additional_key=_DocumentLLM)
class DocumentLLM(_DocumentLLM):
    """
    Handles processing documents with a specific LLM.

    This class serves as an abstraction for interacting with an LLM. It provides functionality
    for querying the LLM with text or image inputs, and manages prompt preparation and token
    usage tracking. An LLM can be used on its own or, with a role assigned, as a part of
    a model group.

    :ivar model: Model identifier in format {model_provider}/{model_name}.
        See https://docs.litellm.ai/docs/providers for supported providers.
    :vartype model: str
    :ivar deployment_id: Deployment ID for the LLM. Primarily used with Azure OpenAI.
    :vartype deployment_id: str | None
    :ivar api_key: API key for LLM authentication. Not required for local models (e.g., Ollama).
        Not included in the serialized LLM data, i.e. to be set again after deserialization.
        Not shown when the LLM is printed, nor in the text of a validation error of the LLM,
        which names the invalid parameter and the reason without showing the values passed.
        The values are still returned by the ``errors()`` and ``json()`` methods of the error,
        unless ``include_input=False`` is passed, with ``'**********'`` in place of a key
        that is passed as a value, e.g. a rejected key or the key among the parameters that
        an LLM is created with, and in place of the value of any parameter that an LLM does
        not have, as it may be the key passed under a misspelled name. An LLM that is among
        the values is returned by ``errors()`` as it is, i.e. with the key as its attribute,
        and is written out by ``json()`` without its key. A key that is passed as the value
        of another parameter, e.g. of ``model``, is returned as any other value.
    :vartype api_key: str | None
    :ivar api_base: Base URL of the API endpoint. Not included in the serialized LLM data,
        i.e. to be set again after deserialization.
    :vartype api_base: str | None
    :ivar api_version: API version. Primarily used with Azure OpenAI.
    :vartype api_version: str | None
    :ivar role: Free-form role label of the LLM (up to 64 printable characters), used for routing in a ``DocumentModelGroup``:
        the LLM extracts the aspects/concepts with the same ``model_role``. Defaults to None,
        in which case the LLM extracts the aspects/concepts without ``model_role``. Ignored
        when the LLM is used on its own, as it then extracts all aspects/concepts.
    :vartype role: str | None
    :ivar system_message: Preparatory system-level message to set context for LLM responses.
        Defaults to None, in which case the default system message for extraction is set,
        rendered for the output language, also when the output language is changed later.
        Assigning None restores the default system message, while an empty string disables
        the system message, i.e. none is sent.
    :vartype system_message: str | None
    :ivar temperature: Sampling temperature (0.0 or higher; the upper limit depends on
        the model) controlling response creativity.
        Lower values produce more predictable outputs, higher values generate more varied responses.
        Defaults to None, in which case the parameter is not sent, and the default of the model
        applies. A value is sent to any model; if ``litellm`` reports it as not supported
        for the model, it is dropped from the request, with a warning logged.
    :vartype temperature: float | None
    :ivar max_tokens: Maximum output tokens allowed per request. For reasoning (CoT-capable)
        models, the reasoning tokens count towards this limit. Defaults to None, in which case
        no limit is sent with the requests, and the default of the model applies (for most
        providers, the maximum output of the model). A response cut off at the limit
        fails the extraction, and is returned with a warning logged in a chat.
    :vartype max_tokens: int | None
    :ivar reasoning_effort: The effort level for the LLM to reason about the input. Can be set to
        ``"none"``, ``"minimal"``, ``"low"``, ``"medium"``, ``"high"``, ``"xhigh"``, or
        ``"max"``. The parameter is sent only to the models detected as supporting reasoning.
        Support for specific values varies by model and provider: if ``litellm`` reports
        a value as not supported for the model, it is dropped from the request, with a warning
        logged, while a value that is rejected otherwise, e.g. by the provider API, fails
        the request. Note that
        ``"none"`` explicitly disables reasoning on models that support this value, which
        differs from the default ``None`` (parameter not sent, provider default applies).
        Relevant for reasoning (CoT-capable) models. Defaults to None.
    :vartype reasoning_effort: ReasoningEffort | None
    :ivar top_p: Nucleus sampling value (0.0 to 1.0) controlling output focus/randomness.
        Lower values make output more deterministic, higher values produce more diverse outputs.
        Defaults to ``None`` (the parameter is not sent). Some providers (e.g. Claude 4.x)
        reject requests that set both ``temperature`` and ``top_p``.
    :vartype top_p: float | None
    :ivar num_retries_failed_request: Number of retries when LLM request fails with a transient
        error (request timeout, rate limit, server or connection error). Retries are performed
        with exponential backoff and jitter. Non-transient errors (e.g. invalid request,
        failed authentication, exceeded context window) are not retried. Defaults to 3.
    :vartype num_retries_failed_request: int
    :ivar max_retries_failed_request: LLM provider-specific retry count for failed requests,
        i.e. retries performed by the provider's SDK, where supported. Applies to the first
        attempt of a request only. Defaults to 0.
    :vartype max_retries_failed_request: int
    :ivar max_retries_invalid_data: Number of retries when LLM request succeeds but LLM returns
        invalid data. Each retry is a new request, i.e. the LLM is asked for the same
        data up to ``1 + max_retries_invalid_data`` times. A fallback LLM applies its own
        setting in the same way: it is asked once, and then again up to its
        ``max_retries_invalid_data`` times. If the content of a call is sent
        in parts, a retry is a request for the part that the invalid data was returned
        for, i.e. the parts answered with valid data are not sent again. Defaults to 3.
    :vartype max_retries_invalid_data: int
    :ivar timeout: Timeout in seconds for LLM API calls. As responses are not streamed,
        the whole response has to be received within the timeout. Defaults to 600 seconds.
    :vartype timeout: int
    :ivar pricing_details: ModelPricing object with pricing details for cost calculation.
        Defaults to None.
    :vartype pricing_details: ModelPricing | None
    :ivar auto_pricing: Enable automatic LLM cost calculation using genai-prices.
        Cannot be set together with ``pricing_details``, and is not supported for
        local models (e.g. ``ollama/``, ``ollama_chat/``, ``lm_studio/``): both
        raise ``ValueError``. Defaults to ``False``.
    :vartype auto_pricing: bool
    :ivar auto_pricing_refresh: Whether genai-prices should auto-refresh its cached
        pricing data, e.g. to price a recently released model that the bundled
        pricing data does not include yet. Defaults to ``False``.
    :vartype auto_pricing_refresh: bool
    :ivar is_fallback: Indicates whether the LLM is a fallback model. Defaults to False.
    :vartype is_fallback: bool
    :ivar fallback_models: DocumentLLMs to use as fallbacks if the current one fails, tried
        in the given order until one of them succeeds. Each must have ``is_fallback=True``
        and the same role, output language, system message, tools, ``tool_choice`` and
        ``parallel_tool_calls`` as the current LLM, must differ from the current LLM and
        from each other in at least one config param, and must support vision if
        the current LLM is used to extract concepts from images. A decision model
        (``DocumentDecisionModel``) with ``is_fallback=True`` and the same role can
        also be a fallback: it replaces the LLM only in the extraction, provided that
        a decision model can extract everything the LLM is to extract, and is skipped
        in a chat. Each fallback applies its own retry settings: a fallback LLM that
        returns invalid data during extraction is asked again up to its
        ``max_retries_invalid_data`` times, as the current LLM is, and a fallback
        decision model is asked once. Defaults to an empty list.
    :vartype fallback_models: list[DocumentLLM | DocumentDecisionModel]
    :ivar output_language: Language for produced output text (justifications, explanations).
        Can be "en" (English) or "adapt" (adapts to document/image language). Defaults to "en".
        Applies to the extraction prompts, and to the default system message: with a custom
        system message, it applies to the extraction prompts only.
    :vartype output_language: LanguageRequirement
    :ivar async_limiter: Controls frequency of async LLM API requests for concurrent tasks.
        Defaults to allowing 10 acquisitions per 10-second period to prevent rate limit issues.
        The rate applies to all the requests to the LLM that are sent concurrently, i.e.
        the extraction calls that run at the same time share it.
        Requests to a fallback model are controlled by the limiter of that fallback model.
        See https://github.com/mjpieters/aiolimiter for configuration details.
    :vartype async_limiter: AsyncLimiter
    :ivar seed: Seed for random number generation to help produce more consistent outputs
        across multiple runs. When set to a specific integer value, the LLM will attempt
        to use this seed for sampling operations. However, deterministic output is still
        not guaranteed even with the same seed, as other factors may influence the model's
        response. Support depends on the model provider: a seed is mostly of use with local
        and open-weight models, while some providers (e.g. Anthropic) do not support it.
        If ``litellm`` reports the seed as not supported for the model, it is dropped from
        the request, with a warning logged. Defaults to None (the parameter is not sent).
    :vartype seed: int | None

    Note:

        - Model groups
            Refer to the :class:`~contextgem.public.models.DocumentModelGroup` class for more information on constructing model groups,
            which are a collection of models with unique roles, used for complex document processing tasks.

        - Model role
            The ``role`` of an LLM is a free-form label used to route aspects/concepts to the models
            of a :class:`~contextgem.public.models.DocumentModelGroup`. For example, if an aspect/concept is assigned
            ``model_role="complex"``, it means that the aspect/concept is extracted from the document
            using the model in the group with the ``role`` set to "complex", while the aspects/concepts
            without ``model_role`` are extracted using the model without ``role``. This helps to channel
            different tasks to different models, ensuring that the task is handled by the most
            appropriate model. Usually, domain expertise is required to determine the most appropriate
            model for a specific aspect/concept. A role is just a label and implies no capabilities
            of the model. When an LLM is used on its own, i.e. not in a group, you can skip the role
            assignment completely: the LLM extracts all aspects/concepts, regardless of their ``model_role``.

        - Explicit capability declaration
            Model vision capabilities are automatically detected using
            ``litellm.supports_vision()``. If this function does not correctly identify your model's capabilities,
            an error will be raised when the model is to be sent images, and you can explicitly declare
            the capability by setting ``_supports_vision=True`` on the LLM instance.

    Example:
        .. literalinclude:: ../../../dev/usage_examples/docstrings/llms/def_llm.py
            :language: python
            :caption: LLM definition
    """


@_expose_in_registry(additional_key=_ChatSession)
class ChatSession(_ChatSession):
    """
    Stateful chat session that preserves message history across turns.

    To be used as ``chat_session=...`` parameter for ``DocumentLLM.chat(...)``
    or ``DocumentLLM.chat_async(...)``.

    Note:
        A chat session keeps the first system message it is given: the system message
        of an LLM is added to the session only if the session has none yet. The system
        message of the session is sent with each request of the session, also by any
        other LLM that the session is used with, instead of the system message
        of that LLM.

        In a chat with tools, the history of the session keeps the tool calls
        of each answered turn and their results, which are sent to the LLM with
        the next turns. The results include the errors of the tools, which the LLM
        receives as it calls them. A turn that ends without an answer, e.g. on
        an error of the LLM API, is started over when it is sent again, i.e. its
        tool calls are not kept. ``messages`` lists the messages of the user and
        the answers of the LLM, without the tool calls.

        The history also keeps the thinking blocks that an LLM returns with its tool
        calls and answers, e.g. those of Claude when ``reasoning_effort`` is set.
        They are sent back unchanged, so that the LLM keeps its reasoning across
        tool rounds and turns, but only to the same model with the same tools and
        system message: a provider may reject them otherwise.
    """
