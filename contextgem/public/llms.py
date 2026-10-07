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

from contextgem.internal.base.llms import (
    _ChatSession,
    _DocumentLLM,
    _DocumentModelGroup,
)
from contextgem.internal.decorators import _expose_in_registry


@_expose_in_registry(additional_key=_DocumentModelGroup)  # ty: ignore[too-many-positional-arguments]
class DocumentModelGroup(_DocumentModelGroup):
    """
    Represents a group of DocumentLLMs with unique roles for processing document content.

    This class manages multiple LLMs, each of which extracts the aspects and concepts routed
    to it by its role, e.g. to use a smaller model for simpler tasks and a more capable
    model for more complex ones.

    :ivar models: A list of DocumentLLM instances, each with a unique ``role``. An aspect/concept
        is extracted by the LLM with the ``role`` equal to the ``model_role`` of the aspect/concept,
        and an aspect/concept without ``model_role`` is extracted by the LLM without ``role``.
        Only one LLM in the group can have no role. At least 2 instances are required.
    :vartype models: list[DocumentLLM]
    :ivar output_language: Language for produced output text (justifications, explanations).
        Values: "en" (always English) or "adapt" (matches document/image language).
        All LLMs in the group must share the same output_language setting.
        Defaults to "en". Applies only when DocumentLLMs' default system messages are used.
    :vartype output_language: LanguageRequirement

    Note:
        Refer to the :class:`DocumentLLM` class for more information on constructing LLMs
        for the group.

    Example:
        .. literalinclude:: ../../../dev/usage_examples/docstrings/llms/def_model_group.py
            :language: python
            :caption: Model group definition
    """

    pass


@_expose_in_registry(additional_key=_DocumentLLM)  # ty: ignore[too-many-positional-arguments]
class DocumentLLM(_DocumentLLM):
    """
    Handles processing documents with a specific LLM.

    This class serves as an abstraction for interacting with a LLM. It provides functionality
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
    :vartype system_message: str | None
    :ivar temperature: Sampling temperature (0.0 to 1.0) controlling response creativity.
        Lower values produce more predictable outputs, higher values generate more varied responses.
        Defaults to 0.3.
    :vartype temperature: float | None
    :ivar max_tokens: Maximum tokens allowed in the generated response. Defaults to 4096.
    :vartype max_tokens: int
    :ivar max_completion_tokens: Maximum token size for output completions in reasoning
        (CoT-capable) models. Defaults to 16000.
    :vartype max_completion_tokens: int
    :ivar reasoning_effort: The effort level for the LLM to reason about the input. Can be set to
        ``"none"``, ``"minimal"``, ``"low"``, ``"medium"``, ``"high"``, ``"xhigh"``, or
        ``"max"``. Support for specific values varies by model and provider; unsupported
        combinations are rejected by litellm or the provider API at request time. Note that
        ``"none"`` explicitly disables reasoning on models that support this value, which
        differs from the default ``None`` (parameter not sent, provider default applies).
        Relevant for reasoning (CoT-capable) models. Defaults to None.
    :vartype reasoning_effort: ReasoningEffort | None
    :ivar top_p: Nucleus sampling value (0.0 to 1.0) controlling output focus/randomness.
        Lower values make output more deterministic, higher values produce more diverse outputs.
        Defaults to ``None`` (only ``temperature`` is sent). Some providers (e.g. Claude 4.x)
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
        invalid data. Each retry is a new request. Defaults to 3.
    :vartype max_retries_invalid_data: int
    :ivar timeout: Timeout in seconds for LLM API calls. Defaults to 120 seconds.
    :vartype timeout: int
    :ivar pricing_details: ModelPricing object with pricing details for cost calculation.
        Defaults to None.
    :vartype pricing_details: ModelPricing | None
    :ivar auto_pricing: Enable automatic LLM cost calculation using genai-prices.
        Ignored when ``pricing_details`` is provided. Defaults to ``False``.
    :vartype auto_pricing: bool
    :ivar auto_pricing_refresh: Whether genai-prices should auto-refresh its cached
        pricing data, e.g. to price a recently released model that the bundled
        pricing data does not include yet. Defaults to ``False``.
    :vartype auto_pricing_refresh: bool
    :ivar is_fallback: Indicates whether the LLM is a fallback model. Defaults to False.
    :vartype is_fallback: bool
    :ivar fallback_models: DocumentLLMs to use as fallbacks if the current one fails, tried
        in the given order until one of them succeeds. Each must have ``is_fallback=True``
        and the same role as the current LLM, must differ from the current LLM and from
        each other in at least one config param, and must support vision if the current
        LLM is used to extract concepts from images. Defaults to an empty list.
    :vartype fallback_models: list[DocumentLLM]
    :ivar output_language: Language for produced output text (justifications, explanations).
        Can be "en" (English) or "adapt" (adapts to document/image language). Defaults to "en".
        Applies only when DocumentLLM's default system message is used.
    :vartype output_language: LanguageRequirement
    :ivar async_limiter: Controls frequency of async LLM API requests for concurrent tasks.
        Defaults to allowing 3 acquisitions per 10-second period to prevent rate limit issues.
        Requests to a fallback LLM are controlled by the limiter of that fallback LLM.
        See https://github.com/mjpieters/aiolimiter for configuration details.
    :vartype async_limiter: AsyncLimiter
    :ivar seed: Seed for random number generation to help produce more consistent outputs
        across multiple runs. When set to a specific integer value, the LLM will attempt
        to use this seed for sampling operations. However, deterministic output is still
        not guaranteed even with the same seed, as other factors may influence the model's
        response. Defaults to None.
    :vartype seed: int | None

    Note:

        - Model groups
            Refer to the :class:`DocumentModelGroup` class for more information on constructing model groups,
            which are a collection of LLMs with unique roles, used for complex document processing tasks.

        - Model role
            The ``role`` of an LLM is a free-form label used to route aspects/concepts to the LLMs
            of a :class:`DocumentModelGroup`. For example, if an aspect/concept is assigned
            ``model_role="complex"``, it means that the aspect/concept is extracted from the document
            using the LLM in the group with the ``role`` set to "complex", while the aspects/concepts
            without ``model_role`` are extracted using the LLM without ``role``. This helps to channel
            different tasks to different LLMs, ensuring that the task is handled by the most
            appropriate model. Usually, domain expertise is required to determine the most appropriate
            LLM for a specific aspect/concept. A role is just a label and implies no capabilities
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

    pass


@_expose_in_registry(additional_key=_ChatSession)  # ty: ignore[too-many-positional-arguments]
class ChatSession(_ChatSession):
    """
    Stateful chat session that preserves message history across turns.

    To be used as ``chat_session=...`` parameter for ``DocumentLLM.chat(...)``
    or ``DocumentLLM.chat_async(...)``.
    """

    pass
