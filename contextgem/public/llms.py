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

from contextgem.internal.base.llms import _ChatSession, _DocumentLLM, _DocumentLLMGroup
from contextgem.internal.decorators import _expose_in_registry


@_expose_in_registry(additional_key=_DocumentLLMGroup)
class DocumentLLMGroup(_DocumentLLMGroup):
    """
    Represents a group of DocumentLLMs with unique roles for processing document content.

    This class manages multiple LLMs assigned to specific roles for text and vision processing.
    It ensures role compliance and facilitates extraction of aspects and concepts from documents.

    :ivar llms: A list of DocumentLLM instances, each with a unique role (e.g., `extractor_text`,
        `reasoner_text`, `extractor_vision`, `reasoner_vision`). At least 2 instances
        with distinct roles are required.
    :vartype llms: list[DocumentLLM]
    :ivar output_language: Language for produced output text (justifications, explanations).
        Values: "en" (always English) or "adapt" (matches document/image language).
        All LLMs in the group must share the same output_language setting.
        Defaults to "en". Applies only when DocumentLLMs' default system messages are used.
    :vartype output_language: LanguageRequirement

    Note:
        Refer to the :class:`DocumentLLM` class for more information on constructing LLMs
        for the group.

    Example:
        .. literalinclude:: ../../../dev/usage_examples/docstrings/llms/def_llm_group.py
            :language: python
            :caption: LLM group definition
    """

    pass


@_expose_in_registry(additional_key=_DocumentLLM)
class DocumentLLM(_DocumentLLM):
    """
    Handles processing documents with a specific LLM.

    This class serves as an abstraction for interacting with a LLM. It provides functionality
    for querying the LLM with text or image inputs, and manages prompt preparation and token
    usage tracking. The class can be configured with different roles based on the document
    processing task.

    :ivar model: Model identifier in format {model_provider}/{model_name}.
        See https://docs.litellm.ai/docs/providers for supported providers.
    :vartype model: str
    :ivar deployment_id: Deployment ID for the LLM. Primarily used with Azure OpenAI.
    :vartype deployment_id: str | None
    :ivar api_key: API key for LLM authentication. Not required for local models (e.g., Ollama).
    :vartype api_key: str | None
    :ivar api_base: Base URL of the API endpoint.
    :vartype api_base: str | None
    :ivar api_version: API version. Primarily used with Azure OpenAI.
    :vartype api_version: str | None
    :ivar role: Role type for the LLM ("extractor_text", "reasoner_text",
        "extractor_vision", "reasoner_vision", "extractor_multimodal",
        "reasoner_multimodal"). Defaults to "extractor_text".
    :vartype role: LLMRoleAny
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
        ``"minimal"`` (gpt-5 models only), ``"low"``, ``"medium"``, ``"high"``, or ``"xhigh"``
        (gpt-5.2 models only). Relevant for reasoning (CoT-capable) models. Defaults to None.
    :vartype reasoning_effort: ReasoningEffort | None
    :ivar top_p: Nucleus sampling value (0.0 to 1.0) controlling output focus/randomness.
        Lower values make output more deterministic, higher values produce more diverse outputs.
        Defaults to 0.3.
    :vartype top_p: float | None
    :ivar num_retries_failed_request: Number of retries when LLM request fails. Defaults to 3.
    :vartype num_retries_failed_request: int
    :ivar max_retries_failed_request: LLM provider-specific retry count for failed requests.
        Defaults to 0.
    :vartype max_retries_failed_request: int
    :ivar max_retries_invalid_data: Number of retries when LLM returns invalid data. Defaults to 3.
    :vartype max_retries_invalid_data: int
    :ivar timeout: Timeout in seconds for LLM API calls. Defaults to 120 seconds.
    :vartype timeout: int
    :ivar pricing_details: LLMPricing object with pricing details for cost calculation.
        Defaults to None.
    :vartype pricing_details: LLMPricing | None
    :ivar auto_pricing: Enable automatic LLM cost calculation using genai-prices.
        Ignored when ``pricing_details`` is provided. Defaults to ``False``.
    :vartype auto_pricing: bool
    :ivar auto_pricing_refresh: Whether genai-prices should auto-refresh its cached
        pricing data. Defaults to ``False``.
    :vartype auto_pricing_refresh: bool
    :ivar is_fallback: Indicates whether the LLM is a fallback model. Defaults to False.
    :vartype is_fallback: bool
    :ivar fallback_llm: DocumentLLM to use as fallback if current one fails.
        Must have the same role as the current LLM. Defaults to None.
    :vartype fallback_llm: DocumentLLM | None
    :ivar output_language: Language for produced output text (justifications, explanations).
        Can be "en" (English) or "adapt" (adapts to document/image language). Defaults to "en".
        Applies only when DocumentLLM's default system message is used.
    :vartype output_language: LanguageRequirement
    :ivar async_limiter: Controls frequency of async LLM API requests for concurrent tasks.
        Defaults to allowing 3 acquisitions per 10-second period to prevent rate limit issues.
        See https://github.com/mjpieters/aiolimiter for configuration details.
    :vartype async_limiter: AsyncLimiter
    :ivar seed: Seed for random number generation to help produce more consistent outputs
        across multiple runs. When set to a specific integer value, the LLM will attempt
        to use this seed for sampling operations. However, deterministic output is still
        not guaranteed even with the same seed, as other factors may influence the model's
        response. Defaults to None.
    :vartype seed: int | None

    Note:

        - LLM groups
            Refer to the :class:`DocumentLLMGroup` class for more information on constructing LLM groups,
            which are a collection of LLMs with unique roles, used for complex document processing tasks.

        - LLM role
            The ``role`` of an LLM is an abstraction to differentiate between tasks of different complexity.
            For example, if an aspect/concept is assigned ``llm_role="extractor_text"``, it means that the
            aspect/concept is extracted from the document using the LLM with the ``role`` set to "extractor_text".
            This helps to channel different tasks to different LLMs, ensuring that the task is handled
            by the most appropriate model. Usually, domain expertise is required to determine the most
            appropriate role for a specific aspect/concept. But for simple use cases, you can skip the
            role assignment completely, in which case the ``role`` will default to "extractor_text".

        - Explicit capability declaration
            Model vision capabilities are automatically detected using
            ``litellm.supports_vision()``. If this function does not correctly identify your model's capabilities,
            ContextGem will typically issue a warning, and you can explicitly declare the capability by
            setting ``_supports_vision=True`` on the LLM instance.

    Example:
        .. literalinclude:: ../../../dev/usage_examples/docstrings/llms/def_llm.py
            :language: python
            :caption: LLM definition
    """

    pass


@_expose_in_registry(additional_key=_ChatSession)
class ChatSession(_ChatSession):
    """
    Stateful chat session that preserves message history across turns.

    To be used as ``chat_session=...`` parameter for ``DocumentLLM.chat(...)``
    or ``DocumentLLM.chat_async(...)``.
    """

    pass
