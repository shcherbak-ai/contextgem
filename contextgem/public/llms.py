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

import asyncio
import warnings
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from aiolimiter import AsyncLimiter
from jinja2 import Template

from contextgem.internal.exceptions import LLMAPIError
from contextgem.internal.utils import _suppress_litellm_pydantic_warnings_context


with _suppress_litellm_pydantic_warnings_context():
    import litellm

from pydantic import (
    Field,
    PrivateAttr,
    StrictBool,
    StrictFloat,
    StrictInt,
    field_validator,
    model_validator,
)


if TYPE_CHECKING:
    from contextgem.internal.data_models import (
        _LLMCostOutputContainer,
        _LLMUsageOutputContainer,
    )

from typing_extensions import Self

from contextgem.internal.base.llms import _GenericLLMProcessor
from contextgem.internal.data_models import _LLMCall, _LLMCost, _LLMUsage
from contextgem.internal.decorators import _post_init_method
from contextgem.internal.loggers import logger
from contextgem.internal.typings.aliases import (
    DefaultPromptType,
    LanguageRequirement,
    LLMRoleAny,
    NonEmptyStr,
    ReasoningEffort,
)
from contextgem.internal.utils import _get_template, _run_sync, _setup_jinja2_template
from contextgem.public.data_models import LLMPricing
from contextgem.public.images import Image


litellm.suppress_debug_info = True


class DocumentLLMGroup(_GenericLLMProcessor):
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
                          Defaults to "en".
    :vartype output_language: LanguageRequirement

    Note:
        Refer to the :class:`DocumentLLM` class for more information on constructing LLMs
        for the group.

    Example:
        .. literalinclude:: ../../../dev/usage_examples/docstrings/llms/def_llm_group.py
            :language: python
            :caption: LLM group definition
    """

    llms: list[DocumentLLM] = Field(..., min_length=2)
    output_language: LanguageRequirement = Field(default="en")

    _llm_extractor_text: DocumentLLM | None = PrivateAttr(default=None)
    _llm_reasoner_text: DocumentLLM | None = PrivateAttr(default=None)
    _llm_extractor_vision: DocumentLLM | None = PrivateAttr(default=None)
    _llm_reasoner_vision: DocumentLLM | None = PrivateAttr(default=None)

    def _set_private_attrs(self) -> None:
        """
        Initialize and configure private attributes for the LLM group.

        :return: None
        :rtype: None
        """
        self._assign_role_specific_llms()

    @property
    def is_group(self) -> bool:
        """
        Returns True indicating this is a group of LLMs.

        :return: Always True for DocumentLLMGroup instances.
        :rtype: bool
        """
        return True

    @property
    def list_roles(self) -> list[LLMRoleAny]:
        """
        Returns a list of all roles assigned to the LLMs in this group.

        :return: A list of LLM role identifiers
        :rtype: list[LLMRoleAny]
        """
        return [i.role for i in self.llms]

    def group_update_output_language(
        self, output_language: LanguageRequirement
    ) -> None:
        """
        Updates the output language for all LLMs in the group.

        :param output_language: The new output language to set for all LLMs
        :type output_language: LanguageRequirement
        """
        for llm in self.llms:
            llm.output_language = output_language
        self.output_language = output_language
        logger.info(
            f"Updated output language for all LLMs in the group to `{output_language}`"
        )

    def _eq_deserialized_llm_config(
        self,
        other: DocumentLLMGroup,
    ) -> bool:
        """
        Custom config equality method to compare this DocumentLLMGroup with a deserialized instance.

        Uses the `_eq_deserialized_llm_config` method of the DocumentLLM class to compare each LLM
        in the group, including fallbacks, if any.

        :param other: Another DocumentLLMGroup instance to compare with
        :type other: DocumentLLMGroup
        :return: True if the instances are equal, False otherwise
        :rtype: bool
        """
        for self_llm, other_llm in zip(self.llms, other.llms, strict=True):
            if not self_llm._eq_deserialized_llm_config(other_llm):
                return False
        return True

    @field_validator("llms")
    @classmethod
    def _validate_llms(cls, llms: list[DocumentLLM]) -> list[DocumentLLM]:
        """
        Validates the provided list of DocumentLLMs ensuring that each
        LLM has a unique role within the group.

        :param llms: A list of `DocumentLLM` instances to be validated.
        :return: A validated list of `DocumentLLM` instances.
        :raises ValueError: If validation fails.
        """
        seen_roles = set()
        for llm in llms:
            if llm.role in seen_roles:
                raise ValueError("LLMs must have different roles.")
            seen_roles.add(llm.role)
        return llms

    def _assign_role_specific_llms(self) -> None:
        """
        Assigns specific LLMs to dedicated roles based on the role attribute of each LLM.

        :return: None
        """

        def get_llm_by_role(role: str) -> DocumentLLM | None:
            """
            Finds and returns the first LLM with the specified role.

            :param role: The role to search for among the LLMs.
            :type role: str
            :return: The DocumentLLM with the matching role, or None if not found.
            :rtype: DocumentLLM | None
            """
            return next((i for i in self.llms if i.role == role), None)

        self._llm_extractor_text = get_llm_by_role("extractor_text")
        self._llm_reasoner_text = get_llm_by_role("reasoner_text")
        self._llm_extractor_vision = get_llm_by_role("extractor_vision")
        self._llm_reasoner_vision = get_llm_by_role("reasoner_vision")

    def get_usage(self, llm_role: str | None = None) -> list[_LLMUsageOutputContainer]:
        """
        Retrieves the usage information of the LLMs in the group, filtered by the specified
        LLM role if provided.

        :param llm_role: Optional; A string representing the role of the LLM to filter
            the usage data. If None, returns usage for all LLMs in the group.
        :type llm_role: str | None
        :return: A list of usage statistics containers for the specified LLMs and their fallbacks.
        :rtype: list[_LLMUsageOutputContainer]
        :raises ValueError: If no LLM with the specified role exists in the group.
        """
        # Safe cast: _get_usage_or_cost with retrieval_type="usage"
        # returns only list of _LLMUsageOutputContainer
        return cast(
            list["_LLMUsageOutputContainer"],
            self._get_usage_or_cost(retrieval_type="usage", llm_role=llm_role),
        )

    def get_cost(self, llm_role: str | None = None) -> list[_LLMCostOutputContainer]:
        """
        Retrieves the accumulated cost information of the LLMs in the group, filtered by the specified
        LLM role if provided.

        :param llm_role: Optional; A string representing the role of the LLM to filter
            the cost data. If None, returns cost for all LLMs in the group.
        :type llm_role: str | None
        :return: A list of cost statistics containers for the specified LLMs and their fallbacks.
        :rtype: list[_LLMCostOutputContainer]
        :raises ValueError: If no LLM with the specified role exists in the group.
        """
        # Safe cast: _get_usage_or_cost with retrieval_type="cost"
        # returns only list of _LLMCostOutputContainer
        return cast(
            list["_LLMCostOutputContainer"],
            self._get_usage_or_cost(retrieval_type="cost", llm_role=llm_role),
        )

    def reset_usage_and_cost(self, llm_role: str | None = None) -> None:
        """
        Resets the usage and cost statistics for LLMs in the group.

        This method clears accumulated usage and cost data, which is useful when processing
        multiple documents sequentially and tracking metrics for each document separately.

        :param llm_role: Optional; A string representing the role of the LLM to reset statistics for.
            If None, resets statistics for all LLMs in the group.
        :type llm_role: str | None
        :raises ValueError: If no LLM with the specified role exists in the group.
        :return: None
        """
        if llm_role:
            try:
                llm = next(i for i in self.llms if i.role == llm_role)
                llm.reset_usage_and_cost()
            except StopIteration:
                raise ValueError(
                    f"No LLM with the given role `{llm_role}` was found in group."
                ) from None
        else:
            for llm in self.llms:
                llm.reset_usage_and_cost()

    @model_validator(mode="after")
    def _validate_document_llm_group_post(self) -> Self:
        """
        Validates the LLM group to ensure consistency of the `output_language`
        attribute across all LLMs within the group.

        Raises:
            ValueError: Raised if any LLM's `output_language` differs from the
            group's `output_language`.

        :return: The LLM group instance after successful validation.
        :rtype: Self
        """
        # Set private attributes before validation
        self._set_private_attrs()

        if any(i.output_language != self.output_language for i in self.llms):
            raise ValueError(
                "All LLMs in the group must have the same value of "
                "`output_language` attribute as the group."
            )
        return self


class DocumentLLM(_GenericLLMProcessor):
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
    :ivar role: Role type for the LLM (e.g., "extractor_text", "reasoner_text",
        "extractor_vision", "reasoner_vision"). Defaults to "extractor_text".
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
        ``"low"``, ``"medium"``, or ``"high"``. Relevant for reasoning (CoT-capable) models.
        Defaults to None.
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
    :ivar is_fallback: Indicates whether the LLM is a fallback model. Defaults to False.
    :vartype is_fallback: bool
    :ivar fallback_llm: DocumentLLM to use as fallback if current one fails.
        Must have the same role as the current LLM. Defaults to None.
    :vartype fallback_llm: DocumentLLM | None
    :ivar output_language: Language for produced output text (justifications, explanations).
        Can be "en" (English) or "adapt" (adapts to document/image language). Defaults to "en".
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

    # LLM config
    model: NonEmptyStr
    deployment_id: NonEmptyStr | None = Field(default=None)
    api_key: NonEmptyStr | None = Field(default=None)
    api_base: NonEmptyStr | None = Field(default=None)
    api_version: NonEmptyStr | None = Field(default=None)  # specific to Azure OpenAI
    role: LLMRoleAny = Field(default="extractor_text")
    system_message: str | None = Field(default=None)
    temperature: StrictFloat | None = Field(default=0.3, ge=0)
    max_tokens: StrictInt = Field(default=4096, gt=0)
    max_completion_tokens: StrictInt = Field(
        default=16000, gt=0
    )  # for reasoning (CoT-capable) models
    reasoning_effort: ReasoningEffort | None = Field(
        default=None
    )  # for reasoning (CoT-capable) models
    top_p: StrictFloat | None = Field(default=0.3, ge=0)
    num_retries_failed_request: StrictInt = Field(default=3, ge=0)
    max_retries_failed_request: StrictInt = Field(default=0, ge=0)  # provider-specific
    max_retries_invalid_data: StrictInt = Field(default=3, ge=0)
    timeout: StrictInt = Field(default=120, ge=0)
    pricing_details: LLMPricing | None = Field(default=None)
    is_fallback: StrictBool = Field(default=False)
    fallback_llm: DocumentLLM | None = Field(default=None)
    output_language: LanguageRequirement = Field(default="en")
    seed: StrictInt | None = Field(default=None)

    # Prompts
    _extract_aspect_items_prompt: Template = PrivateAttr()
    _extract_concept_items_prompt: Template = PrivateAttr()

    # Async
    _async_limiter: AsyncLimiter = PrivateAttr()

    # Token counting
    _usage: _LLMUsage = PrivateAttr(default_factory=_LLMUsage)
    # Processing cost
    _cost: _LLMCost = PrivateAttr(default_factory=_LLMCost)

    # Async lock to guard shared state during async updates
    _async_lock: asyncio.Lock = PrivateAttr(default_factory=asyncio.Lock)

    # Capabilities (can be overridden by users if litellm.supports_*()
    # is not accurate for the model)
    _supports_vision: bool = PrivateAttr(default=False)
    _supports_reasoning: bool = PrivateAttr(default=False)

    def __init__(self, **data: Any):
        # Pop the async_limiter if provided; otherwise use a default.
        limiter = data.pop("async_limiter", None)
        super().__init__(**data)
        if limiter is not None:
            self.async_limiter = limiter
        else:
            self.async_limiter = AsyncLimiter(3, 10)

    @_post_init_method
    def _post_init(self, __context: Any):
        """
        Post-initialization method that logs model information and API configuration.

        :param __context: Pydantic context (unused).
        :type __context: Any
        """
        logger.info(f"Using model {self.model}")
        if self.api_key is None:
            logger.info("API key was not provided. Set `api_key`, if applicable.")
        if self.api_base is None:
            logger.info("API base was not provided. Set `api_base`, if applicable.")

        # Log helpful message for local model providers that may use smaller models
        local_providers = [
            "ollama/",
            "ollama_chat/",
            "lm_studio/",
        ]
        if any(self.model.startswith(provider) for provider in local_providers):
            logger.info(
                "Using local model provider. If you experience issues like JSON validation errors "
                "with smaller models, see our troubleshooting guide: "
                "https://contextgem.dev/optimizations/optimization_small_llm_troubleshooting.html"
            )

        # Recommend `ollama_chat` prefix for better responses for Ollama models (text-only processing)
        if self.model.startswith("ollama/") and not self.role.endswith("_vision"):
            logger.info(
                "For better responses with Ollama models, consider using "
                "'ollama_chat/' prefix instead of 'ollama/', as recommended by LiteLLM: "
                "https://docs.litellm.ai/docs/providers/ollama"
            )

        # Warn to use `ollama/` prefix for image processing when using local vision models,
        # as the ollama_chat/ does not yet support image inputs
        if self.model.startswith("ollama_chat/") and self.role.endswith("_vision"):
            warnings.warn(
                "Using `ollama_chat/` prefix for local vision models is not recommended, "
                "as it does not yet support image inputs. Please use `ollama/` prefix instead. "
                "See https://github.com/ollama/ollama/issues/10255 and "
                "https://github.com/ollama/ollama/issues/6451 for more details.",
                UserWarning,
                stacklevel=2,
            )

        # TODO: Remove this once LiteLLM's `lm_studio` gpt-oss support is fixed
        # Warn when gpt-oss models are used with LM Studio due to instability
        if self.model.startswith("lm_studio/") and "openai/gpt-oss" in self.model:
            warnings.warn(
                "Using OpenAI gpt-oss models with `lm_studio/` is currently unstable and may "
                "cause errors. For a stable alternative, consider using the model via Ollama "
                "instead (e.g., `ollama_chat/gpt-oss:20b`).",
                UserWarning,
                stacklevel=2,
            )

    def _set_private_attrs(self) -> None:
        """
        Initialize and configure private attributes for the LLM instance.

        :return: None
        :rtype: None
        """
        if self.system_message is None:
            self._set_system_message()
        self._set_prompts()
        self._set_capabilities()

    @property
    def async_limiter(self) -> AsyncLimiter:
        """
        Gets the async rate limiter for this LLM.

        :return: The AsyncLimiter instance controlling request rate limits.
        :rtype: AsyncLimiter
        """
        return self._async_limiter

    @async_limiter.setter
    def async_limiter(self, value: AsyncLimiter) -> None:
        """
        Sets the async rate limiter for this LLM.

        :param value: The AsyncLimiter instance to set.
        :type value: AsyncLimiter
        :raises TypeError: If value is not an AsyncLimiter instance.
        """
        if not isinstance(value, AsyncLimiter):
            raise TypeError("async_limiter must be an AsyncLimiter instance")
        self._async_limiter = value

    @property
    def is_group(self) -> bool:
        """
        Returns False indicating this is a single LLM, not a group.

        :return: Always False for DocumentLLM instances.
        :rtype: bool
        """
        return False

    @property
    def list_roles(self) -> list[LLMRoleAny]:
        """
        Returns a list containing the role of this LLM.

        (For a single LLM, this returns a list with just one element - the LLM's role.
        For LLM groups, the method implementation returns roles of all LLMs in the group.)

        :return: A list containing the role of this LLM.
        :rtype: list[LLMRoleAny]
        """
        return [self.role]

    def chat(self, prompt: str, images: list[Image] | None = None) -> str:
        """
        Synchronously sends a prompt to the LLM and gets a response.
        For models supporting vision, attach images to the prompt if needed.

        This method allows direct interaction with the LLM by submitting your own prompt.

        :param prompt: The input prompt to send to the LLM
        :type prompt: str
        :param images: Optional list of Image instances for vision queries
        :type images: list[Image] | None
        :return: The LLM's response
        :rtype: str
        :raises ValueError: If the prompt is empty or not a string
        :raises ValueError: If images parameter is not a list of Image instances
        :raises ValueError: If images are provided but the model doesn't support vision
        :raises RuntimeError: If the LLM call fails and no fallback is available
        """
        return _run_sync(self.chat_async(prompt, images))

    async def chat_async(self, prompt: str, images: list[Image] | None = None) -> str:
        """
        Asynchronously sends a prompt to the LLM and gets a response.
        For models supporting vision, attach images to the prompt if needed.

        This method allows direct interaction with the LLM by submitting your own prompt.

        :param prompt: The input prompt to send to the LLM
        :type prompt: str
        :param images: Optional list of Image instances for vision queries
        :type images: list[Image] | None
        :return: The LLM's response
        :rtype: str
        :raises ValueError: If the prompt is empty or not a string
        :raises ValueError: If images parameter is not a list of Image instances
        :raises ValueError: If images are provided but the model doesn't support vision
        :raises RuntimeError: If the LLM call fails and no fallback is available
        """

        # Validate prompt
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Prompt must be a non-empty string")

        # Validate images
        if images and (
            not isinstance(images, list)
            or not all(isinstance(image, Image) for image in images)
        ):
            raise ValueError("Images must be a list of Image instances")

        # Check for vision support
        if images and not self._supports_vision:
            raise ValueError(
                f"Model `{self.model}` does not support vision according to "
                f"litellm.supports_vision(). To override this detection, "
                f"manually set `_supports_vision=True` on the LLM instance."
            )

        # Create LLM call object to track the interaction
        llm_call = _LLMCall(prompt_kwargs={}, prompt=prompt)

        # Warn if using default system message
        # _get_template returns a Template object when template_extension == "j2"
        default_system_message = _get_template(
            "default_system_message",
            template_type="system",
            template_extension="j2",
        ).render({"output_language": self.output_language})  # type: ignore[attr-defined]
        if self.system_message == default_system_message:
            warnings.warn(
                "You are using the default system message optimized for extraction tasks. "
                "For simple chat interactions, consider setting system_message='' to disable it, "
                "or provide your own custom system message.",
                stacklevel=2,
            )

        # Send message to LLM
        result = await self._query_llm(
            message=prompt,
            llm_call_obj=llm_call,
            images=images,
            num_retries_failed_request=self.num_retries_failed_request,
            max_retries_failed_request=self.max_retries_failed_request,
            raise_exception_on_llm_api_error=True,  # always True for chat
        )

        # Update usage and cost statistics
        await self._update_usage_and_cost(result)

        response, _ = result

        # If response is None and fallback LLM is available, try with fallback LLM
        if response is None and self.fallback_llm:
            logger.info(f"Using fallback LLM {self.fallback_llm.model} for chat")
            return await self.fallback_llm.chat_async(prompt, images)
        elif response is None:
            raise RuntimeError(
                f"Failed to get response from LLM {self.model} and no fallback is available"
            )

        return response

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
        other: DocumentLLM,
    ) -> bool:
        """
        Custom config equality method to compare this DocumentLLM with a deserialized instance.

        Compares the __dict__ of both instances and performs specific checks for
        certain attributes that require special handling.

        Note that, by default, the reconstructed deserialized DocumentLLM will be only partially
        equal (==) to the original one, as the api credentials are redacted, and the attached prompt
        templates, async limiter, and async lock are not serialized and point to different objects
        in memory post-initialization. Also, usage and cost are reset by default pre-serialization.

        :param other: Another DocumentLLM instance to compare with
        :type other: DocumentLLM
        :return: True if the instances are equal, False otherwise
        :rtype: bool
        """

        # Create a copy of the dictionaries to modify
        self_dict = self.__dict__.copy()
        other_dict = other.__dict__.copy()

        # Skips check for fallback LLM, if it is set
        if self.fallback_llm:
            self_fallback_llm = self_dict.pop("fallback_llm")
            other_fallback_llm = other_dict.pop("fallback_llm")
            if not other_fallback_llm:
                raise RuntimeError("Deserialized fallback LLM was not set")
            if not self_fallback_llm._eq_deserialized_llm_config(other_fallback_llm):
                logger.debug("Fallback LLM config of deserialized LLM is different.")
                return False

        # Skip checks for api_key and api_base that were redacted pre-serialization
        self_dict.pop("api_key")
        other_dict.pop("api_key")
        self_dict.pop("api_base")
        other_dict.pop("api_base")
        if not (other.api_key is None and other.api_base is None):
            raise RuntimeError(
                "Deserialized LLM has api_key or api_base set, "
                "while API credentials should have been redacted pre-serialization"
            )

        # Compare the modified dictionaries
        if self_dict != other_dict:
            logger.debug("LLM __dict__ of deserialized LLM is different.")
            return False

        # Special checks for specific private attributes

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

        # Check that usage and cost stats were reset pre-serialization
        if not (other._usage == _LLMUsage() and other._cost == _LLMCost()):
            raise RuntimeError(
                "Usage and cost stats were not properly reset during serialization"
            )

        # Check _async_limiter
        if (
            self._async_limiter.time_period != other._async_limiter.time_period
            or self._async_limiter.max_rate != other._async_limiter.max_rate
        ):
            logger.debug("Async limiter params of deserialized LLM are different.")
            return False

        # Check _async_lock
        if not (
            isinstance(self._async_lock, asyncio.Lock)
            and isinstance(other._async_lock, asyncio.Lock)
        ):
            logger.debug("Async lock of deserialized LLM is different.")
            return False

        # Check capabilities
        if self._supports_vision != other._supports_vision:
            logger.debug("`_supports_vision` of deserialized LLM is different.")
            return False
        if self._supports_reasoning != other._supports_reasoning:
            logger.debug("`_supports_reasoning` of deserialized LLM is different.")
            return False

        return True

    @field_validator("model")
    @classmethod
    def _validate_model(cls, model: str) -> str:
        """
        Validates the model identifier to ensure it conforms to the expected format.

        :param model: Model identifier string to validate.
        :type model: str
        :return: The validated model string.
        :rtype: str
        :raises ValueError: If `model` does not contain a forward slash ('/') to indicate
            the required format.
        """
        if "/" not in model:
            raise ValueError(
                "Model identifier must be in the form of `<model_provider>/<model_name>`. "
                "See https://docs.litellm.ai/docs/providers for the list of supported providers."
            )
        return model

    @field_validator("fallback_llm")
    @classmethod
    def _validate_fallback_llm(
        cls, fallback_llm: DocumentLLM | None
    ) -> DocumentLLM | None:
        """
        Validates the ``fallback_llm`` input to ensure it meets the expected condition
        of being a fallback LLM model.

        :param fallback_llm: The DocumentLLM instance to be validated.
        :type fallback_llm: DocumentLLM
        :return: The valid fallback_llm that meets the expected criteria.
        :rtype: DocumentLLM
        :raises ValueError: If the ``fallback_llm`` is not a fallback model, as
            indicated by the ``is_fallback`` attribute set to ``False``.
        """
        if fallback_llm is not None and not fallback_llm.is_fallback:
            raise ValueError(
                "Fallback LLM must be a fallback model. Use `is_fallback=True`."
            )
        return fallback_llm

    @model_validator(mode="after")
    def _validate_document_llm_post(self) -> Self:
        """
        Validate the integrity of the document LLM model after initialization.

        If the LLM has a vision role, validates that the model supports vision.

        Also ensures that the fallback logic is correctly implemented, primarily adhering to two constraints:
        1. A fallback LLM cannot itself have a fallback model associated with it.
        2. A fallback LLM must share the same role and output language as its parent LLM to maintain consistency.
        3. A fallback LLM cannot have the same parameters as its parent LLM,
            except for ``is_fallback`` and ``fallback_llm``.

        :raises ValueError: Raised when one of the following conditions is violated:
            - If the LLM has a vision role but does not support vision.
            - If an LLM marked as fallback has its own fallback LLM.
            - If a fallback LLM is assigned a role different from the main LLM.
            - If a fallback LLM has the same parameters as the parent LLM, except for
                ``is_fallback`` and ``fallback_llm``.
        :return: Returns the instance of the current LLM model after successful validation.
        :rtype: Self
        """

        # Set private attributes before validation
        self._set_private_attrs()

        # Vision support validation, when applicable
        if self.role.endswith("_vision") and not self._supports_vision:
            # Allow the user to override _supports_vision if the model is known to support
            # vision while litellm does not detect it as vision-capable
            warnings.warn(
                f"Model `{self.model}` is assigned vision role `{self.role}` but "
                f"litellm does not detect it as vision-capable. This may cause "
                f"vision-related operations to fail. If you know this model supports "
                f"vision, manually set `_supports_vision=True` on the LLM instance.",
                UserWarning,
                stacklevel=2,
            )

        if self.reasoning_effort and not self._supports_reasoning:
            # Allow the user to override _supports_reasoning if the model is known to support
            # reasoning while litellm does not detect it as reasoning-capable
            warnings.warn(
                f"Model `{self.model}` has `reasoning_effort` set but "
                f"litellm does not detect it as reasoning-capable. If you know this model "
                f"supports reasoning, manually set `_supports_reasoning=True` on the LLM instance.",
                UserWarning,
                stacklevel=2,
            )

        # Fallback model validation
        if self.is_fallback and self.fallback_llm:
            raise ValueError(
                "Fallback LLM cannot have its own fallback LLM "
                "and must be attached to a non-fallback model."
            )

        if self.fallback_llm:
            # Check for the consistency of the fallback LLM role and output language
            if self.fallback_llm.role != self.role:
                raise ValueError(
                    f"The fallback LLM must have the same role `{self.role}` as the main one."
                )
            elif self.fallback_llm.output_language != self.output_language:
                raise ValueError(
                    f"The fallback LLM must have the same output language `{self.output_language}` as the main one."
                )

            # Check that the fallback LLM is not the replica of the main LLM, just with different
            # `is_fallback` and `fallback_llm` params
            main_llm_dict = {
                k: v
                for k, v in self.__dict__.items()
                if k not in ["is_fallback", "fallback_llm"]
            }
            fallback_llm_dict = {
                k: v
                for k, v in self.fallback_llm.__dict__.items()
                if k not in ["is_fallback", "fallback_llm"]
            }
            if main_llm_dict == fallback_llm_dict:
                raise ValueError(
                    "Fallback LLM must not have the exact same config params as the main LLM."
                )
        return self

    def _validate_input_tokens(self, messages: list[dict[str, str]]) -> None:
        """
        Validates that the input messages do not exceed the model's maximum input tokens.

        :param messages: List of message dictionaries to validate
        :type messages: list[dict[str, str]]
        :raises ValueError: If the messages exceed the model's maximum input tokens
        :return: None
        """
        context_exceeded = False
        try:
            # Get model information to check context window
            model_info = litellm.get_model_info(self.model)  # type: ignore[attr-defined]
            max_input_tokens = model_info.get("max_input_tokens")

            # If max_input_tokens is not available, skip validation
            if max_input_tokens is None:
                logger.warning(
                    f"Could not determine max_input_tokens for model `{self.model}`. Skipping validation."
                )
                return

            # Count tokens in the messages
            try:
                token_count = litellm.token_counter(model=self.model, messages=messages)  # type: ignore[attr-defined]
            except Exception as e:
                logger.warning(
                    f"Could not count tokens for model `{self.model}`: {e}. Skipping input token validation."
                )
                return

            # Check if we exceed the context window
            if token_count > max_input_tokens:
                context_exceeded = True
                raise ValueError(
                    f"Input messages contain {token_count} tokens, which exceeds the model's "
                    f"maximum input tokens of {max_input_tokens} for model `{self.model}`."
                )

            logger.debug(
                f"Input token validation passed: {token_count}/{max_input_tokens} tokens used"
            )

        except ValueError as e:
            if context_exceeded:
                # Re-raise our own ValueError
                raise
            # If it's a different ValueError, log and continue
            logger.warning(
                f"Could not validate max input tokens for model `{self.model}`: {e}"
            )
        except Exception as e:
            # If we can't get model info, log a warning but don't fail
            logger.warning(
                f"Could not validate max input tokens for model `{self.model}`: {e}"
            )

    def _validate_output_tokens(self) -> None:
        """
        Validates that the configured max_tokens or max_completion_tokens do not exceed
        the model's maximum output tokens.

        :raises ValueError: If the configured tokens exceed the model's maximum output tokens
        :return: None
        """
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
                logger.warning(
                    f"Could not determine max_output_tokens for model `{self.model}`. "
                    f"Skipping max output token validation."
                )
                return

            # Determine which token limit to check based on model type
            if self._supports_reasoning:
                configured_tokens = self.max_completion_tokens
                token_type = "max_completion_tokens"  # nosec B105 - not a password
            else:
                configured_tokens = self.max_tokens
                token_type = "max_tokens"  # nosec B105 - not a password

            # Check if configured tokens exceed the model's limit
            if configured_tokens > max_output_tokens:
                output_exceeded = True
                raise ValueError(
                    f"Configured {token_type} ({configured_tokens}) exceeds the model's "
                    f"maximum output tokens of {max_output_tokens} for model `{self.model}`."
                )

            logger.debug(
                f"Output token validation passed: {configured_tokens}/{max_output_tokens} "
                f"{token_type} configured"
            )

        except ValueError as e:
            if output_exceeded:
                # Re-raise our own ValueError
                raise
            # If it's a different ValueError, log and continue
            logger.warning(
                f"Could not validate max output tokens for model `{self.model}`: {e}"
            )
        except Exception as e:
            # If we can't get model info, log a warning but don't fail
            logger.warning(
                f"Could not validate max output tokens for model `{self.model}`: {e}"
            )

    async def _query_llm(
        self,
        message: str,
        llm_call_obj: _LLMCall,
        images: list[Image] | None = None,
        num_retries_failed_request: int = 3,
        max_retries_failed_request: int = 0,
        async_limiter: AsyncLimiter | None = None,
        drop_params: bool = False,
        raise_exception_on_llm_api_error: bool = True,
    ) -> tuple[str | None, _LLMUsage]:
        """
        Generates a response from an LLM based on the provided message, optional images,
        and system configuration. It formats the input messages according to the
        compatibility with different versions of the LLM, sends the request to the
        LLM API, and processes the generated response.

        :param message: The input message from the user intended for the LLM.
        :type message: str
        :param llm_call_obj: The _LLMCall object holding data on the initiated LLM call.
        :type llm_call_obj: _LLMCall
        :param images: Optional list of Image instances for vision queries.
            If provided, the query will be processed as a vision request.
        :type images: list[Image] | None
        :param num_retries_failed_request: Optional number of retries when LLM request fails. Defaults to 3.
            Note that this parameter may override the value set on the LLM instance to prevent
            accumulation of retries from failed requests and invalid data generation.
        :type num_retries_failed_request: int
        :param max_retries_failed_request: Specific to certain provider APIs (e.g. OpenAI). Optional number of
            retries when LLM request fails. Defaults to 0. This parameter may override the value set on
            the LLM instance to prevent accumulation of retries from failed requests and invalid data generation.
        :type max_retries_failed_request: int
        :param async_limiter: An optional aiolimiter.AsyncLimiter instance that controls the frequency of
            async LLM API requests, when concurrency is enabled for certain tasks. If not provided,
            such requests will be sent synchronously.
        :type async_limiter: AsyncLimiter | None
        :param drop_params: Whether to drop unsupported parameters when calling the LLM API.
            Used internally for automatic retry when UnsupportedParamsError occurs.
        :type drop_params: bool
        :param raise_exception_on_llm_api_error: Whether to raise an exception if the LLM call fails
            due to an error in the LLM API. If False, a warning will be issued instead, and no data
            will be returned. Defaults to True.
        :type raise_exception_on_llm_api_error: bool, optional
        :return: A tuple containing the LLM response and usage statistics.
            The LLM response is None if the LLM call fails.
        :rtype: tuple[str | None, _LLMUsage]
        """

        if images and not self._supports_vision:
            raise ValueError(
                f"Model `{self.model}` does not support vision according to "
                f"litellm.supports_vision(). To override this detection, "
                f"manually set `_supports_vision=True` on the LLM instance."
            )

        request_messages = []

        # Handle system message based on model type
        if self.system_message and self.system_message.strip():
            if not any(i in self.model for i in ["o1-preview", "o1-mini"]):
                # o1/o1-mini models don't support system/developer messages
                request_messages.append(
                    {
                        "role": "system",
                        "content": self.system_message,
                    }
                )
            else:
                warnings.warn(
                    f"System message ignored for the model `{self.model}`.",
                    stacklevel=2,
                )

        # Prepare user message content based on whether images are provided
        if images:
            user_message_content: list[dict[str, str | dict[str, str]]] = [
                {"type": "text", "text": message}
            ]
            for image in images:
                user_message_content.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{image.mime_type};base64,{image.base64_data}",
                        },
                    },
                )
            request_messages.append(
                {
                    "role": "user",
                    "content": user_message_content,
                }
            )
        else:
            request_messages.append(
                {
                    "role": "user",
                    "content": message,
                }
            )

        # Validate max input / output tokens before making the API call
        self._validate_input_tokens(request_messages)
        self._validate_output_tokens()

        # Prepare request dictionary with common parameters
        request_dict = {
            "model": self.model,
            "messages": request_messages,
        }

        # Add model-specific parameters
        if self._supports_reasoning:
            # Reasoning (CoT-capable) models
            model_params: list[str] | None = litellm.get_supported_openai_params(  # type: ignore[attr-defined]
                self.model
            )
            if model_params is not None:
                if "/gpt-5" in self.model and "reasoning_effort" not in model_params:
                    # As of 2025-08-07, LiteLLM does not yet list "reasoning effort" as
                    # a param for gpt-5 models.
                    # TODO: Remove this conditional logic once litellm has "reasoning_effort"
                    # param listed for gpt-5 models.
                    model_params.append("reasoning_effort")
                    request_dict["allowed_openai_params"] = ["reasoning_effort"]
                if "max_completion_tokens" in model_params:
                    if not (self.max_completion_tokens):
                        raise ValueError(
                            "`max_completion_tokens` must be set for reasoning (CoT-capable) models"
                        )
                    request_dict["max_completion_tokens"] = self.max_completion_tokens
                if "reasoning_effort" in model_params and self.reasoning_effort:
                    request_dict["reasoning_effort"] = self.reasoning_effort
            if self.temperature or self.top_p:
                logger.info(
                    "`temperature` and `top_p` parameters are ignored for reasoning (CoT-capable) models"
                )
        else:
            # Non-reasoning models
            request_dict["max_tokens"] = self.max_tokens

            # temperature and top_p might not be supported by some models
            if self.temperature is not None:
                request_dict["temperature"] = self.temperature
            if self.top_p is not None:
                request_dict["top_p"] = self.top_p

        if self.deployment_id:
            # Azure OpenAI-specific
            request_dict["deployment_id"] = self.deployment_id

        if self.seed:
            request_dict["seed"] = self.seed

        # Create an empty usage dict in case the call fails without the possibility to retrieve usage tokens
        usage = _LLMUsage()

        # Make API call and process response
        try:
            task = asyncio.create_task(
                litellm.acompletion(
                    **request_dict,
                    api_key=self.api_key,
                    api_base=self.api_base,
                    api_version=self.api_version,
                    num_retries=num_retries_failed_request,
                    max_retries=max_retries_failed_request,
                    timeout=self.timeout,
                    stream=False,
                    drop_params=drop_params,
                )
            )
            if async_limiter:
                async with async_limiter:
                    chat_completion = await task
            else:
                chat_completion = await task
            answer = (
                chat_completion.choices[  # type: ignore[attr-defined]
                    0
                ].message.content  # type: ignore[attr-defined]
            )  # str, or None if invalid response
            usage.input = chat_completion.usage.prompt_tokens  # type: ignore[attr-defined]
            usage.output = chat_completion.usage.completion_tokens  # type: ignore[attr-defined]
            llm_call_obj._record_response_timestamp()
            llm_call_obj.response = answer
            usage.calls.append(llm_call_obj)  # record the call details (call finished)
            return answer, usage
        except litellm.UnsupportedParamsError as e:  # type: ignore[attr-defined]
            # Handle unsupported model parameters error
            if (
                not drop_params
            ):  # only retry if we haven't already tried with drop_params
                logger.error(f"Exception occurred while calling LLM API: {e}")
                logger.info("Retrying the call with unsupported parameters dropped...")

                # Recursively call with drop_params=True
                return await self._query_llm(
                    message=message,
                    llm_call_obj=llm_call_obj,
                    images=images,
                    num_retries_failed_request=num_retries_failed_request,
                    max_retries_failed_request=max_retries_failed_request,
                    async_limiter=async_limiter,
                    drop_params=True,
                    raise_exception_on_llm_api_error=raise_exception_on_llm_api_error,
                )
            else:
                # If drop_params was already True and we still got UnsupportedParamsError,
                # fall through to regular error handling
                warnings.warn(
                    f"Exception occurred while calling LLM API with drop_params=True: {e}",
                    stacklevel=2,
                )
                if self.fallback_llm:
                    logger.info(
                        "Call will be retried if retry params provided and/or a fallback LLM is configured."
                    )
                else:
                    n_retries = max(
                        num_retries_failed_request, max_retries_failed_request
                    )
                    if raise_exception_on_llm_api_error:
                        raise LLMAPIError(
                            "Exception occurred while calling LLM API",
                            retry_count=n_retries,
                            original_error=e,
                        ) from e
                    else:
                        warnings.warn(
                            f"Exception occurred while calling LLM API with drop_params=True: {e}"
                            + f" ({n_retries} retries).",
                            stacklevel=2,
                        )
        except Exception as e:
            # e.g. rate limit error
            logger.error(f"Exception occurred while calling LLM API: {e}")
            if self.fallback_llm:
                logger.info(
                    "Call will be retried if retry params provided and/or a fallback LLM is configured."
                )
            else:
                n_retries = max(num_retries_failed_request, max_retries_failed_request)
                if raise_exception_on_llm_api_error:
                    raise LLMAPIError(
                        "Exception occurred while calling LLM API",
                        retry_count=n_retries,
                        original_error=e,
                    ) from e
                else:
                    warnings.warn(
                        f"Exception occurred while calling LLM API: {e}"
                        + f" ({n_retries} retries).",
                        stacklevel=2,
                    )

        usage.calls.append(llm_call_obj)  # record the call details (call unfinished)
        return None, usage

    def _set_prompts(self) -> None:
        """
        Sets up prompt templates for various extraction tasks.

        :return: None
        """

        # Templates with placeholders
        # Extraction
        # Safe cast: _get_template with default params returns Template, not str
        self._extract_aspect_items_prompt = cast(
            Template, _get_template("extract_aspect_items")
        )
        self._extract_concept_items_prompt = cast(
            Template, _get_template("extract_concept_items")
        )

    def _set_capabilities(self) -> None:
        """
        Sets the capabilities of the LLM based on litellm.supports_*()
        functions.

        :return: None
        """
        self._supports_vision = litellm.supports_vision(self.model)  # type: ignore[attr-defined]
        self._supports_reasoning = litellm.supports_reasoning(self.model)  # type: ignore[attr-defined]

    def _set_system_message(self) -> None:
        """
        Sets the default system message for the LLM.
        :return: None
        """
        # _get_template returns a Template object when template_extension == "j2"
        self.system_message = _get_template(
            "default_system_message",
            template_type="system",
            template_extension="j2",
        ).render({"output_language": self.output_language})  # type: ignore[attr-defined]

    def _get_raw_usage(self) -> _LLMUsage:
        """
        Retrieves the raw usage information of the LLM.

        :return: _LLMUsage object containing usage data for the LLM.
        """

        return self._usage

    def _get_raw_cost(self) -> _LLMCost:
        """
        Retrieves the raw cost information of the LLM.

        :return: _LLMCost object containing cost data for the LLM.
        """

        if self.pricing_details is None:
            logger.info(
                f"No pricing details provided for the LLM `{self.model}` "
                f"with role `{self.role}`. Costs for this LLM were not calculated."
            )

        return self._cost

    def get_usage(self) -> list[_LLMUsageOutputContainer]:
        """
        Retrieves the usage information of the LLM and its fallback LLM if configured.

        This method collects token usage statistics for the current LLM instance and its
        fallback LLM (if configured), providing insights into API consumption.

        :return: A list of usage statistics containers for the LLM and its fallback.
        :rtype: list[_LLMUsageOutputContainer]
        """

        # Safe cast: _get_usage_or_cost with retrieval_type="usage"
        # returns only list of _LLMUsageOutputContainer
        return cast(
            list["_LLMUsageOutputContainer"],
            self._get_usage_or_cost(retrieval_type="usage"),
        )

    def get_cost(self) -> list[_LLMCostOutputContainer]:
        """
        Retrieves the accumulated cost information of the LLM and its fallback LLM if configured.

        This method collects cost statistics for the current LLM instance and its
        fallback LLM (if configured), providing insights into API usage expenses.

        :return: A list of cost statistics containers for the LLM and its fallback.
        :rtype: list[_LLMCostOutputContainer]
        """

        # Safe cast: _get_usage_or_cost with retrieval_type="cost"
        # returns only list of _LLMCostOutputContainer
        return cast(
            list["_LLMCostOutputContainer"],
            self._get_usage_or_cost(retrieval_type="cost"),
        )

    def reset_usage_and_cost(self) -> None:
        """
        Resets the usage and cost statistics for the LLM and its fallback LLM (if configured).

        This method clears accumulated usage and cost data, which is useful when processing
        multiple documents sequentially and tracking metrics for each document separately.

        :return: None
        """

        for llm in [self, self.fallback_llm]:
            if llm:
                llm._usage = _LLMUsage()
                llm._cost = _LLMCost()

    def _increment_cost(self, usage: _LLMUsage) -> None:
        """
        Calculates and increments the self._cost attribute values based on
        the additional usage details provided. Relevant only if the user has
        provided pricing details for the LLM.

        :param usage: _LLMUsage instance containing usage information on
                      additional number of input and output tokens processed.
        :type usage: _LLMUsage

        :return: None
        """

        if self.pricing_details:
            mil_dec = Decimal("1000000")
            cost_input = (Decimal(str(usage.input)) / mil_dec) * Decimal(
                str(self.pricing_details.input_per_1m_tokens)
            )
            cost_output = (Decimal(str(usage.output)) / mil_dec) * Decimal(
                str(self.pricing_details.output_per_1m_tokens)
            )
            cost_total = cost_input + cost_output

            self._cost.input += cost_input
            self._cost.output += cost_output
            self._cost.total += cost_total

            round_dec = Decimal("0.00001")
            self._cost.input = self._cost.input.quantize(
                round_dec, rounding=ROUND_HALF_UP
            )
            self._cost.output = self._cost.output.quantize(
                round_dec, rounding=ROUND_HALF_UP
            )
            self._cost.total = self._cost.total.quantize(
                round_dec, rounding=ROUND_HALF_UP
            )

    async def _update_usage_and_cost(
        self, result: tuple[Any, _LLMUsage] | None
    ) -> None:
        """
        Updates the LLM usage and cost details based on the given processing result.
        This method  modifies the LLM instance's usage statistics and increments the associated
        cost if pricing details are specified.

        :param result: A tuple containing an optional value and usage data. The usage
            data is used to update the instance's input and output usage, as well as
            the total cost. If the result is None, the method does nothing.
        :type result: tuple[Any, _LLMUsage]
        :return: None
        """
        async with self._async_lock:
            if result is None:
                return
            new_usage = result[1]
            # Pricing data
            if self.pricing_details:
                self._usage.input += new_usage.input
                self._usage.output += new_usage.output
                self._increment_cost(new_usage)
            # Calls data
            self._usage.calls += new_usage.calls
