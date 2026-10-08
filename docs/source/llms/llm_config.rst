.. 
   ContextGem
   
   Copyright 2025 Shcherbak AI AS. All rights reserved. Developed by Sergii Shcherbak.
   
   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at
   
       http://www.apache.org/licenses/LICENSE-2.0
   
   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

:og:description: ContextGem: Configuring LLM(s)


Configuring LLM(s)
===================

This guide explains how to configure :class:`~contextgem.public.llms.DocumentLLM` instances to process documents using various LLM providers.
ContextGem uses LiteLLM under the hood, providing uniform access to a wide range of models. For more information on supported LLMs, see :doc:`supported_llms`.


🚀 Basic Configuration
------------------------

The minimum configuration for a cloud-based LLM requires the ``model`` parameter and an ``api_key``:

.. literalinclude:: ../../../dev/usage_examples/docs/llms/llm_init/llm_api.py
   :language: python
   :caption: Using a cloud-based LLM

For local models, usually you need to specify the ``api_base`` instead of the API key:

.. literalinclude:: ../../../dev/usage_examples/docs/llms/llm_init/llm_local.py
   :language: python
   :caption: Using a local LLM

.. note::
   **LM Studio Connection Error**: If you encounter a connection error (``litellm.APIError: APIError: Lm_studioException - Connection error``) when using LM Studio, check that you have provided a dummy API key. While API keys are usually not expected for local models, this is a specific case where LM Studio requires one:

   .. literalinclude:: ../../../dev/usage_examples/docs/llms/llm_init/lm_studio_connection_error_fix.py
      :language: python
      :caption: LM Studio with dummy API key

   This is a known issue with calling LM Studio API in litellm: https://github.com/openai/openai-python/issues/961


📝 Configuration Parameters
-----------------------------

The :class:`~contextgem.public.llms.DocumentLLM` class accepts the following parameters:

.. list-table::
   :header-rows: 1
   :widths: 20 15 15 50

   * - Parameter
     - Type
     - Default Value
     - Description
   * - ``model``
     - ``str``
     - (Required)
     - Model identifier in format ``<provider>/<model_name>``. See `LiteLLM Providers <https://docs.litellm.ai/docs/providers>`_ for all supported providers.
   * - ``api_key``
     - ``str | None``
     - ``None``
     - API key for authentication. Required for most cloud providers but not for local models.
   * - ``api_base``
     - ``str | None``
     - ``None``
     - Base URL of the API endpoint. Required for local models and some cloud providers (e.g. Azure OpenAI).
   * - ``deployment_id``
     - ``str | None``
     - ``None``
     - Deployment ID for the model. Primarily used with Azure OpenAI.
   * - ``api_version``
     - ``str | None``
     - ``None``
     - API version. Primarily used with Azure OpenAI.
   * - ``role``
     - ``str | None``
     - ``None``
     - Free-form role label of the LLM (up to 64 printable characters), used for routing in a model group. The same label can be explicitly assigned to extraction components (aspects and concepts) in the pipeline, using their ``model_role`` parameter. ContextGem then routes extraction tasks based on these assigned roles, matching components with the LLM of the same role, while the components without a role are routed to the LLM without a role. This allows you to structure your pipeline with different models for different tasks (e.g., using simpler models for basic extractions and more powerful models for complex reasoning). The role implies no capabilities of the model, and is ignored when the LLM is used on its own (not in a group), as a single LLM extracts all the aspects and concepts it is given. For more details, see :ref:`model-roles-label`.
   * - ``system_message``
     - ``str | None``
     - ``None``
     - If not provided (or set to None), ContextGem automatically sets a default system message optimized for extraction tasks, rendered based on the configured ``output_language``. This default system message template can be found `here <https://github.com/shcherbak-ai/contextgem/blob/dev/contextgem/internal/system/default_system_message.j2>`_ in the source code. Overriding this is typically only necessary for advanced use cases, such as custom priming or non-extraction tasks. For simple chat interactions, consider setting ``system_message=''`` to disable the default entirely (meaning no system message will be sent).
   * - ``max_tokens``
     - ``int``
     - ``4096``
     - Maximum tokens in the generated response (applicable to most models).
   * - ``max_completion_tokens``
     - ``int``
     - ``16000``
     - Maximum tokens for output completions in reasoning (CoT-capable) models.
   * - ``reasoning_effort``
     - ``str | None``
     - ``None``
     - Reasoning effort for reasoning (CoT-capable) models. Values: ``"none"``, ``"minimal"``, ``"low"``, ``"medium"``, ``"high"``, ``"xhigh"``, ``"max"``. Support for specific values varies by model and provider; unsupported combinations are rejected by litellm or the provider API at request time. ``"none"`` explicitly disables reasoning on models that support this value, which differs from the default ``None`` (parameter not sent, provider default applies).
   * - ``timeout``
     - ``int``
     - ``120``
     - Timeout in seconds for LLM API calls.
   * - ``num_retries_failed_request``
     - ``int``
     - ``3``
     - Number of retries when LLM request fails with a transient error (request timeout, rate limit, server or connection error). Retries are performed with exponential backoff and jitter. Non-transient errors (e.g. invalid request, failed authentication, exceeded context window) are not retried. See :ref:`llm-retries-label`.
   * - ``max_retries_failed_request``
     - ``int``
     - ``0``
     - LLM provider-specific retry count for failed requests, i.e. retries performed by the provider's SDK, where supported. Applies to the first attempt of a request only.
   * - ``max_retries_invalid_data``
     - ``int``
     - ``3``
     - Number of retries when LLM request succeeds but returns invalid data (JSON parsing and validation fails). Each retry is a new request.
   * - ``pricing_details``
     - ``ModelPricing | None``
     - ``None``
     - :class:`~contextgem.public.data_models.ModelPricing` object with pricing details for cost calculation: the prices of input and output tokens and, optionally, of the input tokens read from and written to the prompt cache of the provider.
   * - ``auto_pricing``
     - ``bool``
     - ``False``
     - Enable automatic cost calculation using ``genai-prices`` based on the configured ``model``. Mutually exclusive with ``pricing_details``. Not supported for local models (e.g., ``ollama/``, ``ollama_chat/``, ``lm_studio/``).
   * - ``auto_pricing_refresh``
     - ``bool``
     - ``False``
     - When ``auto_pricing`` is enabled, allow ``genai-prices`` to auto-refresh its cached pricing data at runtime. Enable it if the pricing data of a recently released model is not found, as the pricing data bundled with ``genai-prices`` may not include such a model yet.
   * - ``is_fallback``
     - ``bool``
     - ``False``
     - Indicates whether the LLM is a fallback model. Fallback LLMs are optionally assigned to the primary LLM instance and are used when the primary LLM fails.
   * - ``fallback_models``
     - ``list[DocumentLLM]``
     - ``[]``
     - :class:`~contextgem.public.llms.DocumentLLM` instances to use as fallbacks if the current one fails. Fallback LLMs are tried in the given order until one of them succeeds. Each must have ``is_fallback=True`` and the same role as the primary LLM.
   * - ``output_language``
     - ``str``
     - ``"en"``
     - Language for output text. Values: ``"en"`` or ``"adapt"`` (adapts to document language). Setting value to ``"adapt"`` ensures that the text output (e.g. justifications, conclusions, explanations) is in the same language as the document. This is particularly useful when working with non-English documents. For example, if you're extracting anomalies from a contract in Spanish, setting ``output_language="adapt"`` ensures that anomaly justifications are also in Spanish, making them immediately understandable by local end users reviewing the document. This parameter applies only when the default system message is used.
   * - ``temperature``
     - ``float | None``
     - ``0.3``
     - Sampling temperature (0.0 to 1.0) controlling response creativity. Lower values produce more predictable outputs, higher values generate more varied responses.
   * - ``top_p``
     - ``float | None``
     - ``None``
     - Nucleus sampling value (0.0 to 1.0) controlling output focus/randomness. Lower values make output more deterministic, higher values produce more diverse outputs. Defaults to ``None`` so only ``temperature`` is sent by default. Some providers (e.g. Claude 4.x) reject requests that set both ``temperature`` and ``top_p``; set one of them to ``None`` when using those models.
   * - ``seed``
     - ``int | None``
     - ``None``
     - Seed for random number generation to help produce more consistent outputs across multiple runs. When set to a specific integer value, the LLM will attempt to use this seed for sampling operations. However, deterministic output is still not guaranteed even with the same seed, as other factors may influence the model's response.
   * - ``tools``
     - ``list[Callable] | None``
     - ``None``
     - Tools for chat via ``DocumentLLM.chat(...)``/``.chat_async(...)``: functions decorated with ``@register_tool``. The tool schemas sent to the LLM are generated from the names, docstrings and type hints of the functions. Handlers must return a string; for structured data, serialize it (e.g., with ``json.dumps``) before returning. Ignored by extraction methods. For more details, see :ref:`llm-chat-with-tools-label`.
   * - ``tool_choice``
     - ``str | dict | None``
     - ``None``
     - Tool choice control passed through to the provider during chat. Ignored by extraction methods.
   * - ``parallel_tool_calls``
     - ``bool | None``
     - ``None``
     - Enable parallel tool calls during chat tool usage, if supported by the model/provider. Ignored by extraction methods.
   * - ``tool_max_rounds``
     - ``int``
     - ``10``
     - Safety limit on the number of tool-execution rounds per chat request to prevent infinite loops.
   * - ``async_limiter``
     - ``AsyncLimiter``
     - ``AsyncLimiter(3, 10)``
     - Relevant when concurrency is enabled for extraction tasks. Controls frequency of async LLM API requests for concurrent tasks. Defaults to allowing 3 acquisitions per 10-second period to prevent rate limit issues. Requests to a fallback LLM are controlled by the ``async_limiter`` of that fallback LLM. See `aiolimiter documentation <https://github.com/mjpieters/aiolimiter>`_ for AsyncLimiter configuration details. See :doc:`../optimizations/optimization_speed` for an example of how to easily set up concurrency for extraction.

.. warning::
   **Auto-pricing accuracy**

   When using ``auto_pricing=True``, cost estimates are approximate. These prices will not be 100% accurate. The price data cannot be exactly correct because model providers do not provide exact price information for their APIs in a format which can be reliably processed. See Pydantic's `genai-prices <https://github.com/pydantic/genai-prices>`_ for more details.

💡 Advanced Configuration Examples
------------------------------------

🔄 Configuring Fallback LLMs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can set up one or more fallback LLMs that will be used if the primary LLM fails. Fallback LLMs are tried in the order they are listed in ``fallback_models``, until one of them succeeds:

.. literalinclude:: ../../../dev/usage_examples/docs/llm_config/fallback_models.py
   :language: python
   :caption: Configuring fallback LLMs

Each fallback LLM must be created with ``is_fallback=True``, have the same ``role`` and ``output_language`` as the primary LLM, and cannot have fallback LLMs of its own. A fallback LLM must also differ from the primary LLM, and from the other fallback LLMs, in at least one configuration parameter, e.g. the model, the API key or the API base, as an exact copy would only repeat the same request. This is checked when the fallback LLMs are set, i.e. when the primary LLM is created with them or when ``fallback_models`` is assigned. If the primary LLM is used to extract concepts from images, or is passed images in a chat, each of its fallback LLMs must support vision as well, which is checked before any requests are sent. Usage and cost are tracked separately for the primary LLM and for each of its fallback LLMs.

The next fallback LLM is tried when the current LLM fails for good, i.e. when its request fails with a non-transient error, when the retries of its request are exhausted, or, during extraction, when it keeps returning invalid data. See :ref:`llm-retries-label` for details.

.. note::
   Each LLM applies its own retry settings and timeout before the next fallback LLM is tried, so the more fallback LLMs you configure, the longer a request can take to fail when all of the LLMs are unavailable.


.. _llm-retries-label:

🔁 How Failed Requests Are Retried
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Retries are configured per LLM, which also applies to each fallback LLM, and depend on why the request failed:

- **Transient API errors** (request timeout, rate limit, server or connection error) are retried up to ``num_retries_failed_request`` times. The delay between the retries grows exponentially (about 1, 2, 4, 8 seconds, capped at 10 seconds) and is randomly shortened by up to a half (jitter), so that concurrent requests that failed together are not all retried at the same moment.
- **Non-transient API errors** (e.g. invalid request, failed authentication, exceeded context window, content policy violation) are not retried, as the same request would fail again in the same way.
- **Invalid data** returned by the LLM during extraction (the response cannot be parsed or validated) is requested again up to ``max_retries_invalid_data`` times, without a delay.

When a request to an LLM fails for good, it is not sent to that LLM again: the next fallback LLM is tried, if configured. If there are no more fallback LLMs to try, ``ModelAPIError`` is raised for a failed API request, or ``ModelExtractionError`` for invalid data. During extraction, a warning is issued instead if ``raise_exception_on_extraction_error=False`` is passed to the extraction method.

.. note::
   ``max_retries_failed_request`` sets the number of retries performed by the LLM provider's SDK, where supported (e.g. for OpenAI), in addition to the retries described above. It applies to the first attempt of a request only.


💰 Setting Up Cost Tracking
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can configure pricing parameters to track costs:

.. literalinclude:: ../../../dev/usage_examples/docs/llm_config/cost_tracking.py
   :language: python
   :caption: Setting up LLM cost tracking


.. _llm-cached-tokens-label:

💾 How Cached Tokens Are Priced
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Some LLM providers cache the beginning of a prompt and reuse it across requests (prompt caching), in which case they report how many input tokens of a request were read from the cache, and some of them also how many were written to it. Tokens read from the cache are usually priced lower than regular input tokens, while tokens written to the cache may be priced higher. ContextGem does not configure prompt caching, but takes such tokens into account in the calculated costs whenever the provider reports them:

- With ``auto_pricing=True``, they are priced at the cache prices of the model, if ``genai-prices`` has such prices, and as regular input tokens otherwise.
- With ``pricing_details``, they are priced at the optional ``cache_read_per_1m_tokens`` and ``cache_write_per_1m_tokens`` of :class:`~contextgem.public.data_models.ModelPricing`. If a cache price is not set, the corresponding tokens are priced as regular input tokens, i.e. at ``input_per_1m_tokens``.

.. note::
   Tokens written to the cache are priced at the standard cache write price of the model. A longer cache lifetime that a provider may offer at a higher price (e.g. 1-hour cache writes of Anthropic models) is not priced separately. Such a lifetime has to be requested explicitly, which ContextGem does not do.


🧠 Using Model-Specific Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For reasoning (CoT-capable) models (such as OpenAI's GPT-5 and GPT-6 models), you can set reasoning-specific parameters:

.. literalinclude:: ../../../dev/usage_examples/docs/llm_config/reasoning_params.py
   :language: python
   :caption: Using model-specific parameters


⚙️ Explicit Capability Declaration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Model vision capabilities are automatically detected using ``litellm.supports_vision()``. If this function does not correctly identify your model's capabilities, an error will be raised when the model is to be sent images (e.g. when extracting concepts with the ``"image"`` or ``"multimodal"`` modality), and you can explicitly declare the capability by setting ``_supports_vision=True`` on the LLM instance:

.. code-block:: python

   from contextgem import DocumentLLM

   # Example: Explicitly declare vision capability
   llm = DocumentLLM(
       model="some_provider/custom_vision_model",
       api_base="http://localhost:3456/v1",
   )
   # Declare capability if automatic detection fails
   # (an error is raised when the model is to be sent images)
   llm._supports_vision = True

.. warning::
   Explicit capability declarations should only be used when automatic capability detection fails. Incorrectly setting this flag may lead to unexpected behavior or API errors.


🤖🤖 Model Groups
-----------------

For complex document processing, you can organize multiple LLMs with different roles into a group:

.. literalinclude:: ../../../dev/usage_examples/docs/llm_config/model_group.py
   :language: python
   :caption: Using model group

Each LLM in the group must have a unique ``role``, and only one LLM in the group can have no role. An aspect/concept is routed to the LLM with the ``role`` equal to the ``model_role`` of the aspect/concept, and an aspect/concept without ``model_role`` is routed to the LLM without ``role``. If an aspect/concept cannot be routed to any LLM in the group, an error is raised. For more details, see :ref:`model-roles-label`.

See a practical example of using a model group in :ref:`multi-llm-pipeline-label`.


📊 Accessing Usage and Cost Statistics
----------------------------------------

You can track input/output token usage and costs:

.. literalinclude:: ../../../dev/usage_examples/docs/llm_config/tracking_usage_and_cost.py
   :language: python
   :caption: Tracking usage and cost

Token usage is tracked for every LLM, while costs are calculated only if pricing is configured for the LLM, i.e. ``pricing_details`` or ``auto_pricing`` is set.

The ``input`` token count includes the tokens that the LLM provider read from, or wrote to, its prompt cache. Such tokens are also counted separately, as ``cache_read`` and ``cache_write``, which are zero if the provider does not report them. Accordingly, the ``input`` cost includes the cost of such tokens (see :ref:`llm-cached-tokens-label`).

The usage statistics include not only token counts but also detailed information about each individual call made to the LLM. You can access the call history, including prompts, responses, and timestamps:

.. literalinclude:: ../../../dev/usage_examples/docs/llm_config/detailed_usage.py
   :language: python
   :caption: Accessing detailed usage information
