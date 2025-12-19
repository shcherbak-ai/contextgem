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
     - ``str``
     - ``"extractor_text"``
     - Role type for the LLM. Values: ``"extractor_text"``, ``"reasoner_text"``, ``"extractor_vision"``, ``"reasoner_vision"``, ``"extractor_multimodal"``, ``"reasoner_multimodal"``. The role parameter is an abstraction that can be explicitly assigned to extraction components (aspects and concepts) in the pipeline. ContextGem then routes extraction tasks based on these assigned roles, matching components with LLMs of the same role. This allows you to structure your pipeline with different models for different tasks (e.g., using simpler models for basic extractions and more powerful models for complex reasoning). For more details, see :ref:`llm-roles-label`.
   * - ``system_message``
     - ``str | None``
     - ``None``
     - If not provided (or set to None), ContextGem automatically sets a default system message optimized for extraction tasks, rendered based on the configured ``output_language``. This default system message template can be found `here <https://github.com/shcherbak-ai/contextgem/blob/dev/contextgem/internal/system/default_system_message.j2>`_ in the source code. Note that for certain models (such as OpenAI's o1-preview), system messages are not supported and will be ignored. Overriding this is typically only necessary for advanced use cases, such as custom priming or non-extraction tasks. For simple chat interactions, consider setting ``system_message=''`` to disable the default entirely (meaning no system message will be sent).
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
     - Reasoning effort for reasoning (CoT-capable) models. Values: ``"minimal"`` (gpt-5 models only), ``"low"``, ``"medium"``, ``"high"``, ``"xhigh"`` (gpt-5.2 models only).
   * - ``timeout``
     - ``int``
     - ``120``
     - Timeout in seconds for LLM API calls.
   * - ``num_retries_failed_request``
     - ``int``
     - ``3``
     - Number of retries when LLM request fails.
   * - ``max_retries_failed_request``
     - ``int``
     - ``0``
     - LLM provider-specific retry count for failed requests.
   * - ``max_retries_invalid_data``
     - ``int``
     - ``3``
     - Number of retries when LLM request succeeds but returns invalid data (JSON parsing and validation fails).
   * - ``pricing_details``
     - ``LLMPricing | None``
     - ``None``
     - :class:`~contextgem.public.data_models.LLMPricing` object with pricing details for cost calculation.
   * - ``auto_pricing``
     - ``bool``
     - ``False``
     - Enable automatic cost calculation using ``genai-prices`` based on the configured ``model``. Mutually exclusive with ``pricing_details``. Not supported for local models (e.g., ``ollama/``, ``ollama_chat/``, ``lm_studio/``).
   * - ``auto_pricing_refresh``
     - ``bool``
     - ``False``
     - When ``auto_pricing`` is enabled, allow ``genai-prices`` to auto-refresh its cached pricing data at runtime.
   * - ``is_fallback``
     - ``bool``
     - ``False``
     - Indicates whether the LLM is a fallback model. Fallback LLMs are optionally assigned to the primary LLM instance and are used when the primary LLM fails.
   * - ``fallback_llm``
     - ``DocumentLLM | None``
     - ``None``
     - :class:`~contextgem.public.llms.DocumentLLM` to use as fallback if current one fails. Must have the same role as the primary LLM.
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
     - ``0.3``
     - Nucleus sampling value (0.0 to 1.0) controlling output focus/randomness. Lower values make output more deterministic, higher values produce more diverse outputs.
   * - ``seed``
     - ``int | None``
     - ``None``
     - Seed for random number generation to help produce more consistent outputs across multiple runs. When set to a specific integer value, the LLM will attempt to use this seed for sampling operations. However, deterministic output is still not guaranteed even with the same seed, as other factors may influence the model's response.
   * - ``tools``
     - ``list[dict] | None``
     - ``None``
     - OpenAI-compatible tool schema used only for chat via ``DocumentLLM.chat(...)``/``.chat_async(...)``. Each tool must have a registered Python handler decorated with ``@register_tool`` and available in scope when creating the LLM. Handlers must return a string; for structured data, serialize it (e.g., with ``json.dumps``) before returning. Ignored by extraction methods. For more details, see :ref:`llm-chat-with-tools-label`.
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
     - Relevant when concurrency is enabled for extraction tasks. Controls frequency of async LLM API requests for concurrent tasks. Defaults to allowing 3 acquisitions per 10-second period to prevent rate limit issues. See `aiolimiter documentation <https://github.com/mjpieters/aiolimiter>`_ for AsyncLimiter configuration details. See :doc:`../optimizations/optimization_speed` for an example of how to easily set up concurrency for extraction.

.. warning::
   **Auto-pricing accuracy**

   When using ``auto_pricing=True``, cost estimates are approximate. These prices will not be 100% accurate. The price data cannot be exactly correct because model providers do not provide exact price information for their APIs in a format which can be reliably processed. See Pydantic's `genai-prices <https://github.com/pydantic/genai-prices>`_ for more details.

💡 Advanced Configuration Examples
------------------------------------

🔄 Configuring a Fallback LLM
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can set up a fallback LLM that will be used if the primary LLM fails:

.. literalinclude:: ../../../dev/usage_examples/docs/llm_config/fallback_llm.py
   :language: python
   :caption: Configuring a fallback LLM


💰 Setting Up Cost Tracking
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can configure pricing parameters to track costs:

.. literalinclude:: ../../../dev/usage_examples/docs/llm_config/cost_tracking.py
   :language: python
   :caption: Setting up LLM cost tracking


🧠 Using Model-Specific Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For reasoning (CoT-capable) models (such as OpenAI's o1/o3/o4), you can set reasoning-specific parameters:

.. literalinclude:: ../../../dev/usage_examples/docs/llm_config/o1_o4.py
   :language: python
   :caption: Using model-specific parameters


⚙️ Explicit Capability Declaration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Model vision capabilities are automatically detected using ``litellm.supports_vision()``. If this function does not correctly identify your model's capabilities, ContextGem will typically issue a warning, and you can explicitly declare the capability by setting ``_supports_vision=True`` on the LLM instance:

.. code-block:: python

   from contextgem import DocumentLLM

   # Example: Explicitly declare vision capability
   # Warning will be issued if automatic vision capability detection fails
   llm = DocumentLLM(
       model="some_provider/custom_vision_model",
       api_base="http://localhost:3456/v1",
       role="extractor_vision"
   )
   # Declare capability if automatic detection fails (warning was issued)
   llm._supports_vision = True

.. warning::
   Explicit capability declarations should only be used when automatic capability detection fails. Incorrectly setting this flag may lead to unexpected behavior or API errors.


🤖🤖 LLM Groups
-----------------

For complex document processing, you can organize multiple LLMs with different roles into a group:

.. literalinclude:: ../../../dev/usage_examples/docs/llm_config/llm_group.py
   :language: python
   :caption: Using LLM group

See a practical example of using an LLM group in :ref:`multi-llm-pipeline-label`.


📊 Accessing Usage and Cost Statistics
----------------------------------------

You can track input/output token usage and costs:

.. literalinclude:: ../../../dev/usage_examples/docs/llm_config/tracking_usage_and_cost.py
   :language: python
   :caption: Tracking usage and cost

The usage statistics include not only token counts but also detailed information about each individual call made to the LLM. You can access the call history, including prompts, responses, and timestamps:

.. literalinclude:: ../../../dev/usage_examples/docs/llm_config/detailed_usage.py
   :language: python
   :caption: Accessing detailed usage information
