.. meta::
   :description: How to configure a DocumentLLM: the model and its credentials, fallback LLMs, retries, output token limits, sampling parameters, reasoning effort, model groups, and cost and usage tracking.


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
     - Free-form role label of the LLM (up to 64 printable characters), used for routing in a model group. The same label can be explicitly assigned to extraction components (aspects and concepts) in the pipeline, using their ``model_role`` parameter. ContextGem then routes extraction tasks based on these assigned roles, matching components with the model of the same role, while the components without a role are routed to the model without a role. This allows you to structure your pipeline with different models for different tasks (e.g., using simpler models for basic extractions and more powerful models for complex reasoning). The role implies no capabilities of the model, and is ignored when the LLM is used on its own (not in a group), as a single LLM extracts all the aspects and concepts it is given. For more details, see :ref:`model-roles-label`.
   * - ``system_message``
     - ``str | None``
     - ``None``
     - If not provided (or set to None), ContextGem automatically sets a default system message optimized for extraction tasks, rendered based on the configured ``output_language``. The default system message follows the output language: when ``output_language`` is changed after the LLM is created, the default system message is rendered for the new one, while a custom system message is kept as it is. Assigning ``None`` to ``system_message`` restores the default system message. This default system message template can be found `here <https://github.com/shcherbak-ai/contextgem/blob/main/contextgem/internal/system/default_system_message.j2>`_ in the source code. Overriding this is typically only necessary for advanced use cases, such as custom priming or non-extraction tasks. For simple chat interactions, consider setting ``system_message=''`` to disable the default entirely (meaning no system message will be sent).
   * - ``max_tokens``
     - ``int | None``
     - ``None``
     - Maximum output tokens allowed per request. For reasoning (CoT-capable) models, the reasoning tokens count towards this limit. If not set (default), no limit is sent with the requests, and the default of the model applies, which for most providers is the maximum output of the model. A value above the maximum output of the model, as known to ``litellm``, raises ``ValueError`` before a request is sent to the model: in a chat, the fallback LLMs are checked before any request is sent, and in extraction, a fallback LLM is checked when it is tried. See :ref:`llm-max-output-tokens-label` for how a response cut off at the limit is handled.
   * - ``reasoning_effort``
     - ``str | None``
     - ``None``
     - Reasoning effort for reasoning (CoT-capable) models. Values: ``"none"``, ``"minimal"``, ``"low"``, ``"medium"``, ``"high"``, ``"xhigh"``, ``"max"``. The parameter is sent only to the models detected as supporting reasoning. Support for specific values varies by model and provider: if ``litellm`` reports a value as not supported for the model, the request is sent without the parameter, with a warning logged, as for the sampling parameters (see :ref:`llm-sampling-params-label`), while a value that is rejected otherwise, e.g. by the provider API, fails the request. ``"none"`` explicitly disables reasoning on models that support this value, which differs from the default ``None`` (parameter not sent, provider default applies).
   * - ``timeout``
     - ``int``
     - ``600``
     - Timeout in seconds for LLM API calls. Responses are not streamed, so the whole response, including the reasoning of the model, has to be received within the timeout. A request that times out is retried as a transient error (see :ref:`llm-retries-label`).
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
     - Number of retries when LLM request succeeds but returns invalid data (JSON parsing and validation fails). Each retry is a new request, i.e. the LLM is asked for the same data up to ``1 + max_retries_invalid_data`` times. A fallback LLM applies its own setting in the same way: it is asked once, and then again up to its ``max_retries_invalid_data`` times. See :ref:`llm-retries-label`.
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
     - When ``auto_pricing`` is enabled, allow ``genai-prices`` to auto-refresh its cached pricing data at runtime. Enable it if the pricing data of a recently released model is not found, as the pricing data bundled with ``genai-prices`` may not include such a model yet. The data is fetched from ``raw.githubusercontent.com``.
   * - ``is_fallback``
     - ``bool``
     - ``False``
     - Indicates whether the LLM is a fallback model. Fallback LLMs are optionally assigned to the primary model instance and are used when the primary model fails.
   * - ``fallback_models``
     - ``list[DocumentLLM | DocumentDecisionModel]``
     - ``[]``
     - Models to use as fallbacks if the current one fails. Fallback models are tried in the given order until one of them succeeds. Each fallback LLM must have ``is_fallback=True`` and the same role, output language, system message, tools, ``tool_choice`` and ``parallel_tool_calls`` as the primary LLM, and, if it has tools, the same support for tool calls (see :ref:`llm-tools-fallback-label`). A fallback can also be a decision model, if it can extract everything that the LLM is to extract (see :ref:`decision-models-fallbacks-label`).
   * - ``output_language``
     - ``str``
     - ``"en"``
     - Language for output text. Values: ``"en"`` or ``"adapt"`` (adapts to document language). Setting value to ``"adapt"`` ensures that the text output (e.g. justifications, conclusions, explanations) is in the same language as the document. This is particularly useful when working with non-English documents. For example, if you're extracting anomalies from a contract in Spanish, setting ``output_language="adapt"`` ensures that anomaly justifications are also in Spanish, making them immediately understandable by local end users reviewing the document. The parameter applies to the extraction prompts, and to the default system message. With a custom system message, it applies to the extraction prompts only, i.e. it has no effect in a chat, and a warning says so.
   * - ``temperature``
     - ``float | None``
     - ``None``
     - Sampling temperature (0.0 or higher; the upper limit depends on the model) controlling response creativity. Lower values produce more predictable outputs, higher values generate more varied responses. If not set (default), the parameter is not sent, and the default of the model applies. See :ref:`llm-sampling-params-label` for when to set it.
   * - ``top_p``
     - ``float | None``
     - ``None``
     - Nucleus sampling value (0.0 to 1.0) controlling output focus/randomness. Lower values make output more deterministic, higher values produce more diverse outputs. If not set (default), the parameter is not sent. Some models (e.g. Anthropic's Claude 4.x) reject requests that set both ``temperature`` and ``top_p``; set only one of them when using those models.
   * - ``seed``
     - ``int | None``
     - ``None``
     - Seed for random number generation to help produce more consistent outputs across multiple runs. When set to a specific integer value, the LLM will attempt to use this seed for sampling operations. However, deterministic output is still not guaranteed even with the same seed, as other factors may influence the model's response. If not set (default), the parameter is not sent. Support depends on the LLM provider: a seed is mostly of use with local and open-weight models, while some providers (e.g. Anthropic) do not support it (see :ref:`llm-sampling-params-label`).
   * - ``tools``
     - ``Sequence[Callable] | None``
     - ``None``
     - Tools for chat via ``DocumentLLM.chat(...)``/``.chat_async(...)``: functions decorated with ``@register_tool``. The tool schemas sent to the LLM are generated from the names, docstrings and type hints of the functions. Handlers must return a non-empty string; for structured data, serialize it (e.g., with ``json.dumps``) before returning. Ignored by extraction methods. For more details, see :doc:`chat_with_tools`.
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
     - ``AsyncLimiter(10, 10)``
     - Relevant when concurrency is enabled for extraction tasks. Controls frequency of async LLM API requests for concurrent tasks. Defaults to allowing 10 acquisitions per 10-second period to prevent rate limit issues. Adjust it to the rate limits of your LLM provider account, e.g. lower it if you send long documents on a low usage tier. The rate applies to all the requests to the LLM that are sent concurrently, i.e. the extraction calls that run at the same time, e.g. for the concepts of different aspects, share it. Requests to a fallback model are controlled by the ``async_limiter`` of that fallback model. See `aiolimiter documentation <https://github.com/mjpieters/aiolimiter>`_ for AsyncLimiter configuration details. See :doc:`../optimizations/optimization_speed` for an example of how to easily set up concurrency for extraction.

.. warning::
   **Auto-pricing accuracy**

   When using ``auto_pricing=True``, cost estimates are approximate. These prices will not be 100% accurate. The price data cannot be exactly correct because model providers do not provide exact price information for their APIs in a format which can be reliably processed. See Pydantic's `genai-prices <https://github.com/pydantic/genai-prices>`_ for more details.

💡 Advanced Configuration Examples
------------------------------------

.. _llm-fallback-llms-label:

🔄 Configuring Fallback LLMs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can set up one or more fallback LLMs that will be used if the primary LLM fails. Fallback LLMs are tried in the order they are listed in ``fallback_models``, until one of them succeeds:

.. literalinclude:: ../../../dev/usage_examples/docs/llm_config/fallback_models.py
   :language: python
   :caption: Configuring fallback LLMs

Each fallback LLM must be created with ``is_fallback=True``, have the same ``role``, ``output_language`` and ``system_message`` as the primary LLM, and cannot have fallback models of its own. An LLM created without ``system_message`` has the default system message, so if you set ``system_message`` for the primary LLM, e.g. to an empty string to disable it for a chat, set the same value for each fallback LLM. Each fallback LLM must also have the same tools as the primary LLM, in any order, the same ``tool_choice`` and ``parallel_tool_calls``, and the same support for tool calls (see :ref:`llm-tools-fallback-label`). Finally, a fallback LLM must differ from the primary LLM, and from the other fallback LLMs, in at least one configuration parameter, e.g. the model, the API key or the API base, as an exact copy would only repeat the same request. This is checked when the fallback LLMs are set, i.e. when the primary LLM is created with them or when ``fallback_models`` is assigned. If the primary LLM is used to extract concepts from images, or is passed images in a chat, each of its fallback LLMs must support vision as well, which is checked before any requests are sent. In a chat, the request parameters of the fallback LLMs are checked before any requests are sent as well, e.g. that a fallback LLM with ``parallel_tool_calls`` set supports parallel tool calls. Usage and cost are tracked separately for the primary LLM and for each of its fallback models.

The next fallback model is tried when the current LLM fails for good, i.e. when its request fails with a non-transient error, when the retries of its request are exhausted, when the prompt exceeds its context window, when it keeps returning invalid data during extraction, when its response is cut off at the max output tokens during extraction, or when it returns an empty response in a chat. See :ref:`llm-retries-label` for details. A refusal of the LLM or of its provider counts as invalid data during extraction, and as an empty response in a chat (see :ref:`llm-refusals-label`).

To change the ``system_message`` of an LLM that already has fallback LLMs, change it for the fallback LLMs first, and then for the primary LLM.

A fallback of an LLM can also be a decision model, and an LLM can be a fallback of a decision model. A decision model extracts only aspects and the concepts whose answers come from a closed set, so an LLM can fall back to it only if it can extract everything that the LLM is to extract: this is checked when an extraction method is called, before any requests are sent. The requirements above for the output language, system message and tools apply between two LLMs only. In a chat, a fallback decision model is skipped. See :ref:`decision-models-fallbacks-label`.

.. note::
   Each model applies its own retry settings and timeout before the next fallback model is tried, so the more fallback models you configure, the longer a request can take to fail when all of the models are unavailable.


.. _llm-retries-label:

🔁 How Failed Requests Are Retried
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Retries are configured per model, which also applies to each fallback model, and depend on why the request failed:

- **Transient API errors** (request timeout, rate limit, server or connection error) are retried up to ``num_retries_failed_request`` times. The delay between the retries grows exponentially (about 1, 2, 4, 8 seconds, capped at 10 seconds) and is randomly shortened by up to a half (jitter), so that concurrent requests that failed together are not all retried at the same moment.
- **Non-transient API errors** (e.g. invalid request, failed authentication, exceeded context window) are not retried, as the same request would fail again in the same way. A prompt that is found to exceed the context window of the model before the request is sent is handled in the same way: the request is not sent.
- **Invalid data** returned by the LLM during extraction (the response cannot be parsed or validated, e.g. it has an ID that the prompt does not show or a value of another type than the concept has, it leaves out an aspect or a concept, which it is to list also without items, or it has more than one item for a concept with ``singular_occurrence=True``) is requested again up to ``max_retries_invalid_data`` times, without a delay. A fallback LLM is asked again in the same way: after it is asked for the first time, up to its own ``max_retries_invalid_data`` times, i.e. as many times in total as a primary LLM with the same setting. A fallback LLM with ``max_retries_invalid_data=0`` is asked once. When the content of a call is sent to the LLM in parts, i.e. with ``max_paragraphs_to_analyze_per_call`` or ``max_images_to_analyze_per_call`` set, the LLM is asked again only for the part that it returned invalid data for, and the call goes on with the parts after it: the parts that were answered with valid data are not sent again. The retries are those of the call, not of each part. A fallback LLM extracts from all the parts.
- **A response cut off at the max output tokens** during extraction is not requested again, as the same request would be cut off in the same way (see :ref:`llm-max-output-tokens-label`).
- **A refusal**, of the model or of its provider, is requested again during extraction as invalid data is, as the verdict of a model or of a content filter can differ on another request (see :ref:`llm-refusals-label`).

When a request to a model fails for good, it is not sent to that model again: the next fallback model is tried, if configured. If there are no more fallback models to try, :class:`~contextgem.public.exceptions.ModelAPIError` is raised for a failed API request, for a response cut off at the max output tokens during extraction, or for an empty response in a chat, or :class:`~contextgem.public.exceptions.ModelExtractionError` for invalid data. A refusal raises :class:`~contextgem.public.exceptions.ModelRefusalError`, a subclass of ``ModelExtractionError``, instead. The exceptions can be imported from ``contextgem``, and ``except ContextGemError`` catches either of them (:class:`~contextgem.public.exceptions.ContextGemError`). The error is the one of the last model that was tried: the ``retry_count`` of ``ModelExtractionError`` is the number of times that this model was asked again for invalid data, i.e. it does not count the first time the model was asked. During extraction, a warning is logged instead if ``raise_exception_on_extraction_error=False`` is passed to the extraction method, with the same number of retries in its text. A prompt that exceeds the context window of the last LLM to try raises ``ValueError``.

With ``use_concurrency=True``, the aspects or concepts are extracted in several calls at the same time. When ``ModelAPIError`` or ``ModelExtractionError`` is to be raised for one of the calls, the other calls are stopped first: their requests in flight are cancelled, and no requests are sent afterwards. The aspects and concepts extracted by then keep their items, and the requests completed by then are recorded in the usage.

.. note::
   ``max_retries_failed_request`` sets the number of retries performed by the LLM provider's SDK, where supported (e.g. for OpenAI), in addition to the retries described above. It applies to the first attempt of a request only.


.. _llm-refusals-label:

🚫 Refusals
~~~~~~~~~~~~

A model or its provider can decline to answer a request. ContextGem tells this apart from the other failures by what ``litellm`` reports for all the providers:

- **The model refuses to answer**, e.g. a model of OpenAI with the ``refusal`` field of its message, or a decision model with a refused answer (see :ref:`decision-models-errors-label`). ``litellm`` also gives a refused answer for a question that a decision model leaves unanswered, or answers with an answer of another kind than the question, which ContextGem cannot tell apart from a refusal.
- **The answer is stopped with the** ``content_filter`` **finish reason**, which ``litellm`` reports for the refusal of a Claude model as well as for the content filters of the providers, e.g. the safety filters of Gemini and of Azure OpenAI, and the guardrails of Amazon Bedrock. The finish reason does not tell which of the two it was.
- **The provider blocks the request** on its input, which ``litellm`` raises as ``ContentPolicyViolationError``.

The answer of a refused request is not used, also if it is valid JSON. During extraction, a refusal is handled as invalid data: the request is sent again, up to ``max_retries_invalid_data`` times, as the verdict of a model or of a content filter can differ on another request, and then to the fallback models, each with its own retries. Set ``max_retries_invalid_data=0`` to send a request to each model once. Each refusal is logged when it occurs, at the ``ERROR`` level. If none of the models answers, :class:`~contextgem.public.exceptions.ModelRefusalError` is raised, a subclass of ``ModelExtractionError``, which ``except ModelExtractionError`` and ``except ContextGemError`` catch. Its message names the last model, what declined the answer, and the aspects or concepts of the call, and never has the answer of the model or the text of the document. A warning is logged instead if ``raise_exception_on_extraction_error=False`` is passed to the extraction method, and the aspects or concepts of the call get no items. Each refused request is recorded in the usage, and its cost counted, as any other request.

In a chat, a refused answer that has a text, e.g. the model's own words on why it does not answer, is returned as it is, with a warning logged. A refused answer without a text, and a blocked request, are handled as an empty answer: the next fallback LLM is tried, and ``ModelRefusalError`` is raised if none is left.

Catch ``ModelRefusalError`` to handle refusals apart from the other failures.


.. _llm-max-output-tokens-label:

📏 Limiting the Output Tokens
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

By default, the output tokens of a response are not limited by ContextGem: ``max_tokens`` is not set, no limit is sent with the requests, and the default of the model applies, which for most providers is the maximum output of the model. Set ``max_tokens`` to limit the output of each request, e.g. to cap the cost of a single request:

.. code-block:: python

   from contextgem import DocumentLLM

   llm = DocumentLLM(
       model="openai/gpt-6-luna",
       api_key="<your-openai-api-key>",
       max_tokens=16000,  # for reasoning models, includes the reasoning tokens
   )

The same parameter applies to all models. For reasoning (CoT-capable) models, the reasoning tokens count towards the limit, so a limit that is too low for the reasoning effort of the model can leave no tokens for the response itself.

A response that reaches the limit is cut off by the LLM provider, whether the limit is set with ``max_tokens`` or is the one of the model:

- **During extraction**, a cut-off response cannot be used, and is not requested again, as the same request would be cut off in the same way. The next fallback model is tried, if configured. If there are no more fallback models to try, ``ModelAPIError`` is raised, or a warning is logged if ``raise_exception_on_extraction_error=False`` is passed to the extraction method. To get a complete response, increase ``max_tokens`` or leave it unset, lower ``reasoning_effort``, or get less output per request by setting ``max_items_per_call`` or ``max_paragraphs_to_analyze_per_call`` (see :doc:`../optimizations/optimization_long_docs`).
- **In a chat**, a cut-off response is returned as it is, with a warning logged, as the limit can be set on purpose to get a short answer. A cut-off response without a text, e.g. when all the tokens are used up by the reasoning of the model, is handled as an empty response.

.. note::
   Some providers (e.g. Anthropic) require a limit in each request. If ``max_tokens`` is not set, ``litellm`` sends the maximum output of the model for such a provider. For a model that ``litellm`` has no data on, e.g. a model released after the pinned ``litellm`` version, it sends a low default limit instead, so set ``max_tokens`` explicitly for such a model.


.. _llm-sampling-params-label:

🎲 Setting the Sampling Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

By default, ``temperature`` and ``top_p`` are not set: they are not sent with the requests, and the defaults of the model apply. If you set them, they are sent to any model:

.. code-block:: python

   from contextgem import DocumentLLM

   llm = DocumentLLM(
       model="ollama_chat/llama3.3:70b",
       api_base="http://localhost:11434",
       temperature=0.2,  # more consistent output for structured extraction
   )

A low temperature (e.g. ``0.0`` to ``0.3``) usually makes the output of structured extraction more consistent. This matters most for small and local models, whose default temperature is often set for conversational use (see :doc:`../optimizations/optimization_small_llm_troubleshooting`).

To make the output of such a model more reproducible across runs, you can also set ``seed``. As with the sampling parameters, it is sent only if set, and to any model.

Whether a value is supported depends on the model, which is left to ``litellm`` and the LLM provider. Some reasoning (CoT-capable) models accept only their default sampling parameters, e.g. OpenAI's GPT-5 and GPT-6 models. If ``litellm`` reports a parameter as not supported for the model, the request is sent without it, and a warning is logged for each such request, so it is better to leave the parameter unset for such a model. A value that is rejected only by the API of the provider fails the request as any other invalid request.


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


.. _llm-unknown-prices-label:

🛑 Stopping a Model Whose Prices Are Unknown
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

With ``auto_pricing=True``, a model that ``genai-prices`` has no prices for gets a warning when it is created, and again when the usage of its calls is recorded, and its costs stay at zero. A budget check of your own, e.g. one that stops a run when ``get_cost()`` reaches a limit, would therefore never stop it: ContextGem has no budget of its own, and checks no cost before a request. To have such a model raise where it is created, make these warnings errors with a warning filter on the start of their text:

.. code-block:: python

   import warnings

   from contextgem import ContextGemWarning

   # An LLM or a decision model without prices raises where it is created
   warnings.filterwarnings(
       "error",
       message="Unable to fetch pricing data",
       category=ContextGemWarning,
   )

The filter does not match the other warning of automatic pricing, on the accuracy of the prices (above). For a model created before the filter is set, the warning issued when the usage of a call is recorded raises instead, and stops the extraction method after the usage of that call is recorded. See :doc:`../logging_config` for the warnings of ContextGem and how to filter them.


🧠 Setting the Reasoning Effort
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For reasoning (CoT-capable) models (such as OpenAI's GPT-5 and GPT-6 models), you can set the reasoning effort. The reasoning tokens count towards ``max_tokens``, if set (see :ref:`llm-max-output-tokens-label`):

.. literalinclude:: ../../../dev/usage_examples/docs/llm_config/reasoning_params.py
   :language: python
   :caption: Setting the reasoning effort


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

A group has at least two models. Each model in the group must have a unique ``role``, and only one model in the group can have no role. An aspect/concept is routed to the model with the ``role`` equal to the ``model_role`` of the aspect/concept, and an aspect/concept without ``model_role`` is routed to the model without ``role``. If an aspect/concept cannot be routed to any model in the group, an error is raised. For more details, see :ref:`model-roles-label`.

See a practical example of using a model group in :ref:`multi-llm-pipeline-label`.

A model group can also include decision models, to which aspects/concepts are routed by their roles in the same way. For example, a decision model can answer the yes/no questions and the classifications, while an LLM extracts the rest. The ``output_language`` of a group applies to its LLMs: a decision model has no output language. See :ref:`decision-models-model-group-label`.


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

The call history is kept in memory for as long as the LLM is, with the prompt and the response of each call, i.e. the text of your documents, and it grows with each call. ``reset_usage_and_cost()`` clears it, together with the token counts and the costs: read the costs first if you keep them, e.g. for a budget.
