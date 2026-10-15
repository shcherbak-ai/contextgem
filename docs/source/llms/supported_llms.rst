:og:description: ContextGem: Supported LLM Providers and Models

.. meta::
   :description: ContextGem works with any cloud or local LLM that LiteLLM supports, e.g. from OpenAI, Anthropic, Google, Ollama or LM Studio, with or without reasoning.


Supported LLMs
===============

ContextGem supports all LLM providers and models available through the LiteLLM integration. This means you can use models from major cloud providers like OpenAI, Anthropic, Google, Azure, and xAI, as well as run local models through providers like Ollama and LM Studio.

Both models that support reasoning (chain of thought) and models that do not are supported.

For a complete list of supported providers, see the `LiteLLM Providers documentation <https://docs.litellm.ai/docs/providers>`_.


☁️ Cloud-based LLMs
---------------------

You can initialize cloud-based LLMs by specifying the provider and model name in the format ``<provider>/<model_name>``:

.. literalinclude:: ../../../dev/usage_examples/docs/llms/llm_init/llm_api.py
   :language: python
   :caption: Using cloud LLM providers


💻 Local LLMs
---------------

For local LLMs, you'll need to specify the provider, model name, and the appropriate API base URL:

.. literalinclude:: ../../../dev/usage_examples/docs/llms/llm_init/llm_local.py
   :language: python
   :caption: Using local LLM providers

.. note::
   **Choosing a model**: ContextGem is tested with hosted LLMs such as ``gpt-6-luna`` (OpenAI) and ``claude-haiku-5-5`` (Anthropic). A smaller LLM, such as many that run locally, may not follow the detailed extraction instructions: check it on a sample of your documents, and see the :doc:`troubleshooting guide <../optimizations/optimization_small_llm_troubleshooting>` if it falls short.

.. note::
   **Vision Models with Ollama**: For local vision models that process images, use the ``ollama/`` prefix instead of ``ollama_chat/``, as the latter does not yet support image inputs. For more details, see the relevant Ollama GitHub issues `#10255 <https://github.com/ollama/ollama/issues/10255>`_ and `#6451 <https://github.com/ollama/ollama/issues/6451>`_.


For a complete list of configuration options available when initializing DocumentLLM instances, see the next section :doc:`llm_config`.

.. seealso::
   **Decision models** are not LLMs: they return the probabilities of the answers to typed questions and generate no text. They are configured with a separate class, :class:`~contextgem.public.decision_models.DocumentDecisionModel`, and have their own list of supported providers. See :doc:`../decision_models/decision_models`.
