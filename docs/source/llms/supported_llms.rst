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

:og:description: ContextGem: Supported LLM Providers and Models


Supported LLMs
===============

ContextGem supports all LLM providers and models available through the LiteLLM integration. This means you can use models from major cloud providers like OpenAI, Anthropic, Google, and Azure, as well as run local models through providers like Ollama and LM Studio.

ContextGem works with both types of LLM architectures:

* Reasoning/CoT-capable models (e.g., ``openai/o4-mini``, ``ollama/deepseek-r1:32b``)
* Non-reasoning models (e.g., ``openai/gpt-4.1``, ``ollama/llama3.1:8b``)

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
   **LM Studio Connection Error**: If you encounter a connection error (``litellm.APIError: APIError: Lm_studioException - Connection error``) when using LM Studio, check that you have provided a dummy API key. While API keys are usually not expected for local models, this is a specific case where LM Studio requires one:

   .. literalinclude:: ../../../dev/usage_examples/docs/llms/llm_init/lm_studio_connection_error_fix.py
      :language: python
      :caption: LM Studio with dummy API key

   This is a known issue with calling LM Studio API in litellm: https://github.com/openai/openai-python/issues/961


For a complete list of configuration options available when initializing DocumentLLM instances, see the next section :doc:`llm_config`.
