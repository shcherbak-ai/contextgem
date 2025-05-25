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

.. literalinclude:: ../../../dev/usage_examples/docs/llms/llm_api.py
   :language: python
   :caption: Using cloud LLM providers


💻 Local LLMs
---------------

For local LLMs, you'll need to specify the provider, model name, and the appropriate API base URL:

.. literalinclude:: ../../../dev/usage_examples/docs/llms/llm_local.py
   :language: python
   :caption: Using local LLM providers


.. _gemini_models:

Google Gemini Models
--------------------

ContextGem also supports Google's Gemini models via LiteLLM. You can use both text-based and vision-capable Gemini models.

Key considerations for Gemini models:

*   **Model Naming**: Specify Gemini models using the ``gemini/`` prefix, for example, ``gemini/gemini-pro`` for text and ``gemini/gemini-pro-vision`` for multimodal tasks.
*   **API Keys**: Ensure your Google API key is correctly set up as an environment variable. LiteLLM typically looks for ``GOOGLE_API_KEY`` or ``GEMINI_API_KEY``. Refer to the `LiteLLM documentation <https://docs.litellm.ai/docs/providers/gemini>`_ for the most current details on API key configuration.
*   **Vision Capabilities**: Models like ``gemini/gemini-pro-vision`` can process images. You can provide images to the ``chat`` method as shown in the example.

Here's how you can initialize and use Gemini models:

.. literalinclude:: ../../../dev/usage_examples/docs/llms/gemini_example.py
   :language: python
   :caption: Using Google Gemini Models (Text and Vision)


For a complete list of configuration options available when initializing DocumentLLM instances, see the next section :doc:`llm_config`.
