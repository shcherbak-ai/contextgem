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

:og:description: ContextGem: Choosing the Right LLM(s)

Choosing the Right LLM(s)
==========================


🧭 General Guidance
---------------------

Your choice of LLM directly affects the accuracy, speed, and cost of your extraction pipeline. ContextGem integrates with various LLM providers (via `LiteLLM <https://github.com/BerriAI/litellm>`_), enabling you to select models that best fit your needs.

Since ContextGem specializes in deep single-document analysis, models with large context windows are recommended. While each use case has unique requirements, our experience suggests the following practical guidelines. However, please note that for sensitive applications (e.g., contract review) where accuracy is paramount and speed/cost are secondary concerns, using the most capable model available for all tasks is often the safest approach.

.. list-table:: Choosing LLMs - General Guidance
   :header-rows: 1
   :widths: 50 50

   * - Aspect Extraction
     - Concept Extraction
   * - A **smaller/distilled non-reasoning model** capable of identifying relevant document sections (e.g., ``gpt-4o-mini``). This extraction resembles multi-label classification. Complex aspects may occasionally require larger or reasoning models.
     - For *basic concepts* (e.g., titles, payment amounts, dates), the same **smaller/distilled non-reasoning model** is often sufficient (e.g., ``gpt-4o-mini``). For *complex concepts* requiring nuanced understanding within specific aspects or the entire document, consider a **larger non-reasoning model** (e.g., ``gpt-4o``). For concepts requiring advanced understanding or complex reasoning (e.g., logical deductions, evaluation), a **reasoning model** like ``o3-mini`` may be appropriate.


.. _llm-roles-label:

🏷️ LLM Roles
--------------

Each LLM serves a specific role in your pipeline, defined by the ``role`` parameter. This allows different models to handle different tasks. For instance, when an aspect or concept has ``llm_role="extractor_text"``, it will be processed by an LLM with that matching role.

You can assign any pre-defined role to an LLM regardless of whether it's actually a "reasoning" model (like o3-mini) or not (like gpt-4o). This abstraction helps organize your pipeline based on your assessment of each model's capabilities and task complexity. For simpler use cases, you can omit role assignments, which will default to ``"extractor_text"``.

.. literalinclude:: ../../../dev/usage_examples/docs/optimizations/optimization_choosing_llm.py
    :language: python
    :caption: Example of selecting different LLMs for different tasks
