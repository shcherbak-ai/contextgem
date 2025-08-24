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

.. seealso::
   **Small Model Issues?** If you're experiencing issues with smaller models (e.g. 8B parameter models), such as JSON validation errors or inconsistent results, see our :doc:`troubleshooting guide <optimization_small_llm_troubleshooting>` for specific solutions and workarounds.


.. _llm-roles-label:

🏷️ LLM Roles
--------------

The ``role`` of an LLM is an abstraction used to assign various LLMs tasks of different complexity. For example, if an aspect/concept is assigned ``llm_role="extractor_text"``, this aspect/concept is extracted from the document using the LLM with ``role="extractor_text"``. This helps to channel different tasks to different LLMs, ensuring that the task is handled by the most appropriate model. Usually, domain expertise is required to determine the most appropriate role for a specific aspect/concept.

In LLM groups, unique role assignments are especially important: each model in the group must have a distinct role so routing can unambiguously send each aspect/concept to the intended model.

For simple use cases, when working with text-only documents and a single LLM, you can skip the role assignments completely, in which case the roles will default to ``"extractor_text"``.

.. list-table:: Available LLM roles
   :header-rows: 1
   :widths: 20 20 20 20

   * - Role
     - Extraction Context
     - Extracted Item Types
     - Required LLM Capabilities
   * - ``"extractor_text"``
     - Text
     - Aspects and concepts (aspect- and document-level)
     - No reasoning required
   * - ``"reasoner_text"``
     - Text
     - Aspects and concepts (aspect- and document-level)
     - Reasoning-capable model
   * - ``"extractor_vision"``
     - Images
     - Document-level concepts
     - Vision-capable model
   * - ``"reasoner_vision"``
     - Images
     - Document-level concepts
     - Vision-capable and reasoning-capable model
   * - ``"extractor_multimodal"``
     - Text and/or images
     - Document-level concepts
     - Multimodal model supporting text and image inputs
   * - ``"reasoner_multimodal"``
     - Text and/or images
     - Document-level concepts
     - Reasoning-capable multimodal model supporting text and image inputs

.. note::
  🧠 Only LLMs that support reasoning (chain of thought) should be assigned reasoning roles (``"reasoner_text"``, ``"reasoner_vision"``). For such models, internal prompts include reasoning-specific instructions intended for these models to produce higher-quality responses.

.. note::
  👁️ Only LLMs that support vision can be assigned vision roles (``"extractor_vision"``, ``"reasoner_vision"``).

.. note::
  🔀 Multimodal roles (``"extractor_multimodal"``, ``"reasoner_multimodal"``) reuse the existing text and vision extraction paths. If text exists, the text path runs first; if images exist, the vision path runs next. References are only supported for multimodal concepts when text is used.

.. literalinclude:: ../../../dev/usage_examples/docs/optimizations/optimization_choosing_llm.py
    :language: python
    :caption: Example of selecting different LLMs for different tasks
