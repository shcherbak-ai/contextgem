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
   * - A **smaller/distilled model** capable of identifying relevant document sections (e.g., ``gpt-6-luna``). This extraction resembles multi-label classification. Complex aspects may occasionally require larger models.
     - For *basic concepts* (e.g., titles, payment amounts, dates), the same **smaller/distilled model** is often sufficient (e.g., ``gpt-6-luna``). For *complex concepts* requiring nuanced understanding within specific aspects or the entire document, consider a **larger model** (e.g., ``gpt-6.1-sol``). For concepts requiring advanced understanding or complex reasoning (e.g., logical deductions, evaluation), consider the most capable model available to you, with a higher ``reasoning_effort`` if the model supports it.

.. seealso::
   **Small Model Issues?** If you're experiencing issues with smaller models (e.g. 8B parameter models), such as JSON validation errors or inconsistent results, see our :doc:`troubleshooting guide <optimization_small_llm_troubleshooting>` for specific solutions and workarounds.


.. _llm-roles-label:

🏷️ LLM Roles
--------------

The ``role`` of an LLM is a free-form label used to route aspects/concepts to different LLMs in a :class:`~contextgem.public.llms.DocumentLLMGroup`. For example, if an aspect/concept is assigned ``llm_role="reasoner"``, this aspect/concept is extracted from the document using the LLM with ``role="reasoner"``, while the aspects/concepts without ``llm_role`` are extracted using the LLM without ``role``. This helps to channel different tasks to different LLMs, ensuring that the task is handled by the most appropriate model. Usually, domain expertise is required to determine the most appropriate LLM for a specific aspect/concept.

A role is a label of your choice (e.g. ``"reasoner"``, ``"complex"``, ``"vision"``) and implies no capabilities of the model: it is you who decides which model stands behind each role. A role is a string of up to 64 printable characters: the surrounding whitespace is removed, and line breaks, tabs, and other control or invisible characters are not allowed. Roles are compared exactly, i.e. they are case-sensitive.

.. list-table:: Routing of aspects/concepts to LLMs
   :header-rows: 1
   :widths: 30 35 35

   * - Extraction with
     - Aspect/concept without ``llm_role``
     - Aspect/concept with ``llm_role="reasoner"``
   * - Single LLM
     - Extracted by the LLM
     - Extracted by the LLM
   * - LLM group
     - Extracted by the LLM without ``role``
     - Extracted by the LLM with ``role="reasoner"``

In LLM groups, unique role assignments are especially important: each model in the group must have a distinct role (only one model can have no role), so routing can unambiguously send each aspect/concept to the intended model. If an aspect/concept is assigned a role that no LLM in the group has, an error is raised. This is checked before any requests are sent by the extraction method, including for the nested aspects/concepts that the method processes.

For simple use cases, when working with a single LLM, you can skip the role assignments completely: a single LLM extracts all the aspects/concepts it is given, regardless of their ``llm_role``.

.. seealso::
   **Upgrading from an earlier version?** Before v1.0, roles were predefined names that also determined the content to extract from. See :doc:`../migration_v1` for updating your code and serialized data.

.. note::
  🧠 For LLMs that support reasoning (chain of thought), internal prompts include reasoning-specific instructions intended for these models to produce higher-quality responses. Reasoning support is detected automatically using ``litellm.supports_reasoning()``, regardless of the role of the LLM. If this function does not correctly identify your model's capabilities, you can explicitly declare the capability by setting ``_supports_reasoning=True`` on the LLM instance.

.. literalinclude:: ../../../dev/usage_examples/docs/optimizations/optimization_choosing_llm.py
    :language: python
    :caption: Example of selecting different LLMs for different tasks


.. _concept-modality-label:

🖼️ Concept Modality
---------------------

The ``modality`` of a concept defines the document content that the concept is extracted from. The modality is independent of LLM roles: a concept is extracted from the specified content by whichever LLM it is given or routed to.

.. list-table:: Available concept modalities
   :header-rows: 1
   :widths: 20 20 30 30

   * - Modality
     - Extraction Context
     - Extracted Item Types
     - Required LLM Capabilities
   * - ``"text"`` (default)
     - Text
     - Concepts (aspect- and document-level). Aspects are always extracted from text.
     - No specific capabilities required
   * - ``"image"``
     - Images
     - Document-level concepts
     - Vision-capable model
   * - ``"multimodal"``
     - Text and/or images
     - Document-level concepts
     - Vision-capable model, when the document has images

.. note::
  👁️ Only LLMs that support vision can extract concepts from images. Vision support is detected automatically using ``litellm.supports_vision()``. If the LLM that is to extract concepts from images, or any of its fallback LLMs, does not support vision, an error is raised before any requests are sent. See :doc:`../llms/llm_config` for declaring the capability explicitly.

.. note::
  🔀 Concepts with the ``"multimodal"`` modality reuse the existing text and image extraction paths. If text exists, the text path runs first; if images exist, the image path runs next. References are only supported for multimodal concepts when text is used.

.. note::
  🤖 A single multimodal LLM can extract concepts of all modalities. If you prefer a dedicated model for images, add it to an LLM group with a role (e.g. ``role="vision"``) and assign the same ``llm_role`` to the concepts with the ``"image"`` or ``"multimodal"`` modality.
