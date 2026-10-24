.. meta::
   :description: How to choose LLMs for your extraction, and how to send each aspect and concept to the model that suits it with model roles and concept modality.

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
   **Small Model Issues?** If a smaller model, such as many that run locally, gives you JSON validation errors or inconsistent results, see our :doc:`troubleshooting guide <optimization_small_llm_troubleshooting>` for specific solutions and workarounds.

.. seealso::
   **Yes/No Questions, Classifications or Ratings?** The concepts whose answers come from a closed set (:class:`~contextgem.public.concepts.BooleanConcept`, :class:`~contextgem.public.concepts.LabelConcept`, :class:`~contextgem.public.concepts.RatingConcept`), as well as aspects, can be extracted with a decision model instead of an LLM. A decision model generates no text and returns the probabilities of the answers. See :doc:`../decision_models/decision_models` for when it is a good fit and when an LLM is the better choice.


.. _model-roles-label:

🏷️ Model Roles
--------------

The ``role`` of a model is a free-form label used to route aspects/concepts to different models in a :class:`~contextgem.public.models.DocumentModelGroup`. For example, if an aspect/concept is assigned ``model_role="reasoner"``, this aspect/concept is extracted from the document using the model with ``role="reasoner"``, while the aspects/concepts without ``model_role`` are extracted using the model without ``role``. This helps to channel different tasks to different models, ensuring that the task is handled by the most appropriate model. Usually, domain expertise is required to determine the most appropriate model for a specific aspect/concept.

A role is a label of your choice (e.g. ``"reasoner"``, ``"complex"``, ``"vision"``) and implies no capabilities of the model: it is you who decides which model stands behind each role. A role is a string of up to 64 printable characters: the surrounding whitespace is removed, and line breaks, tabs, and other control or invisible characters are not allowed. Roles are compared exactly, i.e. they are case-sensitive.

.. list-table:: Routing of aspects/concepts to models
   :header-rows: 1
   :widths: 30 35 35

   * - Extraction with
     - Aspect/concept without ``model_role``
     - Aspect/concept with ``model_role="reasoner"``
   * - Single model
     - Extracted by the model
     - Extracted by the model
   * - Model group
     - Extracted by the model without ``role``
     - Extracted by the model with ``role="reasoner"``

In model groups, unique role assignments are especially important: each model in the group must have a distinct role (only one model can have no role), so routing can unambiguously send each aspect/concept to the intended model. If an aspect/concept is assigned a role that no model in the group has, an error is raised. This is checked before any requests are sent by the extraction method, including for the nested aspects/concepts that the method processes.

For simple use cases, when working with a single model, you can skip the role assignments completely: a single model extracts all the aspects/concepts it is given, regardless of their ``model_role``.

Roles route aspects/concepts to the decision models of a group in the same way, e.g. to have yes/no questions and classifications answered by a decision model, and the rest extracted by an LLM. See :ref:`decision-models-model-group-label`.

.. seealso::
   **Upgrading from an earlier version?** Before v1.0, roles were predefined names that also determined the content to extract from. See :doc:`../migration_v1` for updating your code and serialized data.

.. note::
  🧠 For LLMs that support reasoning (chain of thought), internal prompts include reasoning-specific instructions intended for these models to produce higher-quality responses. Reasoning support is detected automatically using ``litellm.supports_reasoning()``. If this function does not correctly identify your model's capabilities, declare the capability with ``supports_reasoning=True`` (see :ref:`llm-capabilities-label`).

.. literalinclude:: ../../../dev/usage_examples/docs/optimizations/optimization_choosing_llm.py
    :language: python
    :caption: Example of selecting different LLMs for different tasks


.. _concept-modality-label:

🖼️ Concept Modality
---------------------

The ``modality`` of a concept defines the document content that the concept is extracted from. The modality is independent of model roles: a concept is extracted from the specified content by whichever LLM it is given or routed to.

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
  👁️ Only LLMs that support vision can extract concepts from images. Vision support is detected automatically using ``litellm.supports_vision()``. If the LLM that is to extract concepts from images, or any of its fallback LLMs, does not support vision, an error is raised before any requests are sent. See :ref:`llm-capabilities-label` for declaring the capability with ``supports_vision=True``.

.. note::
  🔀 Concepts with the ``"multimodal"`` modality are extracted from the text first, if the document has text, and then from the images, if it has images. References are supported for such concepts only for what is extracted from the text.

.. note::
  🤖 A single vision-capable LLM can extract concepts of all modalities. If you prefer a dedicated model for images, add it to a model group with a role (e.g. ``role="vision"``) and assign the same ``model_role`` to the concepts with the ``"image"`` or ``"multimodal"`` modality.
