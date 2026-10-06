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

:og:description: ContextGem: Migrating to v1.0

Migrating to v1.0
==================

This guide describes the changes in ContextGem v1.0 that require updating the code and the serialized data created with the earlier versions.


🏷️ LLM Roles and Concept Modality
-----------------------------------

Before v1.0, LLMs, aspects and concepts were assigned one of six predefined roles: ``"extractor_text"``, ``"reasoner_text"``, ``"extractor_vision"``, ``"reasoner_vision"``, ``"extractor_multimodal"``, and ``"reasoner_multimodal"``. A role combined two things: the LLM to route an aspect/concept to (``extractor`` / ``reasoner``), and the document content to extract from (``text`` / ``vision`` / ``multimodal``).

In v1.0, these are two independent settings:

- The ``role`` of an LLM and the ``llm_role`` of an aspect/concept are free-form labels (strings of up to 64 printable characters), which are used only to route aspects/concepts to the LLMs in an LLM group. They default to ``None`` (no role). See :ref:`llm-roles-label`.
- The ``modality`` of a concept (``"text"``, ``"image"`` or ``"multimodal"``) sets the document content the concept is extracted from. It defaults to ``"text"``. See :ref:`concept-modality-label`.

The predefined role names have no special meaning in v1.0. If they are still used, they are treated as any other role labels.


Updating the Code
~~~~~~~~~~~~~~~~~~

.. list-table:: Replacing the predefined role names
   :header-rows: 1
   :widths: 30 70

   * - Before v1.0
     - In v1.0
   * - ``"extractor_text"``
     - Omit the role (this role was the default)
   * - ``"reasoner_text"``
     - A role label of your choice, e.g. ``"reasoner"``
   * - ``"extractor_vision"``
     - Omit the role, and set ``modality="image"`` on the concept
   * - ``"reasoner_vision"``
     - A role label of your choice, e.g. ``"reasoner"``, and ``modality="image"`` on the concept
   * - ``"extractor_multimodal"``
     - Omit the role, and set ``modality="multimodal"`` on the concept
   * - ``"reasoner_multimodal"``
     - A role label of your choice, e.g. ``"reasoner"``, and ``modality="multimodal"`` on the concept

.. code-block:: python
   :caption: Before v1.0

   from contextgem import BooleanConcept, DocumentLLM, DocumentLLMGroup, StringConcept

   llm_group = DocumentLLMGroup(
       llms=[
           DocumentLLM(model="openai/gpt-4o-mini", role="extractor_text"),
           # The same model, added again to extract from images
           DocumentLLM(model="openai/gpt-4o-mini", role="extractor_vision"),
           DocumentLLM(model="openai/o3-mini", role="reasoner_text"),
       ]
   )

   concepts = [
       StringConcept(name="Title", description="Contract title"),
       StringConcept(
           name="Stamp text",
           description="Text of the stamp",
           llm_role="extractor_vision",
       ),
       BooleanConcept(
           name="Liability cap",
           description="Liability of the parties is capped",
           llm_role="reasoner_text",
       ),
   ]

.. code-block:: python
   :caption: In v1.0

   from contextgem import BooleanConcept, DocumentLLM, DocumentLLMGroup, StringConcept

   llm_group = DocumentLLMGroup(
       llms=[
           # No role: extracts all aspects/concepts that are not routed to another LLM,
           # from text and from images
           DocumentLLM(model="openai/gpt-4o-mini"),
           DocumentLLM(model="openai/o3-mini", role="reasoner"),
       ]
   )

   concepts = [
       StringConcept(name="Title", description="Contract title"),
       StringConcept(
           name="Stamp text",
           description="Text of the stamp",
           modality="image",
       ),
       BooleanConcept(
           name="Liability cap",
           description="Liability of the parties is capped",
           llm_role="reasoner",
       ),
   ]

.. warning::
   A concept that keeps a ``*_vision`` or ``*_multimodal`` role name without ``modality`` being set is extracted from the document **text**, as ``modality`` defaults to ``"text"``. Make sure to set ``modality="image"`` or ``modality="multimodal"`` on all the concepts that are to be extracted from images.


Changes in Behavior
~~~~~~~~~~~~~~~~~~~~

- **No default role name.** ``role`` and ``llm_role`` default to ``None`` instead of ``"extractor_text"``. In an LLM group, an aspect/concept without ``llm_role`` is routed to the LLM without ``role``. An aspect/concept with ``llm_role="extractor_text"`` is no longer matched with an LLM that has no role specified, and vice versa.
- **A single LLM extracts everything it is given.** An LLM used on its own (not in a group) extracts all the aspects/concepts it is given, regardless of their ``llm_role``. Previously, it skipped the aspects/concepts with an ``llm_role`` different from its own ``role``. To extract only some of the aspects/concepts with a single LLM, use the ``from_aspects`` / ``from_concepts`` parameters of the extraction methods.
- **An LLM group does not skip aspects/concepts.** An LLM group raises an error when an aspect/concept cannot be routed to any LLM in the group, i.e. when no LLM in the group has the ``role`` equal to the ``llm_role`` of the aspect/concept. Previously, such aspects/concepts were skipped.
- **A role does not restrict what an LLM can process.** The same model no longer needs to be added to an LLM group several times, with different roles, to process both text and images. To use a dedicated model for images, give it a role of your choice (e.g. ``role="vision"``) and assign the same ``llm_role`` to the concepts with the ``"image"`` or ``"multimodal"`` modality.
- **Model capabilities are not checked against the role.** The warnings about a model not being detected as vision-capable or reasoning-capable are no longer issued when an LLM with a specific role is created. Instead, an error is raised before any requests are sent, if the LLM that is to extract concepts from images, or any of its fallback LLMs, is not detected as vision-capable. See :doc:`llms/llm_config` for declaring the capability explicitly.
- **The role passed to the usage and cost methods of an LLM group is validated.** The ``llm_role`` argument of ``get_usage()``, ``get_cost()`` and ``reset_usage_and_cost()`` of an LLM group is validated in the same way as the roles assigned to LLMs: the surrounding whitespace is removed, and an invalid role raises an error. Previously, the argument was compared as passed, and an empty string selected all LLMs. Omit the argument to select all LLMs.
- **Aspects accept any role label.** As before, aspects and aspect-level concepts are extracted from text only: a concept with the ``"image"`` or ``"multimodal"`` modality cannot be assigned to an aspect.


Updating the Serialized Data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Documents and extraction pipelines serialized with the earlier versions can be deserialized in v1.0. However, their aspects and concepts keep the predefined role names as regular role labels, and their concepts have the default ``"text"`` modality, as the serialized data has no modality. Update the roles and the modalities after deserializing the data, and serialize the data again:

.. code-block:: python
   :caption: Updating a document serialized before v1.0

   from contextgem import Document

   # Role name used before v1.0 -> (role label, concept modality)
   ROLES_MIGRATION = {
       "extractor_text": (None, "text"),
       "reasoner_text": ("reasoner", "text"),
       "extractor_vision": (None, "image"),
       "reasoner_vision": ("reasoner", "image"),
       "extractor_multimodal": (None, "multimodal"),
       "reasoner_multimodal": ("reasoner", "multimodal"),
   }


   def migrate_roles(instances):
       """Migrates the roles of aspects/concepts, including the nested ones."""
       for instance in instances:
           if instance.llm_role in ROLES_MIGRATION:
               role, modality = ROLES_MIGRATION[instance.llm_role]
               instance.llm_role = role
               if hasattr(instance, "modality"):  # concepts only
                   instance.modality = modality
           migrate_roles(getattr(instance, "aspects", []))  # sub-aspects
           migrate_roles(getattr(instance, "concepts", []))  # aspect-level concepts


   document = Document.from_json(document_json)  # serialized before v1.0
   migrate_roles(document.aspects)
   migrate_roles(document.concepts)
   document_json = document.to_json()

The same applies to an :class:`~contextgem.public.pipelines.ExtractionPipeline`. For a serialized :class:`~contextgem.public.llms.DocumentLLM` or :class:`~contextgem.public.llms.DocumentLLMGroup`, update the ``role`` of each deserialized LLM in the same way, or create the LLMs anew.
