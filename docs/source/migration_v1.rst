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

Documents and extraction pipelines serialized with the earlier versions can be deserialized in v1.0, unless they are affected by the removal of ``RatingScale`` or ``DocumentPipeline``: such data is to be updated first, as described in :ref:`migration-v1-rating-scales-label` and :ref:`migration-v1-pipelines-label`. However, the aspects and concepts of the deserialized documents and pipelines keep the predefined role names as regular role labels, and their concepts have the default ``"text"`` modality, as the serialized data has no modality. Update the roles and the modalities after deserializing the data, and serialize the data again:

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

The same applies to an :class:`~contextgem.public.pipelines.ExtractionPipeline`. A serialized :class:`~contextgem.public.llms.DocumentLLM` or :class:`~contextgem.public.llms.DocumentLLMGroup` cannot be deserialized in v1.0 as is: either create the LLMs anew, or update the serialized data as described in :ref:`migration-v1-fallback-llms-label`, and then update the ``role`` of each deserialized LLM in the same way.


.. _migration-v1-fallback-llms-label:

🔄 Fallback LLMs
------------------

The ``fallback_llm`` parameter and attribute of :class:`~contextgem.public.llms.DocumentLLM`, which accepted a single fallback LLM, are removed. Use ``fallback_llms`` instead, which accepts a list of fallback LLMs. The fallback LLMs are tried in the given order until one of them succeeds. See :doc:`llms/llm_config` for details.

.. code-block:: python
   :caption: Before v1.0

   llm = DocumentLLM(
       model="openai/gpt-4o-mini",
       api_key="<your-openai-api-key>",
       fallback_llm=fallback_llm,
   )

   llm.fallback_llm = other_fallback_llm
   current_fallback_llm = llm.fallback_llm

.. code-block:: python
   :caption: In v1.0

   llm = DocumentLLM(
       model="openai/gpt-4o-mini",
       api_key="<your-openai-api-key>",
       fallback_llms=[fallback_llm],
   )

   llm.fallback_llms = [other_fallback_llm]
   current_fallback_llm = llm.fallback_llms[0] if llm.fallback_llms else None

Passing ``fallback_llm`` to ``DocumentLLM``, as well as reading or assigning the ``fallback_llm`` attribute, raises an error in v1.0.

**Serialized data.** An LLM serialized with the earlier versions has the ``fallback_llm`` key, with the ``null`` value if no fallback LLM was assigned. This means that any serialized ``DocumentLLM`` or ``DocumentLLMGroup``, with or without a fallback LLM, cannot be deserialized in v1.0 as is: a validation error is raised for the ``fallback_llm`` key. As the API credentials are not included in the serialized data, the simplest option is to create the LLMs anew. Otherwise, replace the ``fallback_llm`` key with ``fallback_llms`` before deserializing the data, and serialize the data again:

.. code-block:: python
   :caption: Updating an LLM serialized before v1.0

   import json

   from contextgem import DocumentLLM


   def migrate_llm_dict(llm_dict):
       """Replaces the single fallback LLM with a list of fallback LLMs."""
       for group_llm_dict in llm_dict.get("llms", []):  # LLMs of an LLM group
           migrate_llm_dict(group_llm_dict)
       if "fallback_llm" in llm_dict:
           fallback_llm_dict = llm_dict.pop("fallback_llm")
           llm_dict["fallback_llms"] = (
               [migrate_llm_dict(fallback_llm_dict)] if fallback_llm_dict else []
           )
       return llm_dict


   llm_dict = json.loads(llm_json)  # serialized before v1.0
   llm = DocumentLLM.from_dict(migrate_llm_dict(llm_dict))
   llm_json = llm.to_json()

For a serialized LLM group, use ``DocumentLLMGroup.from_dict()`` with the same function.


.. _migration-v1-pipelines-label:

🔗 Extraction Pipelines
-------------------------

``DocumentPipeline`` is removed. Use :class:`~contextgem.public.pipelines.ExtractionPipeline` instead, which has the same parameters and methods:

.. code-block:: python
   :caption: Before v1.0

   from contextgem import DocumentPipeline

   pipeline = DocumentPipeline(aspects=aspects, concepts=concepts)

.. code-block:: python
   :caption: In v1.0

   from contextgem import ExtractionPipeline

   pipeline = ExtractionPipeline(aspects=aspects, concepts=concepts)

**Serialized data.** A pipeline serialized as ``DocumentPipeline`` cannot be deserialized in v1.0 as is, as the serialized data stores the name of the class. Replace the class name before deserializing the data, and serialize the data again:

.. code-block:: python
   :caption: Updating a pipeline serialized as DocumentPipeline

   import json

   from contextgem import ExtractionPipeline

   pipeline_dict = json.loads(pipeline_json)  # serialized as DocumentPipeline
   pipeline_dict["__class__"] = "ExtractionPipeline"
   pipeline = ExtractionPipeline.from_dict(pipeline_dict)
   pipeline_json = pipeline.to_json()

Serialized documents are not affected: a document stores the aspects and concepts of the pipeline assigned to it, not the pipeline itself.


.. _migration-v1-rating-scales-label:

⭐ Rating Scales
-----------------

``RatingScale`` is removed. Define the rating scale of a :class:`~contextgem.public.concepts.RatingConcept` as a tuple of ``(start, end)`` integers instead. The requirements for the scale are the same: the start value cannot be negative, and the end value must be greater than the start value.

.. code-block:: python
   :caption: Before v1.0

   from contextgem import RatingConcept, RatingScale

   concept = RatingConcept(
       name="Product quality",
       description="Quality of the product",
       rating_scale=RatingScale(start=1, end=5),
   )

   scale_start = concept.rating_scale.start
   scale_end = concept.rating_scale.end

.. code-block:: python
   :caption: In v1.0

   from contextgem import RatingConcept

   concept = RatingConcept(
       name="Product quality",
       description="Quality of the product",
       rating_scale=(1, 5),
   )

   scale_start, scale_end = concept.rating_scale

``RatingScale()`` created without arguments had the scale from 0 to 10, which corresponds to ``rating_scale=(0, 10)``.

**Serialized data.** A rating scale defined as a ``RatingScale`` object was serialized as an object (``{"start": 1, "end": 5, "__class__": "RatingScale"}``), while a rating scale defined as a tuple is serialized as a list (``[1, 5]``). Documents, pipelines, aspects, and concepts with the rating scales serialized as objects cannot be deserialized in v1.0 as is. Replace such rating scales with lists before deserializing the data, and serialize the data again:

.. code-block:: python
   :caption: Updating a document serialized with RatingScale objects

   import json

   from contextgem import Document


   def migrate_rating_scales(instance_dict):
       """Replaces the serialized RatingScale objects with [start, end] lists."""
       for concept_dict in instance_dict.get("concepts", []):
           rating_scale = concept_dict.get("rating_scale")
           if isinstance(rating_scale, dict):
               concept_dict["rating_scale"] = [rating_scale["start"], rating_scale["end"]]
       for aspect_dict in instance_dict.get("aspects", []):  # including sub-aspects
           migrate_rating_scales(aspect_dict)
       return instance_dict


   document_dict = json.loads(document_json)  # serialized before v1.0
   document = Document.from_dict(migrate_rating_scales(document_dict))
   document_json = document.to_json()

The same function can be used for a serialized pipeline or aspect. For a concept that was serialized separately, replace the value of its ``rating_scale`` key in the same way. The rating scales that were defined as tuples are not affected.


.. _migration-v1-docx-converter-label:

📄 DOCX Converter
------------------

``DocxConverter`` is removed. Convert your files to text with a dedicated document conversion library, e.g. `Docling <https://github.com/docling-project/docling>`_ or `MarkItDown <https://github.com/microsoft/markitdown>`_, and pass the text to :class:`~contextgem.public.documents.Document`:

.. code-block:: python
   :caption: Before v1.0

   from contextgem import DocxConverter

   document = DocxConverter().convert("path/to/file.docx")

.. code-block:: python
   :caption: In v1.0

   from contextgem import Document

   # text = <plain text or Markdown produced by the conversion library of your choice>
   document = Document(raw_text=text)

``DocxConverter.convert_to_text_format()`` and the exceptions of the converter (``DocxConverterError`` and its subclasses) are removed as well, and ContextGem no longer depends on ``lxml``.

What else the converter provided, and how to get it in v1.0:

- **Images.** The converter attached the images embedded in a DOCX file to the document. Extract the images with your conversion library, and pass them to the document via ``images``. See :func:`~contextgem.public.utils.create_image`.
- **Additional context of paragraphs.** The converter described the style, list and table position of each paragraph in its ``additional_context``. To keep providing such details to the LLM, create the paragraphs yourself. See :ref:`paragraph-additional-context`.
- **Markdown text.** The converter kept two versions of the text: the plain text in ``raw_text``, which was used for paragraph and sentence segmentation and was returned in references, and the Markdown text, which was sent to the LLM instead of the plain text. In v1.0, a document has only one text, ``raw_text``.

Markdown can be passed as ``raw_text``: it is sent to the LLM as it is. Keep in mind that in this case:

- The paragraphs and sentences returned in references contain the Markdown syntax.
- With the default paragraph segmentation, each line of the text is a paragraph, e.g. each row of a Markdown table.
- Sentence segmentation, which is used for sentence-level references (``reference_depth="sentences"``), is more accurate on plain text. On Markdown, a link, a table row or the emphasis markers can split a sentence or end up as a separate sentence. If you rely on sentence-level references, pass plain text.
- Use the default ``paragraph_segmentation_mode="newlines"``. The ``"sat"`` mode detects considerably fewer paragraph breaks in Markdown than in plain text.

**Serialized data.** Documents serialized with the earlier versions, including the documents created with ``DocxConverter``, can be deserialized in v1.0, subject to the other sections of this guide. The Markdown text stored in such documents is ignored: text is extracted from ``raw_text``, and the Markdown text is not kept when the document is serialized again. The additional context of the paragraphs is preserved.
