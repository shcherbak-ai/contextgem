:og:description: ContextGem: Migrating to v1.0

.. meta::
   :description: How to upgrade code and serialized data from earlier ContextGem versions to v1.0: renamed classes and parameters, model roles and concept modality, and other changes in behavior.

Migrating to v1.0
==================

This guide describes the changes in ContextGem v1.0 that require updating the code and the serialized data created with the earlier versions.

ContextGem v1.0 requires Python 3.11-3.14. Python 3.10, which has reached its end of life, is no longer supported: on Python 3.10, package installers such as ``pip`` and ``uv`` do not install v1.0, and resolve to the latest earlier version.

ContextGem pins the versions of ``litellm`` and ``genai-prices``, which are raised in v1.0 (see the changelog), and ``litellm`` now installs ``boto3`` as a required dependency. If your project pins these packages itself, update its pins.

The sections below end with what to do with the serialized data. Several of them can apply to the same data. The updates to make, in their order, by the kind of the data:

- **A document, or an** ``ExtractionPipeline``, **without aspects and concepts** is deserialized as is. A pipeline serialized as ``DocumentPipeline`` needs its class renamed first (:ref:`migration-v1-pipelines-label`).
- **A document, a pipeline, an aspect or a concept.** Before deserializing the data: rename the role keys (:ref:`migration-v1-renames-label`), replace the rating scales stored as objects (:ref:`migration-v1-rating-scales-label`), and rename the class of a ``DocumentPipeline`` (:ref:`migration-v1-pipelines-label`). After deserializing it: update the roles and the modalities (:ref:`migration-v1-roles-label`).
- **An LLM or a model group.** Before deserializing the data: remove from a group the LLMs that would get the same role (:ref:`migration-v1-roles-label`), give a fallback LLM the system message of its primary LLM if the two differ, and update the data with ``migrate_llm_dict()`` (:ref:`migration-v1-fallback-llms-label`). After deserializing it: update the roles (:ref:`migration-v1-roles-label`) and, if you want the defaults of v1.0, apply them (:ref:`migration-v1-llm-defaults-label`).


.. _migration-v1-renames-label:

✏️ Renamed Classes and Parameters
-----------------------------------

In v1.0, the classes, parameters and attributes that are not specific to LLMs are renamed, so that their names refer to models in general. :class:`~contextgem.public.llms.DocumentLLM` and the ``LLMToolLoopLimitError`` exception keep their names.

.. list-table:: Renamed classes, parameters and attributes
   :header-rows: 1
   :widths: 60 40

   * - Before v1.0
     - In v1.0
   * - ``DocumentLLMGroup``
     - ``DocumentModelGroup``
   * - ``llms`` parameter and attribute of ``DocumentLLMGroup``
     - ``models``
   * - ``llm_role`` parameter and attribute of aspects and concepts
     - ``model_role``
   * - ``llm_role`` argument of ``get_usage()``, ``get_cost()`` and ``reset_usage_and_cost()`` of ``DocumentLLMGroup``
     - ``model_role``
   * - ``llm_roles`` property of documents, pipelines and aspects
     - ``model_roles``
   * - ``LLMPricing``
     - ``ModelPricing``
   * - ``LLMAPIError``
     - ``ModelAPIError``
   * - ``LLMExtractionError``
     - ``ModelExtractionError``

Only the names are changed by the renaming, except for ``model_roles``, which also reports the roles differently (see :ref:`migration-v1-roles-label`). The other changes to model groups, roles and fallback LLMs are described in the sections below. The old names are removed: using them raises an error.

The exceptions (``ModelAPIError``, ``ModelExtractionError`` and ``LLMToolLoopLimitError``) can now be imported from ``contextgem``. Before v1.0, they could be imported only from the internal ``contextgem.internal.exceptions`` module, which still works.

.. code-block:: python
   :caption: Before v1.0

   from contextgem import BooleanConcept, DocumentLLM, DocumentLLMGroup, LLMPricing

   llm_group = DocumentLLMGroup(
       llms=[
           DocumentLLM(
               model="openai/gpt-6-luna",
               pricing_details=LLMPricing(  # the prices of your provider
                   input_per_1m_tokens=1.00,
                   output_per_1m_tokens=2.00,
               ),
           ),
           DocumentLLM(model="openai/gpt-6.1-sol", role="reasoner_text"),
       ]
   )

   concept = BooleanConcept(
       name="Liability cap",
       description="Liability of the parties is capped",
       llm_role="reasoner_text",
   )

   reasoner_usage = llm_group.get_usage(llm_role="reasoner_text")

.. code-block:: python
   :caption: In v1.0

   from contextgem import BooleanConcept, DocumentLLM, DocumentModelGroup, ModelPricing

   model_group = DocumentModelGroup(
       models=[
           DocumentLLM(
               model="openai/gpt-6-luna",
               pricing_details=ModelPricing(  # the prices of your provider
                   input_per_1m_tokens=1.00,
                   output_per_1m_tokens=2.00,
               ),
           ),
           DocumentLLM(model="openai/gpt-6.1-sol", role="reasoner"),
       ]
   )

   concept = BooleanConcept(
       name="Liability cap",
       description="Liability of the parties is capped",
       model_role="reasoner",
   )

   reasoner_usage = model_group.get_usage(model_role="reasoner")

The role names used in the example are changed as well, see :ref:`migration-v1-roles-label`.

**Module of the model group.** The model group is also moved to another module, as it is not specific to LLMs: ``DocumentLLMGroup`` was defined in ``contextgem.public.llms``, while :class:`~contextgem.public.models.DocumentModelGroup` is defined in ``contextgem.public.models``. This affects only the code that imports the class from its module. Importing from ``contextgem``, as in the examples above, works as before:

.. code-block:: python
   :caption: Before v1.0

   from contextgem.public.llms import DocumentLLMGroup

.. code-block:: python
   :caption: In v1.0

   from contextgem import DocumentModelGroup

   # or: from contextgem.public.models import DocumentModelGroup

:class:`~contextgem.public.llms.DocumentLLM` stays in ``contextgem.public.llms``.

**Serialized data.** The serialized data stores the names of the parameters and of the classes. Aspects and concepts serialized with the earlier versions have the ``llm_role`` key, so they, and the documents and pipelines that have aspects or concepts, cannot be deserialized in v1.0 as is: a validation error is raised for the ``llm_role`` key. A document without aspects and concepts needs no update, and neither does such a pipeline, unless it was serialized as ``DocumentPipeline`` (:ref:`migration-v1-pipelines-label`). Rename the key before deserializing the data:

.. code-block:: python
   :caption: Renaming the role key in a document serialized before v1.0

   import json

   from contextgem import Document


   def rename_role_keys(instance_dict):
       """Renames the role key of aspects and concepts, including the nested ones."""
       if "llm_role" in instance_dict:
           instance_dict["model_role"] = instance_dict.pop("llm_role")
       for key in ("aspects", "concepts"):
           for nested_dict in instance_dict.get(key, []):
               rename_role_keys(nested_dict)
       return instance_dict


   document_dict = json.loads(document_json)  # serialized before v1.0
   document = Document.from_dict(rename_role_keys(document_dict))

The same function can be used for a serialized pipeline, aspect or concept. The deserialized aspects and concepts still have the role names used before v1.0, which are to be updated as described in :ref:`migration-v1-roles-label`. For the serialized LLMs and model groups, see :ref:`migration-v1-fallback-llms-label`.


.. _migration-v1-roles-label:

🏷️ Model Roles and Concept Modality
-----------------------------------

Before v1.0, LLMs, aspects and concepts were assigned one of six predefined roles: ``"extractor_text"``, ``"reasoner_text"``, ``"extractor_vision"``, ``"reasoner_vision"``, ``"extractor_multimodal"``, and ``"reasoner_multimodal"``. A role combined two things: the LLM to route an aspect/concept to (``extractor`` / ``reasoner``), and the document content to extract from (``text`` / ``vision`` / ``multimodal``).

In v1.0, these are two independent settings:

- The ``role`` of a model and the ``model_role`` of an aspect/concept (``llm_role`` before v1.0, see :ref:`migration-v1-renames-label`) are free-form labels (strings of up to 64 printable characters), which are used only to route aspects/concepts to the models in a model group. They default to ``None`` (no role). See :ref:`model-roles-label`.
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
           DocumentLLM(model="openai/gpt-6-luna", role="extractor_text"),
           # The same model, added again to extract from images
           DocumentLLM(model="openai/gpt-6-luna", role="extractor_vision"),
           DocumentLLM(model="openai/gpt-6.1-sol", role="reasoner_text"),
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

   from contextgem import BooleanConcept, DocumentLLM, DocumentModelGroup, StringConcept

   model_group = DocumentModelGroup(
       models=[
           # No role: extracts the aspects/concepts that have no `model_role`,
           # from text and from images
           DocumentLLM(model="openai/gpt-6-luna"),
           DocumentLLM(model="openai/gpt-6.1-sol", role="reasoner"),
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
           model_role="reasoner",
       ),
   ]

.. warning::
   A concept that keeps a ``*_vision`` or ``*_multimodal`` role name without ``modality`` being set is extracted from the document **text**, as ``modality`` defaults to ``"text"``. Make sure to set ``modality="image"`` or ``modality="multimodal"`` on all the concepts that are to be extracted from images.


Changes in Behavior
~~~~~~~~~~~~~~~~~~~~

- **No default role name.** ``role`` and ``model_role`` default to ``None`` instead of ``"extractor_text"``. In a model group, an aspect/concept without ``model_role`` is routed to the model without ``role``. An aspect/concept with ``model_role="extractor_text"`` is no longer matched with an LLM that has no role specified, and vice versa.
- **A single LLM extracts everything it is given.** An LLM used on its own (not in a group) extracts all the aspects/concepts it is given, regardless of their ``model_role``. Previously, it skipped the aspects/concepts with a ``model_role`` different from its own ``role``. To extract only some of the aspects/concepts with a single LLM, use the ``from_aspects`` / ``from_concepts`` parameters of ``extract_aspects_from_document()``, ``extract_concepts_from_document()`` and ``extract_concepts_from_aspect()``. ``extract_all()`` has no such parameters, and extracts everything.
- **A model group does not skip aspects/concepts.** A model group raises an error when an aspect/concept cannot be routed to any model in the group, i.e. when no model in the group has the ``role`` equal to the ``model_role`` of the aspect/concept. Previously, such aspects/concepts were skipped.
- **A role does not restrict what an LLM can process.** The same model no longer needs to be added to a model group several times, with different roles, to process both text and images. To use a dedicated model for images, give it a role of your choice (e.g. ``role="vision"``) and assign the same ``model_role`` to the concepts with the ``"image"`` or ``"multimodal"`` modality.
- **Model capabilities are not checked against the role.** The warnings about a model not being detected as vision-capable or reasoning-capable are no longer issued when an LLM with a specific role is created. Instead, an error is raised before any requests are sent, if the LLM that is to extract concepts from images, or any of its fallback LLMs, is not detected as vision-capable. See :doc:`llms/llm_config` for declaring the capability explicitly.
- **The role passed to the usage and cost methods of a model group is validated.** The ``model_role`` argument of ``get_usage()``, ``get_cost()`` and ``reset_usage_and_cost()`` of a model group is validated in the same way as the roles assigned to models: the surrounding whitespace is removed, and an invalid role raises an error. Previously, the argument was compared as passed, and an empty string selected all LLMs. Omit the argument to select all models. The model without a role, which was selected with ``llm_role="extractor_text"`` before v1.0, cannot be selected with ``model_role``: filter the returned entries (``[u for u in model_group.get_usage() if u.role is None]``), or call the method on that model.
- **No role is reported as None.** The ``model_roles`` property of documents, pipelines and aspects (``llm_roles`` before v1.0) contains ``None`` if there are aspects/concepts without a role, where it contained ``"extractor_text"`` before, and now also includes the roles of sub-aspects and of their concepts. The ``list_roles`` property of a model or a model group, and the ``role`` of the entries returned by ``get_usage()`` and ``get_cost()``, are ``None`` for a model without a role. Code that expects strings, e.g. ``sorted(document.model_roles)``, has to handle ``None``.
- **Aspects accept any role label.** Before v1.0, an aspect accepted only the two ``*_text`` role names. As before, aspects and aspect-level concepts are extracted from text only: a concept with the ``"image"`` or ``"multimodal"`` modality cannot be assigned to an aspect.


Updating the Serialized Data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Documents and extraction pipelines that have aspects or concepts, serialized with the earlier versions, cannot be deserialized in v1.0 as is: the role key of their aspects and concepts is to be renamed first, as described in :ref:`migration-v1-renames-label`. The data affected by the removal of ``RatingScale`` or ``DocumentPipeline`` is to be updated before deserializing as well, as described in :ref:`migration-v1-rating-scales-label` and :ref:`migration-v1-pipelines-label`.

The aspects and concepts of the deserialized documents and pipelines keep the predefined role names as regular role labels, and their concepts have the default ``"text"`` modality, as the serialized data has no modality. Update the roles and the modalities after deserializing the data, and serialize the data again:

.. code-block:: python
   :caption: Updating a document serialized before v1.0

   import json

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
           if instance.model_role in ROLES_MIGRATION:
               role, modality = ROLES_MIGRATION[instance.model_role]
               instance.model_role = role
               if hasattr(instance, "modality"):  # concepts only
                   instance.modality = modality
           migrate_roles(getattr(instance, "aspects", []))  # sub-aspects
           migrate_roles(getattr(instance, "concepts", []))  # aspect-level concepts


   document_dict = json.loads(document_json)  # serialized before v1.0
   # If the data has rating scales stored as objects, apply migrate_rating_scales()
   # to it first (see "Rating Scales")
   # rename_role_keys() is defined in "Renamed Classes and Parameters"
   document = Document.from_dict(rename_role_keys(document_dict))
   migrate_roles(document.aspects)
   migrate_roles(document.concepts)
   document_json = document.to_json()

The same applies to an :class:`~contextgem.public.pipelines.ExtractionPipeline`.

A serialized LLM or model group cannot be deserialized in v1.0 as is either: create the LLMs anew, or update the serialized data as described in :ref:`migration-v1-fallback-llms-label`, and then update the roles of the deserialized LLMs.

In a model group, only one model can have no role. If the serialized group has the same model under several roles, e.g. ``"extractor_text"`` and ``"extractor_vision"``, these roles become the same one: remove the extra LLMs from the ``llms`` list of the serialized group before you apply ``migrate_llm_dict()`` to it. Otherwise, no error is raised when the roles are updated, and the group fails later, when an extraction method is called or when the group is serialized and deserialized again. A group needs at least two models: if only one is left, deserialize it as a single LLM.

If such roles are held by different models, e.g. a dedicated model for images under ``"extractor_vision"``, do not remove the model: give its role a label of its own in ``ROLES_MIGRATION``, e.g. ``"extractor_vision": ("vision", "image")``, so that the model and the concepts routed to it get the same label.

When the roles are updated, the fallback LLMs come first, as a fallback LLM must have the role of its primary LLM:

.. code-block:: python
   :caption: Updating the roles of an LLM or model group serialized before v1.0

   def migrate_llm_roles(model):
       """Updates the roles of a deserialized LLM, or of the LLMs of a model group."""
       for llm in getattr(model, "models", [model]):
           # The fallback LLMs first: each must have the role of its primary LLM
           for fallback_llm in llm.fallback_models:
               if fallback_llm.role in ROLES_MIGRATION:
                   fallback_llm.role = ROLES_MIGRATION[fallback_llm.role][0]
           if llm.role in ROLES_MIGRATION:
               llm.role = ROLES_MIGRATION[llm.role][0]


.. _migration-v1-fallback-llms-label:

🔄 Fallback LLMs
------------------

The ``fallback_llm`` parameter and attribute of :class:`~contextgem.public.llms.DocumentLLM`, which accepted a single fallback LLM, are removed. Use ``fallback_models`` instead, which accepts a list of fallback models. The fallback models are tried in the given order until one of them succeeds. See :doc:`llms/llm_config` for details.

.. code-block:: python
   :caption: Before v1.0

   llm = DocumentLLM(
       model="openai/gpt-6-luna",
       api_key="<your-openai-api-key>",
       fallback_llm=fallback_llm,
   )

   llm.fallback_llm = other_fallback_llm
   current_fallback_llm = llm.fallback_llm

.. code-block:: python
   :caption: In v1.0

   llm = DocumentLLM(
       model="openai/gpt-6-luna",
       api_key="<your-openai-api-key>",
       fallback_models=[fallback_llm],
   )

   llm.fallback_models = [other_fallback_llm]
   current_fallback_llm = llm.fallback_models[0] if llm.fallback_models else None

Passing ``fallback_llm`` to ``DocumentLLM``, as well as reading or assigning the ``fallback_llm`` attribute, raises an error in v1.0. ``fallback_models`` does not accept ``None``: where ``fallback_llm=None`` was passed or assigned before v1.0, e.g. to remove the fallback LLM, use an empty list (``llm.fallback_models = []``).

**System message.** In v1.0, a fallback LLM must have the same ``system_message`` as the primary LLM, otherwise an error is raised when the fallback LLMs are set, and when the ``system_message`` of the primary LLM is changed. To change the system message of an LLM that has fallback LLMs, e.g. ``llm.system_message = ""`` for a chat, change it on each of its fallback LLMs first. An LLM created without ``system_message`` has the default system message, so if the primary LLM is created with ``system_message``, e.g. with an empty string to disable it for a chat, pass the same value to each of its fallback LLMs:

.. code-block:: python
   :caption: In v1.0

   llm = DocumentLLM(
       model="openai/gpt-6-luna",
       api_key="<your-openai-api-key>",
       system_message="You are a helpful assistant.",
       fallback_models=[
           DocumentLLM(
               model="anthropic/claude-haiku-4-5",
               api_key="<your-anthropic-api-key>",
               system_message="You are a helpful assistant.",  # same as of the primary LLM
               is_fallback=True,
           )
       ],
   )

Before v1.0, a fallback LLM could have another system message. A fallback LLM created without ``system_message`` then used the default system message, which is written for extraction, also in a chat for which the primary LLM had another system message or none.

**Checks before a chat.** In v1.0, ``chat()`` checks the fallback LLMs before any request is sent, and raises ``ValueError`` if a fallback LLM cannot take the request: if images are passed and it is not detected as vision-capable, if ``parallel_tool_calls`` is set and it is not detected as supporting parallel tool calls, or if its ``max_tokens`` exceeds the maximum output of its model. Before v1.0, this was checked only when the fallback LLM was to be used, so such a chat worked as long as the primary LLM answered. If the model has the capability, declare it on the fallback LLM (``_supports_vision = True``, ``_supports_parallel_tool_calls = True``); otherwise change the configuration.

**Serialized data.** A serialized LLM or model group cannot be deserialized in v1.0 as is, with or without a fallback LLM. An LLM serialized with the earlier versions has the ``fallback_llm`` key (``null`` if no fallback LLM was assigned) and the ``max_completion_tokens`` key, which are removed in v1.0. It also stores the class names used before v1.0 (see :ref:`migration-v1-renames-label`), including those of its usage and cost containers, which are always empty, as the usage and cost statistics are not included in the serialized data. The first error raised is ``TypeError`` for a class name (``Class _LLMUsage does not match _ModelUsage``; for an LLM with pricing details, ``Class LLMPricing does not match ModelPricing``; for a group, ``Class DocumentLLMGroup does not match DocumentModelGroup``), and with the class names updated, a validation error for the two keys. A serialized LLM whose fallback LLM has another system message cannot be deserialized either: set the ``system_message`` key of the fallback LLM to the one of the primary LLM. The serialized data also stores the output token limits used before v1.0 (see :ref:`migration-v1-llm-defaults-label`). As the API credentials are not included in the serialized data, the simplest option is to create the LLMs anew. Otherwise, update the data before deserializing it, and serialize the data again:

.. code-block:: python
   :caption: Updating an LLM serialized before v1.0

   import json

   from contextgem import DocumentLLM


   def migrate_llm_dict(llm_dict):
       """Updates a serialized LLM or model group to the format of v1.0."""
       if llm_dict["__class__"] == "DocumentLLMGroup":
           llm_dict["__class__"] = "DocumentModelGroup"
           llm_dict["models"] = [migrate_llm_dict(d) for d in llm_dict.pop("llms")]
           return llm_dict
       if "fallback_llm" in llm_dict:
           fallback_llm_dict = llm_dict.pop("fallback_llm")
           llm_dict["fallback_models"] = (
               [migrate_llm_dict(fallback_llm_dict)] if fallback_llm_dict else []
           )
       if llm_dict.get("pricing_details"):
           llm_dict["pricing_details"]["__class__"] = "ModelPricing"
       # Output token limits used before v1.0: with `max_tokens` not set, the default
       # of the model applies. To keep a limit, set `max_tokens` to it instead.
       llm_dict.pop("max_completion_tokens", None)
       llm_dict["max_tokens"] = None
       # Empty usage and cost containers, created anew when the LLM is deserialized
       llm_dict.pop("_usage", None)
       llm_dict.pop("_cost", None)
       return llm_dict


   llm_dict = json.loads(llm_json)  # serialized before v1.0
   llm = DocumentLLM.from_dict(migrate_llm_dict(llm_dict))
   llm_json = llm.to_json()

For a serialized model group, use ``DocumentModelGroup.from_dict()`` with the same function.


.. _migration-v1-llm-defaults-label:

📏 Output Token Limit and LLM Defaults
----------------------------------------

The ``max_completion_tokens`` parameter and attribute of :class:`~contextgem.public.llms.DocumentLLM` are removed. Before v1.0, the output tokens of a response were limited by ``max_tokens`` (4096 by default) for the models without reasoning support, and by ``max_completion_tokens`` (16000 by default) for reasoning (CoT-capable) models, while the other of the two parameters was ignored. In v1.0, ``max_tokens`` applies to all models and is not set by default: no limit is sent with the requests, and the default of the model applies, which for most providers is the maximum output of the model. See :ref:`llm-max-output-tokens-label`.

.. code-block:: python
   :caption: Before v1.0

   llm = DocumentLLM(
       model="openai/gpt-6-luna",  # a reasoning model
       api_key="<your-openai-api-key>",
       max_completion_tokens=32000,
   )

.. code-block:: python
   :caption: In v1.0

   llm = DocumentLLM(
       model="openai/gpt-6-luna",
       api_key="<your-openai-api-key>",
       max_tokens=32000,  # or omit it to use the default of the model
   )

Passing ``max_completion_tokens`` to ``DocumentLLM``, as well as reading or assigning the ``max_completion_tokens`` attribute, raises an error in v1.0.

Changes in behavior:

- **No output token limit by default.** If you relied on the previous default limits, e.g. to cap the cost of a single request, set the limit explicitly: ``max_tokens=16000`` for a reasoning model, or ``max_tokens=4096`` for a model without reasoning support.
- **``max_tokens`` limits the output of reasoning models.** Before v1.0, ``max_tokens`` was ignored for reasoning models. If your code sets ``max_tokens`` for such a model, the value now limits the output of the model, including its reasoning tokens. Remove the value, or make sure that it leaves enough tokens for both the reasoning and the response.
- **A cut-off response is not requested again during extraction.** Before v1.0, a response cut off at the output token limit was handled as invalid data: it was requested again up to ``max_retries_invalid_data`` times, and the extraction then failed with an error about invalid JSON. In v1.0, such a response is not requested again: the next fallback model is tried, if configured, and ``ModelAPIError`` is raised otherwise, with a message telling that the response was cut off. In a chat, a cut-off response is still returned, now with a warning logged.
- **No default temperature.** Before v1.0, ``temperature`` defaulted to 0.3. In v1.0, it is not set by default: the parameter is not sent, and the default of the model applies. This changes the default behavior only for the models without reasoning support, as the temperature was not sent to reasoning models before v1.0. To keep the previous behavior for such a model, e.g. a local model, set ``temperature=0.3``. See :ref:`llm-sampling-params-label`.
- **Sampling parameters are sent to reasoning models.** Before v1.0, ``temperature`` and ``top_p`` were ignored for reasoning models. In v1.0, a value that is set is sent to any model. If ``litellm`` reports it as not supported for the model, e.g. for OpenAI's GPT-5 and GPT-6 models, the request is sent without it, with a warning logged for each such request. If your code sets ``temperature`` or ``top_p`` for a reasoning model, remove the value unless you intend it to be applied.
- **The requests differ from those sent before v1.0.** With the default settings, a request no longer has an output token limit, nor, for a model without reasoning support, a temperature. The responses that were recorded or cached for the requests of the earlier versions, e.g. for the tests of your code, no longer match the requests.
- **Longer request timeout.** The default ``timeout`` is raised from 120 to 600 seconds, as a response has to be received in full within the timeout, and a long response of a reasoning model may take more than 120 seconds. A request that times out is retried as a transient error, so a request to an unresponsive endpoint takes longer to fail. Set ``timeout=120`` to keep the previous behavior.
- **Higher default rate of concurrent requests.** With ``use_concurrency=True``, the default ``async_limiter`` of an LLM allows 10 requests per 10 seconds instead of 3. If you run into the rate limits of your LLM provider, set the limiter explicitly, e.g. ``llm.async_limiter = AsyncLimiter(3, 10)`` for the previous rate.
- **The limiter paces all the concurrent requests to an LLM.** Before v1.0, with ``use_concurrency=True``, each extraction call had a limiter of its own. The calls that run concurrently, e.g. for the sub-aspects and for the concepts of different aspects in ``extract_all()``, or the extractions that your code runs concurrently with the async methods, together sent requests at several times the rate of ``async_limiter``. In v1.0, they share the rate. An extraction that relied on the higher rate takes longer: set the limiter to the rate limits of your provider account.
- **A fallback model has a limiter of its own.** With ``use_concurrency=True``, the requests to a fallback model are paced by the ``async_limiter`` of that fallback model. Before v1.0, they were paced by the limiter of the primary LLM. If you lowered the rate on the primary LLM only, set it on its fallback LLMs as well.
- **Only transient errors are retried.** ``num_retries_failed_request`` applies to request timeouts, rate limits, server and connection errors. Other API errors, e.g. an invalid request or a failed authentication, fail right away or are passed to the next fallback model; before v1.0, they were retried as well. ``retry_count`` of ``ModelAPIError`` is the number of retries attempted, not the configured number. During extraction, a fallback LLM applies its own ``num_retries_failed_request``; before v1.0, it failed on its first API error. A request that failed with an API error is not sent to the same LLM again: before v1.0, with a fallback LLM configured or ``raise_exception_on_extraction_error=False`` passed, it was repeated up to ``max_retries_invalid_data`` times, which now applies only to invalid data. ``max_retries_failed_request``, i.e. the retries of the SDK of the provider, applies to the first attempt of a request only.

**Serialized data.** An LLM serialized with the earlier versions has the ``max_completion_tokens`` key, which is not accepted in v1.0, and the ``max_tokens`` key with the value used before v1.0, i.e. 4096 unless it was changed, which would now limit the output of a reasoning model as well. The ``migrate_llm_dict()`` function in :ref:`migration-v1-fallback-llms-label` removes both limits, so that the default of the model applies. The ``temperature``, the ``timeout`` and the ``async_limiter`` of a deserialized LLM keep the values the LLM was serialized with, i.e. the defaults used before v1.0 unless they were changed. This means that the temperature of 0.3 is now sent to a reasoning model as well. Set them anew to use the defaults of v1.0:

.. code-block:: python
   :caption: Applying the defaults of v1.0 to an LLM serialized before v1.0

   import json

   from aiolimiter import AsyncLimiter

   from contextgem import DocumentLLM

   llm_dict = json.loads(llm_json)  # serialized before v1.0
   # migrate_llm_dict() is defined in "Fallback LLMs"
   llm = DocumentLLM.from_dict(migrate_llm_dict(llm_dict))
   llm.temperature = None
   llm.timeout = 600
   llm.async_limiter = AsyncLimiter(10, 10)

The same applies to each fallback LLM, and to each LLM of a deserialized model group.


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
   # If the data has rating scales stored as objects, apply migrate_rating_scales()
   # to it first (see "Rating Scales")
   # rename_role_keys() is defined in "Renamed Classes and Parameters"
   pipeline = ExtractionPipeline.from_dict(rename_role_keys(pipeline_dict))
   pipeline_json = pipeline.to_json()

Serialized documents are not affected by the removal of ``DocumentPipeline``: a document stores the aspects and concepts of the pipeline assigned to it, not the pipeline itself.


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
   # rename_role_keys() is defined in "Renamed Classes and Parameters"
   document_dict = rename_role_keys(migrate_rating_scales(document_dict))
   document = Document.from_dict(document_dict)
   document_json = document.to_json()

The same function can be used for a serialized pipeline or aspect. For a concept that was serialized separately, replace the value of its ``rating_scale`` key in the same way. The rating scales that were defined as tuples are not affected.


.. _migration-v1-probabilities-label:

🎲 Probabilities of Extracted Items
-------------------------------------

In v1.0, each extracted item has the ``probabilities`` attribute. It holds the probabilities of the possible answers for an item extracted by a decision model, which is a new kind of model in v1.0 (see :doc:`decision_models/decision_models`), and is ``None`` for every item extracted by an LLM. No changes to the code are required.

**Serialized data.** The serialized data of each extracted item now includes the ``probabilities`` key, with the ``null`` value for an item without probabilities. Documents, aspects and concepts serialized with the earlier versions have no such key, which needs no update: their items are deserialized with ``probabilities=None``, subject to the other sections of this guide. The reverse does not work: the data serialized in v1.0 cannot be deserialized with the earlier versions. If the serialized data is also read outside ContextGem, e.g. validated against a schema of your own, add the key there. The same applies to the other keys added in v1.0: ``modality`` of concepts, ``decision_threshold`` of aspects, ``BooleanConcept`` and ``LabelConcept``, and ``cache_read_per_1m_tokens`` and ``cache_write_per_1m_tokens`` of pricing details.


.. _migration-v1-tools-label:

🛠️ Tools
----------

In v1.0, a tool for chat is defined in a single way: as a function decorated with ``@register_tool``, which is passed to ``tools`` of the LLM. The tool schema sent to the LLM (the name, the description and the parameters of the tool) is generated from the function. See :doc:`llms/chat_with_tools`.

**Tool schemas as dictionaries.** ``tools`` no longer accepts tool definitions as dictionaries in the OpenAI-compatible schema. Pass the functions instead, and move the details of each schema to the function: the description of the tool to the docstring, the descriptions of the parameters to the docstring or to ``Annotated`` type hints, and their constraints to ``Annotated`` type hints with ``Field`` (an ``enum`` becomes a ``Literal`` type hint): a constraint written in a docstring is not a part of the schema.

.. code-block:: python
   :caption: Before v1.0

   from contextgem import DocumentLLM, register_tool


   @register_tool
   def search_database(query: str, limit: int = 10) -> str:
       return f"Results for '{query}'"


   llm = DocumentLLM(
       model="openai/gpt-6-luna",
       tools=[
           {
               "type": "function",
               "function": {
                   "name": "search_database",
                   "description": "Search the product database",
                   "parameters": {
                       "type": "object",
                       "properties": {
                           "query": {"type": "string", "description": "Search query"},
                           "limit": {"type": "integer", "minimum": 1, "maximum": 50},
                       },
                       "required": ["query"],
                   },
               },
           }
       ],
   )

.. code-block:: python
   :caption: In v1.0

   from typing import Annotated

   from pydantic import Field

   from contextgem import DocumentLLM, register_tool


   @register_tool
   def search_database(
       query: Annotated[str, "Search query"],
       limit: Annotated[int, Field(ge=1, le=50)] = 10,
   ) -> str:
       """
       Search the product database.
       """
       return f"Results for '{query}'"


   llm = DocumentLLM(model="openai/gpt-6-luna", tools=[search_database])

Accordingly, ``llm.tools`` now returns the functions, not the tool schemas.

**Requirements for tool functions.** ``@register_tool`` now raises an error for a function that has no docstring, or that accepts ``*args`` or ``**kwargs``. Before v1.0, such a function was accepted: a tool without a docstring was sent to the LLM without a description, and ``*args`` and ``**kwargs`` were ignored. A parameter without a type hint is reported when the function is decorated, rather than when the LLM is created.

**Errors of tools.** A tool function must return a non-empty string: for an empty string, ``ValueError`` is raised from ``chat()``, and for a value of another type, ``TypeError``, where ``LLMAPIError`` was raised before v1.0. Tools with the same name are rejected when they are set on an LLM; before v1.0, the last of them replaced the others. The tools of an LLM are no longer sent with the requests of the extraction methods.

**Serialized data.** LLMs with tools serialized with the earlier versions can be deserialized in v1.0, subject to the other sections of this guide. As before, the functions of the tools have to be decorated with ``@register_tool`` and be defined or imported, under their names, in the code that loads the LLM. The tool schemas stored in the serialized data are no longer used: the schemas are generated from the functions.


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

For a file converted with Docling, :meth:`~contextgem.public.documents.Document.from_docling` creates the document with the images of the file and the additional context of its paragraphs, which the converter provided (see :doc:`documents/docling`).

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


.. _migration-v1-other-changes-label:

⚙️ Other Changes in Behavior
------------------------------

These changes require no update of the code in most cases, but can change what the same code does:

- **Whitespace between words is kept as a space.** A tab, a line break or another whitespace character between two words is now sent to the model as a space. Before v1.0, such a character was removed and the words around it were joined: ``Fee:<tab>5,000 EUR`` was sent as ``Fee:5,000 EUR``, and a paragraph with a line break inside it, e.g. one segmented with a SaT model or passed as a ``Paragraph``, was sent with the last word of a line joined to the first word of the next one. The prompts for such texts differ from those sent before v1.0: the responses cached or recorded for them no longer match, and the extracted values can differ. A text without such characters is sent unchanged.
- **Negative per-call limits are rejected.** A negative ``max_paragraphs_to_analyze_per_call`` or ``max_images_to_analyze_per_call`` passed to an extraction method raises ``ValueError``. Before v1.0, such a value was ignored or made the LLM extract nothing without an error. Pass ``0``, the default, to set no limit.
- **The reference sentences of an aspect are the extracted sentences.** For an aspect with ``reference_depth="sentences"``, ``aspect.reference_sentences`` has the sentences that were extracted for the aspect, i.e. the reference sentences of its extracted items. Before v1.0, it had all the sentences of each paragraph that a sentence was extracted from. Code that used it as the list of all the sentences of the paragraphs of the aspect is to read them from the paragraphs: ``[s for p in aspect.reference_paragraphs for s in p.sentences]``. An aspect in a document serialized before v1.0 keeps the list it was serialized with, i.e. all the sentences, until it is extracted again. To update it without a new extraction, assign the reference sentences of its extracted items, each once: ``aspect.reference_sentences = list({s.unique_id: s for item in aspect.extracted_items for s in item.reference_sentences}.values())``.
- **References are the paragraphs and sentences of the document itself.** The paragraphs in ``reference_paragraphs`` and the sentences in ``reference_sentences`` of an aspect, of a sub-aspect and of an extracted item are the same objects as in ``document.paragraphs``, after each extraction method and in a document restored with ``from_dict()``, ``from_json()`` or ``from_disk()``. Before v1.0, those of the sub-aspects, of the concepts of sub-aspects, and of everything in a restored document were equal copies with the same unique IDs. Code that changes such an object, e.g. sets ``custom_data`` on a reference paragraph, now changes the paragraph of the document and every other reference to it: copy the object first (``paragraph.clone()``) if the change is to stay local. Comparing with ``==``, and locating references with ``get_paragraph_index()`` and ``get_sentence_index()``, work as before. See :ref:`locating-paras-and-sents`.
- **An empty response in a chat is an error.** For an empty response of the LLM, ``chat()`` tries the next fallback LLM, and raises ``ModelAPIError`` if there is none left to try. Before v1.0, a response without a text raised ``RuntimeError`` if there was no fallback LLM, and an empty text was returned as the answer.
- **A fallback LLM has its own retries for invalid data.** A fallback LLM that returns invalid data during extraction is called once, and then again up to its ``max_retries_invalid_data`` times, as the primary LLM is. Before v1.0, the first call of a fallback LLM was counted as one of its retries: it was called three times with the default of 3, and was not asked again at all with ``max_retries_invalid_data=1``. The same code can therefore call a fallback LLM that keeps returning invalid data one more time, which is recorded in its usage and cost. To keep the previous number of calls, lower the ``max_retries_invalid_data`` of the fallback LLM by one; with ``0``, a fallback LLM is called once, as before v1.0. The ``retry_count`` of ``ModelExtractionError``, and the number of retries in the warning issued instead with ``raise_exception_on_extraction_error=False``, is the number of times that the last model tried was called again: ``0`` is reported for a fallback LLM with ``max_retries_invalid_data=0``, for which ``1`` was reported before v1.0. See :ref:`llm-retries-label`.
- **The error raised for a failed extraction can differ.** With ``use_concurrency=True``, an API error that fails the extraction is raised as ``ModelAPIError``; before v1.0, it was raised as ``ModelExtractionError`` if ``max_retries_invalid_data=0`` was set. An error other than a failure of the model itself (a failed API request, a cut-off response, invalid data) is raised as it is, as without concurrency, e.g. ``ValueError`` for ``max_tokens`` above the limit of the model, or for text concepts of a document that has no text; before v1.0, when the extraction took several calls, such an error was handled as invalid data, and with ``max_retries_invalid_data=0`` and no fallback LLM it was reported as ``ModelExtractionError``, or as a warning only. With or without concurrency, a prompt that exceeds the context window of an LLM is passed to the next fallback LLM, and ``ValueError`` is raised only if there is none left to try; before v1.0, it was raised right away. A sentence ID that is not a number, in the output of an LLM for an aspect with ``reference_depth="sentences"``, is handled as invalid data, where ``ValueError`` was raised before v1.0. Code that catches a specific error for these cases is to catch the one that is raised now.
- **Saving and loading a file raise the errors of the file and of its content.** ``to_disk()`` and ``from_disk()`` raise the error of a file that cannot be written or read as it is, e.g. ``FileNotFoundError`` or ``PermissionError`` (subclasses of ``OSError``), and ``from_disk()`` raises the error of what the file holds as ``from_json()`` raises it, with a note that names the file: ``ValueError`` for data that is not valid JSON or fails validation, ``TypeError`` for the data of another class. Before v1.0, each of these errors was raised as ``RuntimeError``, with the error as its cause. Catch these errors where ``RuntimeError`` was caught (see :doc:`serialization`).
- **Assigning a pipeline to a document with aspects or concepts raises** ``ValueError``. ``assign_pipeline()`` without ``overwrite_existing=True`` raises ``ValueError`` for a document that already has aspects or concepts. Before v1.0, it raised ``RuntimeError``.
- **A call that fails for good stops the other concurrent calls.** With ``use_concurrency=True``, when ``ModelAPIError`` or ``ModelExtractionError`` is to be raised for one of the calls of an extraction, the other calls are stopped first: their requests in flight are cancelled, and no requests are sent afterwards. Before v1.0, the other calls were completed, so their aspects and concepts had their items when the error was raised. If your code catches the error and uses what was extracted, pass ``raise_exception_on_extraction_error=False``: a warning is logged for the failed call, and the other calls are completed.
- **Failures of a run are logged, not issued as Python warnings.** The failure of an extraction with ``raise_exception_on_extraction_error=False``, a chat response cut off at the max output tokens, and the answers that a decision model refuses to give in aspect extraction are logged at the ``WARNING`` level only. Before v1.0, such a message was also issued as a Python warning. Code that checks these messages with ``pytest.warns()`` or a warning filter is to read the log instead. Python warnings are now issued only for what in your code will not have the expected effect, with the ``ContextGemWarning`` category (see :doc:`logging_config`).
- **Validation errors of models do not show the values.** The text of a validation error of a ``DocumentLLM`` or a ``DocumentModelGroup`` no longer has ``input_value=..., input_type=...``, and an LLM no longer shows its ``api_key`` when printed. Code that matches the text of such errors, e.g. a test, is to be updated. The values are still returned by the ``errors()`` and ``json()`` methods of the error, except that ``json()`` writes a model that is among them out without its ``api_key``; ``errors()`` returns such a model as it is, with the key as its attribute. An ``api_key`` that is among the values as it was passed, e.g. a rejected one or the key among the parameters that a model is created with, is returned by both methods as ``'**********'``, and so is the value of any parameter that a model or a group does not have, e.g. of a misspelled ``apikey``. Code that reads the value of a rejected parameter from ``errors()`` or ``json()`` gets the placeholder for these parameters.
- **Cached input tokens are priced separately.** The input tokens that the LLM provider reports as read from, or written to, its prompt cache are now priced at the cache prices of the model with ``auto_pricing=True``, so the calculated input costs are lower for the requests served from the cache, and can be higher for the requests written to it. Previously, all input tokens were priced as regular input tokens, which still applies with ``pricing_details``, unless the cache prices are set. See :ref:`llm-cached-tokens-label`. Two more changes affect the numbers that the same code gets: with ``auto_pricing=True``, each request is priced on its own, where the tokens of several requests were summed up before being priced, and token usage is tracked for every model, where ``get_usage()`` reported zero input and output tokens for an LLM with neither ``pricing_details`` nor ``auto_pricing`` set.
