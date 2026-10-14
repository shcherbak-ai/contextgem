:og:description: ContextGem: Serialization

.. meta::
   :description: How to save and load documents, pipelines and models as dictionaries, JSON strings or JSON files, and what happens to the API credentials of models.

Serializing objects and results
================================

ContextGem provides multiple serialization methods to preserve your document processing pipeline components and results. These methods enable you to save your work, transfer data between systems, or integrate with other applications.

💾 Serialization Methods
-------------------------

The following ContextGem objects support serialization:

* :class:`~contextgem.public.documents.Document` - Contains document content and extracted information
* :class:`~contextgem.public.pipelines.ExtractionPipeline` - Defines extraction structure and logic
* :class:`~contextgem.public.llms.DocumentLLM` - Stores LLM configuration for document processing
* :class:`~contextgem.public.decision_models.DocumentDecisionModel` - Stores decision model configuration for document processing
* :class:`~contextgem.public.models.DocumentModelGroup` - Stores the configuration of the models in the group

Each object supports three serialization methods:

* ``to_json()`` - Converts the object to a JSON string for cross-platform compatibility
* ``to_dict()`` - Converts the object to a Python dictionary for in-memory operations
* ``to_disk(file_path)`` - Saves the object to a JSON file at the specified path, which must end with ``.json``

🔄 Deserialization Methods
---------------------------

To reconstruct objects from their serialized forms, use the corresponding class methods:

* ``from_json(json_string)`` - Creates an object from a JSON string
* ``from_dict(dict_object)`` - Creates an object from a Python dictionary
* ``from_disk(file_path)`` - Loads an object from a file on disk

.. note::
   Deserialization reconstructs the full state of the original document or pipeline: all fields, extracted data, and unique IDs are preserved, so a round-tripped object compares equal to the original (e.g. ``Document.from_dict(document.to_dict()) == document``). The serialized data stores each reference paragraph and reference sentence of the aspects and of the extracted items on its own, while in the restored document they are the paragraphs and sentences of the document itself, i.e. the same objects as in ``document.paragraphs``, as after an extraction (see :ref:`locating-paras-and-sents`). For models, see :ref:`serialization-llm-credentials-label`.


.. _serialization-llm-credentials-label:

🔑 API Credentials of Models
-----------------------------

The API credentials of a model, i.e. its ``api_key`` and ``api_base``, are never included in the serialized data, so that they do not end up in the files, logs or other systems that the serialized data is passed to. The usage and cost statistics of a model are not included either. The other configuration parameters of the model are preserved, together with its fallback models.

A deserialized model therefore has no API credentials. Set them again before using the model, both for the model and for each of its fallback models:

.. code-block:: python
   :caption: Setting the API credentials of a deserialized LLM

   from contextgem import DocumentLLM

   llm = DocumentLLM.from_json(llm_config_json)  # or from_dict() / from_disk()

   llm.api_key = "<your-api-key>"
   llm.api_base = "<your-api-base>"  # if applicable
   for fallback_llm in llm.fallback_models:
       fallback_llm.api_key = "<your-fallback-api-key>"
       fallback_llm.api_base = "<your-fallback-api-base>"  # if applicable

For a deserialized :class:`~contextgem.public.models.DocumentModelGroup`, do the same for each model in its ``models``.

📝 Example Usage
-----------------

.. literalinclude:: ../../dev/usage_examples/docs/serialization/serialization.py
   :language: python

🚀 Use Cases
-------------

* **Caching Results**: Save processed documents to avoid repeating expensive model calls
* **Transfer Between Systems**: Export results from one environment and import in another
* **API Integration**: Convert objects to JSON for API responses
* **Workflow Persistence**: Save pipeline configurations for later reuse
