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

:og:description: ContextGem: Serialization

Serializing objects and results
================================

ContextGem provides multiple serialization methods to preserve your document processing pipeline components and results. These methods enable you to save your work, transfer data between systems, or integrate with other applications.

When using serialization, all extracted data is preserved in the serialized objects.

💾 Serialization Methods
-------------------------

The following ContextGem objects support serialization:

* :class:`~contextgem.public.documents.Document` - Contains document content and extracted information
* :class:`~contextgem.public.pipelines.ExtractionPipeline` - Defines extraction structure and logic
* :class:`~contextgem.public.llms.DocumentLLM` - Stores LLM configuration for document processing
* :class:`~contextgem.public.llms.DocumentModelGroup` - Stores the configuration of the LLMs in the group

Each object supports three serialization methods:

* ``to_json()`` - Converts the object to a JSON string for cross-platform compatibility
* ``to_dict()`` - Converts the object to a Python dictionary for in-memory operations
* ``to_disk(file_path)`` - Saves the object directly to disk at the specified path

🔄 Deserialization Methods
---------------------------

To reconstruct objects from their serialized forms, use the corresponding class methods:

* ``from_json(json_string)`` - Creates an object from a JSON string
* ``from_dict(dict_object)`` - Creates an object from a Python dictionary
* ``from_disk(file_path)`` - Loads an object from a file on disk

.. note::
   Deserialization reconstructs the full state of the original document or pipeline: all fields, extracted data, and unique IDs are preserved, so a round-tripped object compares equal to the original (e.g. ``Document.from_dict(document.to_dict()) == document``). Internal state derived on demand — such as the caches behind :meth:`~contextgem.public.documents.Document.get_paragraph_index` and :meth:`~contextgem.public.documents.Document.get_sentence_index` — is never part of the serialized output and does not affect this equality. For LLMs, see :ref:`serialization-llm-credentials-label`.


.. _serialization-llm-credentials-label:

🔑 API Credentials of LLMs
---------------------------

The API credentials of an LLM, i.e. its ``api_key`` and ``api_base``, are never included in the serialized data, so that they do not end up in the files, logs or other systems that the serialized data is passed to. The usage and cost statistics of an LLM are not included either. The other configuration parameters of the LLM are preserved, together with its fallback LLMs.

A deserialized LLM therefore has no API credentials. Set them again before using the LLM, both for the LLM and for each of its fallback LLMs:

.. code-block:: python
   :caption: Setting the API credentials of a deserialized LLM

   from contextgem import DocumentLLM

   llm = DocumentLLM.from_json(llm_config_json)  # or from_dict() / from_disk()

   llm.api_key = "<your-api-key>"
   llm.api_base = "<your-api-base>"  # if applicable
   for fallback_llm in llm.fallback_models:
       fallback_llm.api_key = "<your-fallback-api-key>"
       fallback_llm.api_base = "<your-fallback-api-base>"  # if applicable

For a deserialized :class:`~contextgem.public.llms.DocumentModelGroup`, do the same for each LLM in its ``models``.

📝 Example Usage
-----------------

.. literalinclude:: ../../dev/usage_examples/docs/serialization/serialization.py
   :language: python

🚀 Use Cases
-------------

* **Caching Results**: Save processed documents to avoid repeating expensive LLM calls
* **Transfer Between Systems**: Export results from one environment and import in another
* **API Integration**: Convert objects to JSON for API responses
* **Workflow Persistence**: Save pipeline configurations for later reuse
