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
* :class:`~contextgem.public.pipelines.DocumentPipeline` - Defines extraction structure and logic
* :class:`~contextgem.public.llms.DocumentLLM` - Stores LLM configuration for document processing

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
