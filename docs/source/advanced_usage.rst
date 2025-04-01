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

Advanced usage examples
=======================

Below are complete, self-contained examples demonstrating advanced usage of ContextGem.


🔍 Extracting Aspects Containing Concepts
------------------------------------------

.. tip::
   Concept extraction is useful for extracting specific data points from a document or an aspect. For example, a "Payment terms" aspect in a contract may have multiple concepts:

   * "Payment amount"
   * "Payment due date"
   * "Payment method"

.. literalinclude:: ../../dev/usage_examples/docs/advanced/advanced_aspects_with_concepts.py
   :language: python


📊 Extracting Aspects and Concepts from a Document
----------------------------------------------------

.. tip::
   This example demonstrates how to extract both document-level concepts and aspect-specific concepts from a document with references. Using concurrency can significantly speed up extraction when working with multiple aspects and concepts.
   
   Document-level concepts apply to the entire document (like "Is Privacy Policy" or "Last Updated Date"), while aspect-specific concepts are tied to particular sections or themes within the document.
   
.. literalinclude:: ../../dev/usage_examples/docs/advanced/advanced_aspects_and_concepts_document.py
   :language: python


🔄 Using a Multi-LLM Pipeline to Extract Data from Several Documents
---------------------------------------------------------------------

.. tip::
   A pipeline is a reusable configuration of extraction steps. You can use the same pipeline to extract data from multiple documents.

   For example, if your app extracts data from invoices, you can configure a pipeline once, and then use it for each incoming invoice.

.. literalinclude:: ../../dev/usage_examples/docs/advanced/advanced_multiple_docs_pipeline.py
   :language: python
