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

Quickstart examples
====================

This guide will help you get started with ContextGem by walking through basic extraction examples.

Below are complete, self-contained examples showing how to extract data from a document using ContextGem.


🔄 Extraction Process
----------------------

ContextGem follows a simple extraction process:

1. Create a :class:`~contextgem.public.documents.Document` instance with your content
2. Define :class:`~contextgem.public.aspects.Aspect` instances for sections of interest
3. Define concept instances (:class:`~contextgem.public.concepts.StringConcept`, :class:`~contextgem.public.concepts.BooleanConcept`, :class:`~contextgem.public.concepts.NumericalConcept`, :class:`~contextgem.public.concepts.DateConcept`, :class:`~contextgem.public.concepts.JsonObjectConcept`, :class:`~contextgem.public.concepts.RatingConcept`) for specific data points to extract, and attach them to :class:`~contextgem.public.aspects.Aspect` (for aspect context) or :class:`~contextgem.public.documents.Document` (for document context).
4. Use :class:`~contextgem.public.llms.DocumentLLM` or :class:`~contextgem.public.llms.DocumentLLMGroup` to perform the extraction
5. Access the extracted data in the document object


📋 Aspect Extraction from Document
-----------------------------------

.. tip::
   Aspect extraction is useful for identifying and extracting specific sections or topics from documents. Common use cases include:

   * Extracting specific clauses from legal contracts
   * Identifying specific sections from financial reports
   * Isolating relevant topics from research papers
   * Extracting product features from technical documentation

.. literalinclude:: ../../dev/usage_examples/docs/quickstart/quickstart_aspect.py
   :language: python

.. raw:: html

   <a target="_blank" href="https://colab.research.google.com/github/shcherbak-ai/contextgem/blob/main/dev/notebooks/docs/quickstart/quickstart_aspect.ipynb">
     <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
   </a>


🌳 Extracting Aspect with Sub-Aspects
--------------------------------------

.. tip::
   Sub-aspect extraction helps organize complex topics into logical components. Common use cases include:

   * Breaking down termination clauses in employment contracts into company rights, employee rights, and severance terms
   * Dividing financial report sections into revenue streams, expenses, and forecasts
   * Organizing product specifications into technical details, compatibility, and maintenance requirements

.. literalinclude:: ../../dev/usage_examples/docs/quickstart/quickstart_sub_aspect.py
   :language: python

.. raw:: html

   <a target="_blank" href="https://colab.research.google.com/github/shcherbak-ai/contextgem/blob/main/dev/notebooks/docs/quickstart/quickstart_sub_aspect.ipynb">
     <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
   </a>


🔍 Concept Extraction from Aspect
----------------------------------

.. tip::
   Concept extraction from aspects helps identify specific data points within already extracted sections or topics. Common use cases include:
   
   * Extracting payment amounts from a contract's payment terms
   * Extracting liability cap from a contract's liability section
   * Isolating timelines from delivery terms
   * Extracting a list of features from a product description
   * Identifying programming languages from a CV's experience section

.. literalinclude:: ../../dev/usage_examples/docs/quickstart/quickstart_concept_aspect.py
   :language: python

.. raw:: html

   <a target="_blank" href="https://colab.research.google.com/github/shcherbak-ai/contextgem/blob/main/dev/notebooks/docs/quickstart/quickstart_concept_aspect.ipynb">
     <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
   </a>


📝 Concept Extraction from Document (text)
-------------------------------------------

.. tip::
   Concept extraction from text documents locates specific information directly from text. Common use cases include:
   
   * Extracting anomalies from entire legal documents
   * Identifying financial figures across multiple report sections
   * Extracting citations and references from academic papers
   * Identifying product specifications from technical manuals
   * Extracting contact information from business documents

.. literalinclude:: ../../dev/usage_examples/docs/quickstart/quickstart_concept_document_text.py
   :language: python

.. raw:: html

   <a target="_blank" href="https://colab.research.google.com/github/shcherbak-ai/contextgem/blob/main/dev/notebooks/docs/quickstart/quickstart_concept_document_text.ipynb">
     <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
   </a>


🖼️ Concept Extraction from Document (vision)
---------------------------------------------

.. tip::
   Concept extraction using vision capabilities processes documents with complex layouts or images. Common use cases include:
   
   * Extracting data from scanned contracts or receipts
   * Identifying information from charts and graphs in reports
   * Identifying visual product features from marketing materials

.. literalinclude:: ../../dev/usage_examples/docs/quickstart/quickstart_concept_document_vision.py
   :language: python

.. raw:: html

   <a target="_blank" href="https://colab.research.google.com/github/shcherbak-ai/contextgem/blob/main/dev/notebooks/docs/quickstart/quickstart_concept_document_vision.ipynb">
     <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
   </a>


💬 Lightweight LLM Chat Interface
----------------------------------

.. note::
   While ContextGem is primarily designed for advanced structured data extraction, it also provides a lightweight, unified interface for interacting with LLMs via natural language - across both text and vision - with built-in fallback support.

.. literalinclude:: ../../dev/usage_examples/readme/llm_chat.py
   :language: python

.. raw:: html

   <a target="_blank" href="https://colab.research.google.com/github/shcherbak-ai/contextgem/blob/main/dev/notebooks/readme/llm_chat.ipynb">
     <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
   </a>
