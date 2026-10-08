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

:og:description: ContextGem: How it works

How it works
=============


📏 Leveraging LLM Context Windows
----------------------------------

ContextGem leverages LLMs' long context windows to deliver superior extraction accuracy. Unlike RAG approaches that often `struggle with complex concepts and nuanced insights <https://www.linkedin.com/pulse/raging-contracts-pitfalls-rag-contract-review-shcherbak-ai-ptg3f/>`_, ContextGem is betting on the continuously expanding context capacity, evolving capabilities of modern LLMs, and constantly decreasing LLM costs. This approach enables direct information extraction from full documents, eliminating retrieval inconsistencies and capturing the complete context necessary for accurate understanding.


🧩 Core Components
-------------------

ContextGem's main elements are the Document, Aspect, and Concept models:


📄 **Document**
~~~~~~~~~~~~~~~~

:class:`~contextgem.public.documents.Document` model contains text and/or visual content representing a specific document. Documents can vary in type and purpose, including but not limited to:
    - *Contracts*: legal agreements defining terms and obligations.
    - *Invoices*: financial documents detailing transactions and payments.
    - *Curricula Vitae (CVs)*: resumes outlining an individual's professional experience and qualifications.
    - *General documents*: any other types of documents that may contain text or images.


🔍 **Aspect**
~~~~~~~~~~~~~~~

:class:`~contextgem.public.aspects.Aspect` model contains text representing a defined area or topic within a document (or another aspect) that requires focused attention. Each aspect reflects a specific subject or theme. For example:
    - *Contract aspects*: payment terms, parties involved, or termination clauses.
    - *Invoice aspects*: due dates, line-item breakdowns, or tax details.
    - *CV aspects*: work experience, education, or skills.
    
Aspects may have sub-aspects, for more granular extraction with nested context. This hierarchical structure allows for progressive refinement of focus areas, enabling precise extraction of information from complex documents while maintaining the contextual relationships between different levels of content.


💡 **Concept**
~~~~~~~~~~~~~~~

Concept model contains a unit of information or an entity, derived from an aspect or the broader document context. Concepts represent a wide range of data points and insights, from simple entities (names, dates, monetary values) to complex evaluations, conclusions, and answers to specific questions. Concepts can be:
    - *Factual extractions*: such as a penalty clause in a contract, a total amount due in an invoice, or a certification in a CV.
    - *Analytical insights*: such as risk assessments, compliance evaluations, or pattern identifications.
    - *Reasoned conclusions*: such as determining whether a document meets specific criteria or answers particular questions.
    - *Interpretative judgments*: such as ratings, classifications, or qualitative assessments based on document content.

Concepts may be attached to an aspect or a document. The context for the concept extraction will be the aspect or document, respectively. This flexible attachment allows for both targeted extraction from specific document sections and broader analysis across the entire document content. When attached to aspects, concepts benefit from the focused context, enabling more precise extraction of domain-specific information. When attached to documents, concepts can leverage the complete context to identify patterns, anomalies, or insights that span multiple sections.

Multiple concept types are supported: :class:`~contextgem.public.concepts.StringConcept`, :class:`~contextgem.public.concepts.BooleanConcept`, :class:`~contextgem.public.concepts.NumericalConcept`, :class:`~contextgem.public.concepts.DateConcept`, :class:`~contextgem.public.concepts.JsonObjectConcept`, :class:`~contextgem.public.concepts.RatingConcept`, :class:`~contextgem.public.concepts.LabelConcept`

.. list-table:: Component Examples
   :header-rows: 1
   :widths: 15 20 20 20 20

   * - 
     - Document
     - Aspect
     - Sub-aspect
     - Concept
   * - **Legal**
     - *Software License Agreement*
     - Intellectual Property Rights
     - Patent Indemnification
     - Indemnification Coverage Scope (``JsonObjectConcept``)
   * - **Financial**
     - *Quarterly Earnings Report*
     - Revenue Analysis
     - Regional Performance
     - Year-over-Year Growth Rate (``NumericalConcept``)
   * - **Healthcare**
     - *Medical Research Paper*
     - Methodology
     - Patient Selection Criteria
     - Inclusion/Exclusion Validity (``BooleanConcept``)
   * - **Technical**
     - *System Architecture Document*
     - Security Framework
     - Authentication Protocols
     - Implementation Risk Rating (``RatingConcept``)
   * - **HR**
     - *Employee Handbook*
     - Leave Policy
     - Parental Leave Benefits
     - Eligibility Start Date (``DateConcept``)


🔄 Extraction Workflow
-----------------------

ContextGem uses the following models to extract information from documents:


🤖 **DocumentLLM**
~~~~~~~~~~~~~~~~~~~

**A single configurable LLM to extract information from the document.**

A single LLM extracts all the aspects and concepts it is given: from the document text and, for the concepts with the ``"image"`` or ``"multimodal"`` modality, from the document images (provided that the model supports vision). No role assignment is needed.

An LLM can have one or more configurable fallback LLMs, which are tried in the given order.

See :class:`~contextgem.public.llms.DocumentLLM` and :ref:`concept-modality-label` for more details.


🤖🤖 **DocumentModelGroup**
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**A group of LLMs with different unique roles to extract different information from the document.**

For more complex and granular extraction workflows, a model group can be used to extract different information from the same document using different LLMs with different roles. For example, a simpler LLM e.g. gpt-6-luna can be used to extract specific aspects of the document, and a more powerful LLM e.g. gpt-6.1-sol will handle the extraction of complex concepts that require reasoning over the aspects' context.

The ``role`` of an LLM is a free-form label used to route aspects/concepts to the LLMs in the group. For example, if an aspect/concept is assigned ``model_role="reasoner"``, this aspect/concept is extracted from the document using the LLM with ``role="reasoner"``, while the aspects/concepts without ``model_role`` are extracted using the LLM without ``role``. This helps to channel different tasks to different LLMs, ensuring that the task is handled by the most appropriate model. Usually, domain expertise is required to determine the most appropriate LLM for a specific aspect/concept.

Each LLM can have its own backend and configuration, and one or more fallback LLMs with the same role.

See :class:`~contextgem.public.models.DocumentModelGroup` and :ref:`model-roles-label` for more details.

.. list-table:: Model Group Workflow Example
   :header-rows: 1
   :widths: 15 20 20 20

   * - 
     - LLM 1 (no role)
     - LLM 2 (``role="reasoner"``)
     - LLM 3 (``role="vision"``)
   * - *Model*
     - gpt-6-luna
     - gpt-6.1-sol
     - gpt-6-luna
   * - *Task*
     - Extract payment terms from a contract
     - Detect anomalies in the payment terms
     - Extract invoice amounts
   * - *Fallback LLM(s)* (optional)
     - gpt-5.6-luna
     - gpt-5.6-sol
     - gpt-5.6-luna

.. image:: _static/contextgem_how_it_works_infographics.png
   :width: 100%
   :alt: ContextGem - How it works infographics
   :target: #
   :align: center


ℹ️ What ContextGem Doesn't Offer (Yet)
---------------------------------------

While ContextGem excels at structured data extraction from individual documents, it's important to understand its intentional design boundaries:

**Not a RAG framework**
~~~~~~~~~~~~~~~~~~~~~~~

ContextGem focuses on in-depth single-document analysis, leveraging long context windows of LLMs for maximum accuracy and precision. It does not offer RAG capabilities for cross-document querying or corpus-wide information retrieval. For these use cases, modern RAG frameworks such as LlamaIndex remain more appropriate.

**Not an agent framework**
~~~~~~~~~~~~~~~~~~~~~~~~~~

ContextGem is not designed as an agent framework. It now supports tool calling in chat (function-calling) with optional parallel tool calls and JSON schema validation of tool arguments, which enables building lightweight, task-oriented agents using your own control loop together with :class:`~contextgem.public.llms.ChatSession`. For full agent orchestration (planning/critique, goal decomposition, long-term memory, schedulers, multi-agent coordination), we recommend frameworks specifically designed for this purpose. ContextGem integrates cleanly as a high-accuracy document extraction tool in larger agent systems thanks to its simple API and structured outputs.
