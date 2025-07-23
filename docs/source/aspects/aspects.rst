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

:og:image: https://contextgem.dev/_static/docs_preview_image_aspects.png

Aspect Extraction
===================

:class:`~contextgem.public.aspects.Aspect` is a fundamental component of ContextGem that represents a defined area or topic within a document that requires focused attention. Aspects help identify and extract specific sections or themes from documents according to predefined criteria.


📝 Overview
-------------

Aspects serve as containers for organizing and structuring document content extraction. They allow you to:

* **Extract document sections**: Identify and extract specific parts of documents (e.g., contract clauses, report sections, policy terms)
* **Organize content hierarchically**: Create sub-aspects to break down complex topics into logical components
* **Define extraction scope**: Focus on specific areas of interest before applying detailed concept extraction

While concepts extract specific data points, aspects extract entire sections or topics from documents, providing context for subsequent detailed analysis.


⭐ Key Features
-----------------

Hierarchical Organization
~~~~~~~~~~~~~~~~~~~~~~~~~

Aspects support nested structures through sub-aspects, allowing you to break down complex topics:

* **Parent aspects** represent broad topics (e.g., *"Termination Clauses"*)
* **Sub-aspects** represent specific components (e.g., *"Notice Period"*, *"Severance Terms"*, *"Company Rights"*)

Integration with Concepts
~~~~~~~~~~~~~~~~~~~~~~~~~~

Aspects can contain :class:`~contextgem.internal.base.concepts._Concept` instances for detailed data extraction within the identified sections, creating a two-stage extraction workflow.

.. note::
   See supported concept types in :doc:`../concepts/supported_concepts`. All public concept types inherit from the internal :class:`~contextgem.internal.base.concepts._Concept` base class.


💻 Basic Usage
--------------

Simple Aspect Extraction
~~~~~~~~~~~~~~~~~~~~~~~~~

Here's how to extract a specific section from a document:

.. literalinclude:: ../../../dev/usage_examples/docs/aspects/basic_aspect.py
   :language: python

.. raw:: html

   <a target="_blank" href="https://colab.research.google.com/github/shcherbak-ai/contextgem/blob/main/dev/notebooks/docs/aspects/basic_aspect.ipynb">
     <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
   </a>

Aspect with Sub-Aspects
~~~~~~~~~~~~~~~~~~~~~~~~~

Breaking down complex topics into components:

.. literalinclude:: ../../../dev/usage_examples/docs/aspects/aspect_with_sub_aspects.py
   :language: python

.. raw:: html

   <a target="_blank" href="https://colab.research.google.com/github/shcherbak-ai/contextgem/blob/main/dev/notebooks/docs/aspects/aspect_with_sub_aspects.ipynb">
     <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
   </a>


⚙️ Parameters
--------------

When creating an :class:`~contextgem.public.aspects.Aspect`, you can configure the following parameters:

.. list-table::
   :widths: 20 15 15 50
   :header-rows: 1

   * - Parameter
     - Type
     - Default Value
     - Description
   * - ``name``
     - ``str``
     - (Required)
     - A unique name identifier for the aspect. Must be unique among sibling aspects.
   * - ``description``
     - ``str``
     - (Required)
     - A detailed description of what the aspect represents and what content should be extracted. Must be unique among sibling aspects.
   * - ``aspects``
     - ``list[Aspect]``
     - ``[]``
     - *Optional*. List of sub-aspects for hierarchical organization. Limited to one nesting level.
   * - ``concepts``
     - ``list[_Concept]``
     - ``[]``
     - *Optional*. List of concepts associated with the aspect for detailed data extraction within the aspect's scope. See supported concept types in :doc:`../concepts/supported_concepts`.
   * - ``llm_role``
     - ``str``
     - ``"extractor_text"``
     - The role of the LLM responsible for aspect extraction. Available values: ``"extractor_text"``, ``"reasoner_text"``. For more details, see :ref:`llm-roles-label`. Note that aspects only support text-based extraction. For this reason, aspects cannot have vision LLM roles (i.e. ``llm_role`` parameter value ending with "_vision"). Concepts with vision LLM roles cannot be used within aspects.
   * - ``reference_depth``
     - ``str``
     - ``"paragraphs"``
     - The structural depth of references. Available values: ``"paragraphs"``, ``"sentences"``. Paragraph references are always populated for aspect's extracted items, as aspect's extracted items represent existing text segments. Sentence references are only populated when ``reference_depth="sentences"``.
   * - ``add_justifications``
     - ``bool``
     - ``False``
     - Whether the LLM will output justification for each extracted item. Justifications provide valuable insights into why specific text segments were extracted for the aspect, helping you understand the LLM's reasoning, verify extraction accuracy, and debug unexpected results. This is particularly useful when working with complex aspects.
   * - ``justification_depth``
     - ``str``
     - ``"brief"``
     - The level of detail for justifications. Available values: ``"brief"``, ``"balanced"``, ``"comprehensive"``.
   * - ``justification_max_sents``
     - ``int``
     - ``2``
     - Maximum number of sentences in a justification.


📊 Extracted Items
--------------------

When an :class:`~contextgem.public.aspects.Aspect` is extracted, it is populated with **a list of extracted items** accessible through the ``.extracted_items`` property. Each item is an instance of the ``_StringItem`` class with the following attributes:

.. list-table::
   :widths: 20 20 60
   :header-rows: 1

   * - Attribute
     - Type
     - Description
   * - ``value``
     - str
     - The extracted text segment representing the aspect
   * - ``justification``
     - str
     - Explanation of why this text segment was identified as relevant to the aspect (only if ``add_justifications=True``)
   * - ``reference_paragraphs``
     - list[:class:`~contextgem.public.paragraphs.Paragraph`]
     - List of paragraph objects that contain the extracted aspect content (always populated for aspect's extracted items)
   * - ``reference_sentences``
     - list[:class:`~contextgem.public.sentences.Sentence`]
     - List of sentence objects that contain the extracted aspect content (only if ``reference_depth="sentences"``)


🚀 Advanced Usage
-------------------

Aspects with Concepts
~~~~~~~~~~~~~~~~~~~~~~

Combining aspect extraction with detailed concept extraction:

.. literalinclude:: ../../../dev/usage_examples/docs/aspects/aspect_with_concepts.py
   :language: python

.. raw:: html

   <a target="_blank" href="https://colab.research.google.com/github/shcherbak-ai/contextgem/blob/main/dev/notebooks/docs/aspects/aspect_with_concepts.ipynb">
     <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
   </a>

Complex Hierarchical Structure
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Creating a comprehensive document analysis structure with aspects, sub-aspects and concepts:

.. literalinclude:: ../../../dev/usage_examples/docs/aspects/complex_hierarchy.py
   :language: python

.. raw:: html

   <a target="_blank" href="https://colab.research.google.com/github/shcherbak-ai/contextgem/blob/main/dev/notebooks/docs/aspects/complex_hierarchy.ipynb">
     <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
   </a>

Justifications for Extraction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Justifications provide explanations for why specific text segments were identified as relevant to an aspect. Justifications help users understand the reasoning behind extractions and evaluate their relevance. When enabled, each extracted item includes a generated explanation of why that text segment was considered part of the aspect.

Example:

.. literalinclude:: ../../../dev/usage_examples/docs/aspects/aspect_with_justifications.py
   :language: python

.. raw:: html

   <a target="_blank" href="https://colab.research.google.com/github/shcherbak-ai/contextgem/blob/main/dev/notebooks/docs/aspects/aspect_with_justifications.ipynb">
     <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
   </a>

.. note::
   References are always included for aspects. The ``reference_paragraphs`` field is automatically populated in extracted items of aspects, as they represent existing text segments in the document. The ``reference_sentences`` field is only populated when ``reference_depth`` is set to ``"sentences"``. You can access these references as follows:
   
   .. code-block:: python
   
      # Always available for aspects
      aspect.extracted_items[0].reference_paragraphs
      
      # Only populated if reference_depth="sentences"
      aspect.extracted_items[0].reference_sentences


💡 Best Practices
-----------------

Aspect Definition
~~~~~~~~~~~~~~~~~

* **Be specific**: Provide clear, detailed descriptions that help the LLM understand exactly what content constitutes the aspect
* **Use domain terminology**: Include relevant domain-specific terms that help identify the target content
* **Define scope clearly**: Specify what should and shouldn't be included in the aspect

Structuring Complex Content
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* **Logical decomposition**: Break down complex topics into logical, non-overlapping components
* **Meaningful relationships**: Ensure sub-aspects and/or concepts genuinely belong to their parent aspect


Integration Strategy
~~~~~~~~~~~~~~~~~~~~

* **Two-stage extraction**: Use aspects to identify relevant sections first, then apply sub-aspects and/or concepts for detailed data extraction
* **Scope alignment**: Ensure sub-aspects and/or concepts are relevant to their containing aspects
* **Reference tracking**: Enable references when you need to trace extracted data back to source locations


🎯 Example Use Cases
----------------------

These are examples of how aspects may be used in different domains:

Contract Analysis
~~~~~~~~~~~~~~~~~~

* **Termination Clauses**: Extract and analyze termination conditions, notice periods, and severance terms
* **Payment Terms**: Identify payment schedules, amounts, and conditions
* **Liability Sections**: Extract liability caps, limitations, and indemnification clauses
* **Intellectual Property**: Identify IP ownership, licensing, and usage rights

Financial Reports
~~~~~~~~~~~~~~~~~

* **Revenue Sections**: Extract revenue recognition, breakdown by segments, and growth analysis
* **Compliance Sections**: Identify regulatory compliance statements and audit findings
* **Key Performance Indicators**: Extract precise numerical metrics like EBITDA margins, debt-to-equity ratios, and year-over-year percentage changes

Technical Documentation
~~~~~~~~~~~~~~~~~~~~~~~~

* **Product Specifications**: Extract technical requirements, features, and performance criteria
* **Installation Procedures**: Identify setup steps, configuration requirements, and dependencies
* **Troubleshooting Sections**: Extract problem descriptions, diagnostic steps, and solutions
* **API Documentation**: Identify endpoints, parameters, and usage examples

Research Papers
~~~~~~~~~~~~~~~

* **Methodology Sections**: Extract research methods, data collection, and analysis approaches
* **Results Sections**: Identify findings, statistical outcomes, and experimental results
* **Discussion Sections**: Extract interpretation, implications, and future research directions
