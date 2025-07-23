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

:og:image: https://contextgem.dev/_static/docs_preview_image_string_concept.png

StringConcept
==============

:class:`~contextgem.public.concepts.StringConcept` is a versatile concept type in ContextGem that extracts text-based information from documents, ranging from simple data fields to complex analytical insights.


📝 Overview
-------------

:class:`~contextgem.public.concepts.StringConcept` is used when you need to extract text values from documents, including:

* **Simple fields**: names, titles, descriptions, identifiers
* **Complex analyses**: conclusions, assessments, recommendations, summaries
* **Detected elements**: anomalies, patterns, key findings, critical insights

This concept type offers flexibility to extract both factual information and interpretive content that requires advanced understanding.


💻 Usage Example
------------------

Here's a simple example of how to use :class:`~contextgem.public.concepts.StringConcept` to extract a person's name from a document:

.. literalinclude:: ../../../dev/usage_examples/docs/concepts/string_concept/string_concept.py
   :language: python

.. raw:: html

   <a target="_blank" href="https://colab.research.google.com/github/shcherbak-ai/contextgem/blob/main/dev/notebooks/docs/concepts/string_concept/string_concept.ipynb">
     <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
   </a>


⚙️ Parameters
--------------

When creating a :class:`~contextgem.public.concepts.StringConcept`, you can specify the following parameters:

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
     - A unique name identifier for the concept
   * - ``description``
     - ``str``
     - (Required)
     - A clear description of what the concept represents and what should be extracted
   * - ``examples``
     - ``list[StringExample]``
     - ``[]``
     - Optional. Example values that help the LLM better understand what to extract and the expected format (e.g., *"Party Name (Role)"* format for contract parties). This additional guidance helps improve extraction accuracy and consistency.
   * - ``llm_role``
     - ``str``
     - ``"extractor_text"``
     - The role of the LLM responsible for extracting the concept. Available values: ``"extractor_text"``, ``"reasoner_text"``, ``"extractor_vision"``, ``"reasoner_vision"``. For more details, see :ref:`llm-roles-label`.
   * - ``add_justifications``
     - ``bool``
     - ``False``
     - Whether to include justifications for extracted items. Justifications provide explanations of why the LLM extracted specific values and the reasoning behind the extraction, which is especially useful for complex extractions or when debugging results.
   * - ``justification_depth``
     - ``str``
     - ``"brief"``
     - Justification detail level. Available values: ``"brief"``, ``"balanced"``, ``"comprehensive"``.
   * - ``justification_max_sents``
     - ``int``
     - ``2``
     - Maximum sentences in a justification.
   * - ``add_references``
     - ``bool``
     - ``False``
     - Whether to include source references for extracted items. References indicate the specific locations in the document where the information was either directly found or from which it was inferred, helping to trace back extracted values to their source content even when the extraction involves reasoning or interpretation.
   * - ``reference_depth``
     - ``str``
     - ``"paragraphs"``
     - Source reference granularity. Available values: ``"paragraphs"``, ``"sentences"``.
   * - ``singular_occurrence``
     - ``bool``
     - ``False``
     - Whether this concept is restricted to having only one extracted item. If ``True``, only a single extracted item will be extracted. This is particularly relevant when it might be unclear for the LLM whether to focus on the concept as a single item or extract multiple items. For example, when extracting the total amount of payments in a contract, where payments might be mentioned in different parts of the document but you only want the final total. Note that with advanced LLMs, this constraint may not be strictly required as they can often infer the appropriate number of items to extract from the concept's name, description, and type (e.g., "document title" vs "key findings").
   * - ``custom_data``
     - ``dict``
     - ``{}``
     - Optional. Dictionary for storing any additional data that you want to associate with the concept. This data must be JSON-serializable. This data is not used for extraction but can be useful for custom processing or downstream tasks.


🚀 Advanced Usage
--------------------

✏️ Adding Examples
~~~~~~~~~~~~~~~~~~~~

You can add examples to improve the extraction accuracy and set the expected format for a :class:`~contextgem.public.concepts.StringConcept`:

.. literalinclude:: ../../../dev/usage_examples/docs/concepts/string_concept/adding_examples.py
   :language: python

.. raw:: html

   <a target="_blank" href="https://colab.research.google.com/github/shcherbak-ai/contextgem/blob/main/dev/notebooks/docs/concepts/string_concept/adding_examples.ipynb">
     <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
   </a>

🔍 References and Justifications for Extraction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can configure a :class:`~contextgem.public.concepts.StringConcept` to include justifications and references. Justifications help explain the reasoning behind extracted values, especially for complex or inferred information like conclusions or assessments, while references point to the specific parts of the document that informed the extraction:

.. literalinclude:: ../../../dev/usage_examples/docs/concepts/string_concept/refs_and_justifications.py
   :language: python

.. raw:: html

   <a target="_blank" href="https://colab.research.google.com/github/shcherbak-ai/contextgem/blob/main/dev/notebooks/docs/concepts/string_concept/refs_and_justifications.ipynb">
     <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
   </a>


📊 Extracted Items
--------------------

When a :class:`~contextgem.public.concepts.StringConcept` is extracted, it is populated with **a list of extracted items** accessible through the ``.extracted_items`` property. Each item is an instance of the ``_StringItem`` class with the following attributes:

.. list-table::
   :widths: 20 20 60
   :header-rows: 1

   * - Attribute
     - Type
     - Description
   * - ``value``
     - str
     - The extracted text string
   * - ``justification``
     - str
     - Explanation of why this string was extracted (only if ``add_justifications=True``)
   * - ``reference_paragraphs``
     - list[:class:`~contextgem.public.paragraphs.Paragraph`]
     - List of paragraph objects that informed the extraction (only if ``add_references=True``)
   * - ``reference_sentences``
     - list[:class:`~contextgem.public.sentences.Sentence`]
     - List of sentence objects that informed the extraction (only if ``add_references=True`` and ``reference_depth="sentences"``)


💡 Best Practices
-------------------

Here are some best practices to optimize your use of :class:`~contextgem.public.concepts.StringConcept`:

- Provide a clear and specific description that helps the LLM understand exactly what to extract.
- Include examples (using :class:`~contextgem.public.examples.StringExample`) to improve extraction accuracy and demonstrate the expected format (e.g., *"Party Name (Role)"* for contract parties or *"Revenue: $X million"* for financial figures).
- Enable justifications (using ``add_justifications=True``) when you need to see why the LLM extracted certain values.
- Enable references (using ``add_references=True``) when you need to trace back to where in the document the information was found or understand what evidence informed extracted values (especially for inferred information).
- When relevant, enforce only a single item extraction (using ``singular_occurrence=True``). This is particularly relevant when it might be unclear for the LLM whether to focus on the concept as a single item or extract multiple items. For example, when extracting the total amount of payments in a contract, where payments might be mentioned in different parts of the document but you only want the final total.
