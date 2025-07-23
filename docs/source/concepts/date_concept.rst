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

:og:image: https://contextgem.dev/_static/docs_preview_image_date_concept.png

DateConcept
=============

:class:`~contextgem.public.concepts.DateConcept` is a specialized concept type that extracts, interprets, and processes date information from documents, returning standardized ``datetime.date`` objects.


📝 Overview
-------------

:class:`~contextgem.public.concepts.DateConcept` is used when you need to extract date information from documents, allowing you to:

* **Extract explicit dates**: Identify dates that are directly mentioned in various formats (e.g., "January 15, 2025", "15/01/2025", "2025-01-15")
* **Infer implicit dates**: Deduce dates from contextual information (e.g., "next Monday", "two weeks from signing", "the following quarter")
* **Calculate derived dates**: Determine dates based on other temporal references (e.g., "30 days after delivery", "the fiscal year ending")
* **Normalize date representations**: Convert various date formats into standardized Python ``datetime.date`` objects for consistent processing

This concept type is particularly valuable for extracting temporal information from documents such as:

* Contract effective dates, expiration dates, and renewal periods
* Report publication dates and data collection periods
* Event scheduling information and deadline specifications
* Historical dates and chronological sequences


💻 Usage Example
------------------

Here's a simple example of how to use :class:`~contextgem.public.concepts.DateConcept` to extract a publication date from a document:

.. literalinclude:: ../../../dev/usage_examples/docs/concepts/date_concept/date_concept.py
   :language: python

.. raw:: html

   <a target="_blank" href="https://colab.research.google.com/github/shcherbak-ai/contextgem/blob/main/dev/notebooks/docs/concepts/date_concept/date_concept.ipynb">
     <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
   </a>


⚙️ Parameters
--------------

When creating a :class:`~contextgem.public.concepts.DateConcept`, you can specify the following parameters:

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
     - A clear description of what date information to extract, which can include explicit dates to find, implicit dates to infer, or temporal relationships to identify. For date concepts, be specific about the exact date information sought (e.g., *"the contract signing date"* rather than just *"dates in the document"*) to ensure consistent and accurate extractions.
   * - ``llm_role``
     - ``str``
     - ``"extractor_text"``
     - The role of the LLM responsible for extracting the concept. Available values: ``"extractor_text"``, ``"reasoner_text"``, ``"extractor_vision"``, ``"reasoner_vision"``. For more details, see :ref:`llm-roles-label`.
   * - ``add_justifications``
     - ``bool``
     - ``False``
     - Whether to include justifications for extracted items. Justifications provide explanations of why specific dates were extracted, which is especially valuable when dates are inferred from contextual clues (e.g., *"next quarter"* or *"30 days after signing"*) or when resolving ambiguous date references in the document.
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
     - Whether to include source references for extracted items. References indicate the specific locations in the document where date information was found, derived, or inferred from. This is particularly useful for tracing dates back to their original context, understanding how relative dates were calculated (e.g., *"30 days after delivery"*), or verifying how the system resolved ambiguous temporal references (e.g., *"next fiscal year"*).
   * - ``reference_depth``
     - ``str``
     - ``"paragraphs"``
     - Source reference granularity. Available values: ``"paragraphs"``, ``"sentences"``.
   * - ``singular_occurrence``
     - ``bool``
     - ``False``
     - Whether this concept is restricted to having only one extracted item. If ``True``, only a single date will be extracted. For date concepts, this parameter is particularly useful when you want to extract a specific, unique date in the document (e.g., *"publication date"* or *"contract signing date"*) rather than identifying multiple dates throughout the document. Note that with advanced LLMs, this constraint may not be required as they can often infer the appropriate cardinality from the concept's name, description, and type.
   * - ``custom_data``
     - ``dict``
     - ``{}``
     - Optional. Dictionary for storing any additional data that you want to associate with the concept. This data must be JSON-serializable. This data is not used for extraction but can be useful for custom processing or downstream tasks.


🚀 Advanced Usage
--------------------

🔍 References and Justifications for Extraction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can configure a :class:`~contextgem.public.concepts.DateConcept` to include justifications and references. Justifications help explain the reasoning behind extracted dates, especially for complex or inferred temporal information (like dates derived from expressions such as *"30 days after delivery"* or *"next fiscal year"*), while references point to the specific parts of the document that contained the date information or based on which date information was inferred:

.. literalinclude:: ../../../dev/usage_examples/docs/concepts/date_concept/refs_and_justifications.py
   :language: python

.. raw:: html

   <a target="_blank" href="https://colab.research.google.com/github/shcherbak-ai/contextgem/blob/main/dev/notebooks/docs/concepts/date_concept/refs_and_justifications.ipynb">
     <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
   </a>


📊 Extracted Items
--------------------

When a :class:`~contextgem.public.concepts.DateConcept` is extracted, it is populated with **a list of extracted items** accessible through the ``.extracted_items`` property. Each item is an instance of the ``_DateItem`` class with the following attributes:

.. list-table::
   :widths: 20 20 60
   :header-rows: 1

   * - Attribute
     - Type
     - Description
   * - ``value``
     - datetime.date
     - The extracted date as a Python ``datetime.date`` object
   * - ``justification``
     - str
     - Explanation of why this date was extracted (only if ``add_justifications=True``)
   * - ``reference_paragraphs``
     - list[:class:`~contextgem.public.paragraphs.Paragraph`]
     - List of paragraph objects where the date was found or from which it was calculated, derived, or inferred (only if ``add_references=True``)
   * - ``reference_sentences``
     - list[:class:`~contextgem.public.sentences.Sentence`]
     - List of sentence objects where the date was found or from which it was calculated, derived, or inferred (only if ``add_references=True`` and ``reference_depth="sentences"``)


💡 Best Practices
-------------------

Here are some best practices to optimize your use of :class:`~contextgem.public.concepts.DateConcept`:

- Provide a clear and specific description that helps the LLM understand exactly what date to extract, using precise and unambiguous language (e.g., *"contract signing date"* rather than just *"date"*).
- For dates that require interpretation or calculation (like *"30 days after delivery"* or *"end of next fiscal year"*), include these requirements explicitly in your description to ensure the LLM performs the necessary temporal reasoning.
- Break down complex date extractions into multiple simpler date concepts when appropriate. Instead of one concept extracting *"all contract dates,"* consider separate concepts for *"contract signing date,"* *"effective date,"* and *"termination date."*
- Enable justifications (using ``add_justifications=True``) when you need to understand the reasoning behind date calculations or extractions, especially for relative or inferred dates.
- Enable references (using ``add_references=True``) when you need to trace back to specific parts of the document that contained the original date information or where dates were calculated from (e.g., deriving a project completion date from a start date plus duration information).
- Use ``singular_occurrence=True`` to enforce only a single date extraction. This is particularly useful for concepts that should yield a unique calculated date, such as *"project completion deadline"* where multiple timeline elements need to be synthesized into a single target date, or when multiple date mentions actually refer to the same event.
- Leverage the returned Python ``datetime.date`` objects for direct integration with date-based calculations, comparisons, or formatting in your application logic.