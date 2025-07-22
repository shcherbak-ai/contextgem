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

:og:image: https://contextgem.dev/_static/docs_preview_image_boolean_concept.png

BooleanConcept
==============

:class:`~contextgem.public.concepts.BooleanConcept` is a specialized concept type that evaluates document content and produces True/False assessments based on specific criteria, conditions, or properties you define.


📝 Overview
-------------

:class:`~contextgem.public.concepts.BooleanConcept` is used when you need to determine if a document contains or satisfies specific attributes, properties, or conditions that can be represented as True or False values, such as:

* **Presence checks**: contains confidential information, includes specific clauses, mentions certain topics
* **Compliance assessments**: meets regulatory requirements, follows specific formatting standards
* **Binary classifications**: is favorable/unfavorable, is complete/incomplete, is approved/rejected


💻 Usage Example
------------------

Here's a simple example of how to use :class:`~contextgem.public.concepts.BooleanConcept` to determine if a document mentions confidential information:

.. literalinclude:: ../../../dev/usage_examples/docs/concepts/boolean_concept/boolean_concept.py
   :language: python

.. raw:: html

   <a target="_blank" href="https://colab.research.google.com/github/shcherbak-ai/contextgem/blob/main/dev/notebooks/docs/concepts/boolean_concept/boolean_concept.ipynb">
     <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
   </a>


⚙️ Parameters
--------------

When creating a :class:`~contextgem.public.concepts.BooleanConcept`, you can specify the following parameters:

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
     - A clear description of what condition or property the concept evaluates and the criteria for determining true or false values
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
     - Whether to include source references for extracted items. References indicate the specific locations in the document where evidence supporting the boolean determination was found, helping to trace back the true/false value to relevant content that influenced the decision.
   * - ``reference_depth``
     - ``str``
     - ``"paragraphs"``
     - Source reference granularity. Available values: ``"paragraphs"``, ``"sentences"``.
   * - ``singular_occurrence``
     - ``bool``
     - ``False``
     - Whether this concept is restricted to having only one extracted item. If ``True``, only a single extracted item will be extracted. For boolean concepts, this parameter is particularly useful when you want to make a single true/false determination about the entire document (e.g., "contains confidential information") or a unique determination about a specific aspect (e.g., "is the payment schedule finalized"). This helps distinguish between evaluating overall document properties versus identifying multiple instances where a condition might be true/false. Note that with advanced LLMs, this constraint may not be required as they can often infer the appropriate number of items to extract from the concept's name, description, and type (e.g., "contains confidential information" vs "compliance violations").
   * - ``custom_data``
     - ``dict``
     - ``{}``
     - Optional. Dictionary for storing any additional data that you want to associate with the concept. This data must be JSON-serializable. This data is not used for extraction but can be useful for custom processing or downstream tasks.


🚀 Advanced Usage
--------------------

🔍 References and Justifications for Extraction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can configure a :class:`~contextgem.public.concepts.BooleanConcept` to include justifications and references. Justifications help explain the reasoning behind true/false determinations, while references point to the specific parts of the document that influenced the decision:

.. literalinclude:: ../../../dev/usage_examples/docs/concepts/boolean_concept/refs_and_justifications.py
   :language: python

.. raw:: html

   <a target="_blank" href="https://colab.research.google.com/github/shcherbak-ai/contextgem/blob/main/dev/notebooks/docs/concepts/boolean_concept/refs_and_justifications.ipynb">
     <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
   </a>


📊 Extracted Items
--------------------

When a :class:`~contextgem.public.concepts.BooleanConcept` is extracted, it is populated with **a list of extracted items** accessible through the ``.extracted_items`` property. Each item is an instance of the ``_BooleanItem`` class with the following attributes:

.. list-table::
   :widths: 20 20 60
   :header-rows: 1

   * - Attribute
     - Type
     - Description
   * - ``value``
     - bool
     - The extracted boolean value (True or False)
   * - ``justification``
     - str
     - Explanation of why this boolean value was determined (only if ``add_justifications=True``)
   * - ``reference_paragraphs``
     - list[:class:`~contextgem.public.paragraphs.Paragraph`]
     - List of paragraph objects that influenced the boolean determination (only if ``add_references=True``)
   * - ``reference_sentences``
     - list[:class:`~contextgem.public.sentences.Sentence`]
     - List of sentence objects that influenced the boolean determination (only if ``add_references=True`` and ``reference_depth="sentences"``)


💡 Best Practices
-------------------

Here are some best practices to optimize your use of :class:`~contextgem.public.concepts.BooleanConcept`:

- Provide a clear and specific description that helps the LLM understand exactly what condition to evaluate, using precise and unambiguous language in your concept names and descriptions. Since boolean concepts yield true/false values, focus on describing what criteria should be used to make the determination (e.g., *"whether the document mentions specific compliance requirements"* rather than just *"compliance requirements"*). Avoid vague terms that could be interpreted multiple ways—for example, use *"contains legally binding obligations"* instead of *"contains important content"* to ensure consistent and accurate determinations.
- Break down complex conditions into multiple simpler boolean concepts when appropriate. Instead of one concept checking *"document is complete and compliant and approved,"* consider separate concepts for each condition. This provides more granular insights and makes it easier to identify specific issues when any condition fails.
- Enable justifications (using ``add_justifications=True``) when you need to understand the reasoning behind the LLM's true/false determination.
- Enable references (using ``add_references=True``) when you need to trace back to specific parts of the document that influenced the boolean decision or verify the evidence used to make the determination.
- Use ``singular_occurrence=True`` to enforce only a single boolean determination for the entire document. This is particularly useful for concepts that should yield a single true/false answer, such as *"contains confidential information"* or *"is compliant with regulations,"* rather than identifying multiple instances where the condition might be true or false throughout the document.
