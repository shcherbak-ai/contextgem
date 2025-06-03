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

:og:image: https://contextgem.dev/_static/docs_preview_image_label_concept.png

LabelConcept
=============

:class:`~contextgem.public.concepts.LabelConcept` is a classification concept type in ContextGem that categorizes documents or content using predefined labels, supporting both single-label and multi-label classification approaches.


🏷️ Overview
-------------

:class:`~contextgem.public.concepts.LabelConcept` is used when you need to classify content into predefined categories, including:

* **Document classification**: contract types, document categories, legal classifications
* **Content categorization**: topics, themes, subjects, areas of focus
* **Quality assessment**: compliance levels, risk categories, priority levels
* **Multi-faceted tagging**: multiple applicable labels for comprehensive classification

This concept type supports two classification modes:

* **Multi-class**: Select exactly one label from the predefined set (mutually exclusive labels) - used for classifying the content into a single type or category
* **Multi-label**: Select one or more labels from the predefined set (non-exclusive labels) - used when multiple topics or attributes can apply simultaneously

.. note::
   When none of the predefined labels apply to the content being classified, no extracted items will be returned for the concept (empty ``extracted_items`` list). This ensures that only valid, predefined labels are selected and prevents forced classification when no appropriate label exists.


💻 Usage Example
------------------

Here's a basic example of how to use :class:`~contextgem.public.concepts.LabelConcept` for document type classification:

.. literalinclude:: ../../../dev/usage_examples/docs/concepts/label_concept/label_concept.py
   :language: python

.. raw:: html

   <a target="_blank" href="https://colab.research.google.com/github/shcherbak-ai/contextgem/blob/main/dev/notebooks/docs/concepts/label_concept/label_concept.ipynb">
     <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
   </a>


⚙️ Parameters
--------------

When creating a :class:`~contextgem.public.concepts.LabelConcept`, you can specify the following parameters:

.. list-table::
   :widths: 20 20 60
   :header-rows: 1

   * - Parameter
     - Type
     - Description
   * - ``name``
     - str
     - A unique name identifier for the concept
   * - ``description``
     - str
     - A clear description of what the concept represents and how classification should be performed
   * - ``labels``
     - list[str]
     - List of predefined labels for classification. Must contain at least 2 unique labels
   * - ``classification_type``
     - str
     - Classification mode. Available values: ``"multi_class"`` (select exactly one label), ``"multi_label"`` (select one or more labels). Defaults to ``"multi_class"``
   * - ``llm_role``
     - str
     - The role of the LLM responsible for extracting the concept. Available values: ``"extractor_text"``, ``"reasoner_text"``, ``"extractor_vision"``, ``"reasoner_vision"``. Defaults to ``"extractor_text"``. For more details, see :ref:`llm-roles-label`.
   * - ``add_justifications``
     - bool
     - Whether to include justifications for extracted items (defaults to ``False``). Justifications provide explanations of why specific labels were selected and the reasoning behind the classification decision.
   * - ``justification_depth``
     - str
     - Justification detail level. Available values: ``"brief"``, ``"balanced"``, ``"comprehensive"``. Defaults to ``"brief"``
   * - ``justification_max_sents``
     - int
     - Maximum sentences in a justification (defaults to ``2``)
   * - ``add_references``
     - bool
     - Whether to include source references for extracted items (defaults to ``False``). References indicate the specific locations in the document that informed the classification decision.
   * - ``reference_depth``
     - str
     - Source reference granularity. Available values: ``"paragraphs"``, ``"sentences"``. Defaults to ``"paragraphs"``
   * - ``singular_occurrence``
     - bool
     - Whether this concept is restricted to having only one extracted item. If ``True``, only a single extracted item will be extracted. Defaults to ``False`` (multiple extracted items are allowed). This is particularly useful for global document classifications where only one classification result is expected.
   * - ``custom_data``
     - dict
     - Optional. Dictionary for storing any additional data that you want to associate with the concept. This data must be JSON-serializable. This data is not used for extraction but can be useful for custom processing or downstream tasks. Defaults to an empty dictionary.


🚀 Advanced Usage
--------------------

🏷️ Multi-Class vs Multi-Label Classification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Choose the appropriate classification type based on your use case:

**Multi-Class Classification** (``classification_type="multi_class"``):
- Select exactly one label from the predefined set (mutually exclusive labels)
- Ideal for: document types, priority levels, status categories
- Example: A document can only be one type: "NDA", "Consultancy Agreement", or "Privacy Policy"

**Multi-Label Classification** (``classification_type="multi_label"``):
- Select one or more labels from the predefined set (non-exclusive labels)
- Ideal for: content topics, applicable regulations, feature tags
- Example: A document can cover multiple topics: "Finance", "Legal", "Technology"

Here's an example demonstrating multi-label classification for content topic identification:

.. literalinclude:: ../../../dev/usage_examples/docs/concepts/label_concept/multi_label_classification.py
   :language: python

.. raw:: html

   <a target="_blank" href="https://colab.research.google.com/github/shcherbak-ai/contextgem/blob/main/dev/notebooks/docs/concepts/label_concept/multi_label_classification.ipynb">
     <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
   </a>

🔍 References and Justifications for Classification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can configure a :class:`~contextgem.public.concepts.LabelConcept` to include justifications and references to understand classification decisions. This is particularly valuable when dealing with complex documents that might contain elements of multiple document types:

.. literalinclude:: ../../../dev/usage_examples/docs/concepts/label_concept/refs_and_justifications.py
   :language: python

.. raw:: html

   <a target="_blank" href="https://colab.research.google.com/github/shcherbak-ai/contextgem/blob/main/dev/notebooks/docs/concepts/label_concept/refs_and_justifications.ipynb">
     <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
   </a>

🎯 Document Aspect Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:class:`~contextgem.public.concepts.LabelConcept` can be used to classify extracted :class:`~contextgem.public.aspects.Aspect` instances, providing a powerful way to analyze and categorize specific information that has been extracted from documents. This approach allows you to first extract relevant content using aspects, then apply classification logic to those extracted items.

Here's an example that demonstrates using :class:`~contextgem.public.concepts.LabelConcept` to classify the financial risk level of extracted financial obligations from legal contracts:

.. literalinclude:: ../../../dev/usage_examples/docs/concepts/label_concept/document_aspect_analysis.py
   :language: python

.. raw:: html

   <a target="_blank" href="https://colab.research.google.com/github/shcherbak-ai/contextgem/blob/main/dev/notebooks/docs/concepts/label_concept/document_aspect_analysis.ipynb">
     <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
   </a>

📊 Extracted Items
--------------------

When a :class:`~contextgem.public.concepts.LabelConcept` is extracted, it is populated with **a list of extracted items** accessible through the ``.extracted_items`` property. Each item is an instance of the ``_LabelItem`` class with the following attributes:

.. list-table::
   :widths: 20 20 60
   :header-rows: 1

   * - Attribute
     - Type
     - Description
   * - ``value``
     - list[str]
     - List of selected labels (always a list for API consistency, even for multi-class with single selection)
   * - ``justification``
     - str
     - Explanation of why these labels were selected (only if ``add_justifications=True``)
   * - ``reference_paragraphs``
     - list[:class:`~contextgem.public.paragraphs.Paragraph`]
     - List of paragraph objects that informed the classification (only if ``add_references=True``)
   * - ``reference_sentences``
     - list[:class:`~contextgem.public.sentences.Sentence`]
     - List of sentence objects that informed the classification (only if ``add_references=True`` and ``reference_depth="sentences"``)


💡 Best Practices
-------------------

Here are some best practices to optimize your use of :class:`~contextgem.public.concepts.LabelConcept`:

- **Choose meaningful labels**: Use clear, distinct labels that cover your classification needs without overlap.
- **Provide clear descriptions**: Explain what each classification represents and when each label should be applied.
- **Consider label granularity**: Balance between too few labels (insufficient precision) and too many labels (classification complexity).
- **Include edge cases**: Consider adding labels like "Other" or "Mixed" for content that doesn't fit standard categories.
- **Use appropriate classification type**: Set ``classification_type="multi_class"`` for mutually exclusive categories, ``classification_type="multi_label"`` for potentially overlapping attributes.
- **Enable justifications**: Use ``add_justifications=True`` to understand and validate classification decisions, especially for complex or ambiguous content.
- **Handle empty results**: Design your workflow to handle cases where none of the predefined labels apply (resulting in empty ``extracted_items``). 