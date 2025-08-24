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

* **Multi-class**: Always selects exactly one label from the predefined set (mutually exclusive labels) - used for classifying the content into a single type or category. A label is always returned, even if none perfectly fit the content.
* **Multi-label**: Selects zero, one, or multiple labels from the predefined set (non-exclusive labels) - used when multiple topics or attributes can apply simultaneously. Returns only applicable labels, or no labels if none apply.

.. note::
   **For multi-label classification**: When none of the predefined labels apply to the content being classified, no extracted items will be returned for the concept (empty ``extracted_items`` list). This ensures that only applicable labels are selected.
   
   **For multi-class classification**: A label is always returned, as this classification type requires selecting the best-fitting option from the predefined set, even if none perfectly match the content.

.. important::
   **For multi-class classification**: Since multi-class classification should always return exactly one label, you should consider including a general "other" label (such as "N/A", "misc", "unspecified", etc.) to handle cases where none of the specific labels apply, unless your labels are broad enough to cover all cases, or you know that the classified content always falls under one of the predefined labels without edge cases. This ensures appropriate classification even when the content doesn't clearly fit into any of the predefined specific categories.


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
     - A clear description of what the concept represents and how classification should be performed
   * - ``labels``
     - ``list[str]``
     - (Required)
     - List of predefined labels for classification. Must contain at least 2 unique labels
   * - ``classification_type``
     - ``str``
     - ``"multi_class"``
     - Classification mode. Available values: ``"multi_class"`` (select exactly one label), ``"multi_label"`` (select one or more labels).
   * - ``llm_role``
     - ``str``
     - ``"extractor_text"``
     - The role of the LLM responsible for extracting the concept. Available values: ``"extractor_text"``, ``"reasoner_text"``, ``"extractor_vision"``, ``"reasoner_vision"``, ``"extractor_multimodal"``, ``"reasoner_multimodal"``. For more details, see :ref:`llm-roles-label`.
   * - ``add_justifications``
     - ``bool``
     - ``False``
     - Whether to include justifications for extracted items. Justifications provide explanations of why specific labels were selected and the reasoning behind the classification decision.
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
     - Whether to include source references for extracted items. References indicate the specific locations in the document that informed the classification decision.
   * - ``reference_depth``
     - ``str``
     - ``"paragraphs"``
     - Source reference granularity. Available values: ``"paragraphs"``, ``"sentences"``.
   * - ``singular_occurrence``
     - ``bool``
     - ``False``
     - Whether this concept is restricted to having only one extracted item. If ``True``, only a single extracted item will be extracted. This is particularly useful for global document classifications where only one classification result is expected.
   * - ``custom_data``
     - ``dict``
     - ``{}``
     - Optional. Dictionary for storing any additional data that you want to associate with the concept. This data must be JSON-serializable. This data is not used for extraction but can be useful for custom processing or downstream tasks.


🚀 Advanced Usage
--------------------

🏷️ Multi-Class vs Multi-Label Classification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Choose the appropriate classification type based on your use case:

**Multi-Class Classification** (``classification_type="multi_class"``):

- Always selects exactly one label from the predefined set (mutually exclusive labels)
- A label is always returned, even if none perfectly fit the content
- Ideal for: document types, priority levels, status categories
- Example: A document must be classified as one type: "NDA", "Consultancy Agreement", or "Privacy Policy" (or "Other" if none apply)

**Multi-Label Classification** (``classification_type="multi_label"``):

- Selects zero, one, or multiple labels from the predefined set (non-exclusive labels)
- Returns only applicable labels; can return no labels if none apply
- Ideal for: content topics, applicable regulations, feature tags
- Example: A document can cover multiple topics: "Finance", "Legal", "Technology", or none of these topics

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
- **For multi-class classification**: Consider including a general "other" label (like "Other", "N/A", "Mixed", etc.) since a label is always returned, even when none of the specific labels perfectly fit the content, unless your labels are broad enough to cover all cases, or you know that the classified content always falls under one of the predefined labels without edge cases.
- **For multi-label classification**: Design your workflow to handle cases where none of the predefined labels apply (resulting in empty ``extracted_items``), as this classification type can return zero labels.
- **Use appropriate classification type**: Set ``classification_type="multi_class"`` for mutually exclusive categories where exactly one choice is required, ``classification_type="multi_label"`` for potentially overlapping attributes where zero, one, or multiple labels can apply.
- **Enable justifications**: Use ``add_justifications=True`` to understand and validate classification decisions, especially for complex or ambiguous content.