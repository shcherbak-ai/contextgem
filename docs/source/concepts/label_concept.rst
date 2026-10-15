.. meta::
   :description: How to classify a document or an aspect with predefined labels using LabelConcept, choosing exactly one label (multi-class) or any number of them (multi-label).

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
* **Multi-label**: Selects zero, one, or multiple labels from the predefined set (non-exclusive labels) - used when multiple topics or attributes can apply simultaneously. Returns only applicable labels, or no labels if none apply (the ``extracted_items`` list is then empty).

.. important::
   **For multi-class classification**: Since multi-class classification will always return exactly one label, you should consider including a general "other" label (such as "N/A", "misc", "unspecified", etc.) to handle cases where none of the specific labels apply, unless your labels are broad enough to cover all cases, or you know that the classified content always falls under one of the predefined labels without edge cases. This ensures appropriate classification even when the content doesn't clearly fit into any of the predefined specific categories.


💻 Usage Example
------------------

Here's a basic example of how to use :class:`~contextgem.public.concepts.LabelConcept` for document type classification:

.. literalinclude:: ../../../dev/usage_examples/docs/concepts/label_concept/label_concept.py
   :language: python


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
     - Classification mode. Available values: ``"multi_class"`` (select exactly one label), ``"multi_label"`` (select zero, one, or several labels).
   * - ``model_role``
     - ``str | None``
     - ``None``
     - The role label of the model that extracts the concept when a model group is used: the concept is routed to the model in the group with the same ``role``. With the default ``None``, the concept is routed to the model without a role. Ignored when a single model is used, as it extracts every concept it is given. For more details, see :ref:`model-roles-label`.
   * - ``modality``
     - ``str``
     - ``"text"``
     - The document content the concept is extracted from. Available values: ``"text"``, ``"image"``, ``"multimodal"`` (text and/or images, whichever the document has). ``"image"`` and ``"multimodal"`` are supported for document-level concepts only, and require an LLM that supports vision when the document has images. For more details, see :ref:`concept-modality-label`.
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
   * - ``decision_threshold``
     - ``float | None``
     - ``None``
     - Relevant when the concept is extracted by a decision model (:class:`~contextgem.public.decision_models.DocumentDecisionModel`): the lowest probability (above 0, up to 1) with which a label is selected. Can be set for ``classification_type="multi_label"`` only: with ``"multi_class"``, the most probable label is always selected, and setting the threshold raises a validation error. With the default ``None``, the ``decision_threshold`` of the decision model applies. Ignored when the concept is extracted by an LLM. See :ref:`decision-models-threshold-label`.
   * - ``singular_occurrence``
     - ``bool``
     - ``False``
     - Whether this concept is restricted to having only one extracted item. If ``True``, only a single extracted item will be extracted. This is particularly useful for global document classifications where only one classification result is expected. With a decision model (:class:`~contextgem.public.decision_models.DocumentDecisionModel`), a concept gets at most one item with or without this parameter, unless the text is sent in parts, i.e. ``max_paragraphs_to_analyze_per_call`` is passed to the extraction method: a decision model then returns an item for each part, and rejects a concept with ``singular_occurrence=True`` with ``ValueError`` before any request is sent. See :ref:`sending a long text in parts <decision-models-parts-label>`.
   * - ``custom_data``
     - ``dict``
     - ``{}``
     - Optional. Dictionary for storing any additional data that you want to associate with the concept. This data must be JSON-serializable. This data is not used for extraction but can be useful for custom processing or downstream tasks.


🚀 Advanced Usage
--------------------

🏷️ Multi-Class vs Multi-Label Classification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Choose the classification type based on your use case:

- **Multi-class** (``classification_type="multi_class"``) is ideal for document types, priority levels, and status categories. Example: a document is classified as one of "NDA", "Consultancy Agreement", "Privacy Policy", or "Other" if none of them applies.
- **Multi-label** (``classification_type="multi_label"``) is ideal for content topics, applicable regulations, and feature tags. Example: a document can cover several of the topics "Finance", "Legal", "Technology", or none of them.

Here's an example demonstrating multi-label classification for content topic identification:

.. literalinclude:: ../../../dev/usage_examples/docs/concepts/label_concept/multi_label_classification.py
   :language: python

🔍 References and Justifications for Classification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can configure a :class:`~contextgem.public.concepts.LabelConcept` to include justifications and references to understand classification decisions. This is particularly valuable when dealing with complex documents that might contain elements of multiple document types:

.. literalinclude:: ../../../dev/usage_examples/docs/concepts/label_concept/refs_and_justifications.py
   :language: python

🎯 Document Aspect Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:class:`~contextgem.public.concepts.LabelConcept` can be used to classify extracted :class:`~contextgem.public.aspects.Aspect` instances, providing a powerful way to analyze and categorize specific information that has been extracted from documents. This approach allows you to first extract relevant content using aspects, then apply classification logic to those extracted items.

Here's an example that demonstrates using :class:`~contextgem.public.concepts.LabelConcept` to classify the financial risk level of extracted financial obligations from legal contracts:

.. literalinclude:: ../../../dev/usage_examples/docs/concepts/label_concept/document_aspect_analysis.py
   :language: python

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
   * - ``probabilities``
     - dict[str, float]
     - Probability of each label of the concept, e.g. ``{"NDA": 0.95, "Other": 0.05}`` (only if the concept is extracted by a decision model, ``None`` otherwise; see :ref:`decision-models-probabilities-label`)
   * - ``reference_paragraphs``
     - list[:class:`~contextgem.public.paragraphs.Paragraph`]
     - List of paragraph objects that informed the classification (only if ``add_references=True``)
   * - ``reference_sentences``
     - list[:class:`~contextgem.public.sentences.Sentence`]
     - List of sentence objects that informed the classification (only if ``add_references=True`` and ``reference_depth="sentences"``)

.. note::
   To find the position of a reference paragraph or sentence in the document, e.g. to cite it or to retrieve its surrounding context, see :ref:`locating-paras-and-sents`.


💡 Best Practices
-------------------

Here are some best practices to optimize your use of :class:`~contextgem.public.concepts.LabelConcept`:

- **Choose meaningful labels**: Use clear, distinct labels that cover your classification needs without overlap.
- **Provide clear descriptions**: Explain what each classification represents and when each label should be applied.
- **Consider label granularity**: Balance between too few labels (insufficient precision) and too many labels (classification complexity).
- **Enable justifications**: Use ``add_justifications=True`` to understand and validate classification decisions, especially for complex or ambiguous content.
