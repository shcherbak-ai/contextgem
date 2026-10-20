.. meta::
   :description: How to extract aspects, the sections and topics of a document that you define, with sub-aspects, concepts and justifications, and how to read the extracted paragraphs or sentences.

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

Aspect with Sub-Aspects
~~~~~~~~~~~~~~~~~~~~~~~~~

Breaking down complex topics into components:

.. literalinclude:: ../../../dev/usage_examples/docs/aspects/aspect_with_sub_aspects.py
   :language: python


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
     - *Optional*. List of concepts associated with the aspect for detailed data extraction within the aspect's scope. Aspects are extracted from text, so only concepts with the ``"text"`` modality (the default) can be assigned to an aspect. See supported concept types in :doc:`../concepts/supported_concepts`.
   * - ``model_role``
     - ``str | None``
     - ``None``
     - The role label of the model that extracts the aspect when a model group is used: the aspect is routed to the model in the group with the same ``role``. With the default ``None``, the aspect is routed to the model without a role. Ignored when a single model is used, as it extracts every aspect it is given. For more details, see :ref:`model-roles-label`.
   * - ``reference_depth``
     - ``str``
     - ``"paragraphs"``
     - The unit of text that is extracted. Available values: ``"paragraphs"``, ``"sentences"``. With ``"paragraphs"``, each extracted item is a paragraph of the document, which is referenced in the ``reference_paragraphs`` of the item. With ``"sentences"``, each extracted item is a sentence, which is referenced in ``reference_sentences``, with its paragraph in ``reference_paragraphs``.
   * - ``add_justifications``
     - ``bool``
     - ``False``
     - Whether the LLM will output justification for each extracted item. Justifications provide valuable insights into why specific text segments were extracted for the aspect, helping you understand the LLM's reasoning, verify extraction accuracy, and debug unexpected results. This is particularly useful when working with complex aspects. A decision model provides no justifications, and rejects ``True`` with ``ValueError`` before any request is sent (see :ref:`decision-models-supported-label`).
   * - ``justification_depth``
     - ``str``
     - ``"brief"``
     - The level of detail for justifications. Available values: ``"brief"``, ``"balanced"``, ``"comprehensive"``.
   * - ``justification_max_sents``
     - ``int``
     - ``2``
     - Maximum number of sentences in a justification.
   * - ``decision_threshold``
     - ``float | None``
     - ``None``
     - Relevant when the aspect is extracted by a decision model (:class:`~contextgem.public.decision_models.DocumentDecisionModel`): the lowest probability (above 0, up to 1) with which a paragraph is taken as belonging to the aspect, and, with ``reference_depth="sentences"``, a sentence of such a paragraph. With the default ``None``, the ``decision_threshold`` of the decision model applies. Not passed on to the sub-aspects. Ignored when the aspect is extracted by an LLM. See :ref:`decision-models-threshold-label`.
   * - ``custom_data``
     - ``dict``
     - ``{}``
     - *Optional*. Dictionary for storing any additional data that you want to associate with the aspect. This data must be JSON-serializable. It is not used for extraction.


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
     - The text of the extracted paragraph, or of the extracted sentence with ``reference_depth="sentences"``
   * - ``justification``
     - str
     - Explanation of why this text segment was identified as relevant to the aspect (only if ``add_justifications=True``)
   * - ``probabilities``
     - dict[bool, float]
     - Probabilities that the paragraph, or the sentence with ``reference_depth="sentences"``, belongs to the aspect (``True``) and that it does not (``False``), e.g. ``{True: 0.91, False: 0.09}`` (only if the aspect is extracted by a decision model, ``None`` otherwise; see :ref:`decision-models-aspects-label`)
   * - ``reference_paragraphs``
     - list[:class:`~contextgem.public.paragraphs.Paragraph`]
     - List of paragraph objects that contain the extracted aspect content (always populated for aspect's extracted items)
   * - ``reference_sentences``
     - list[:class:`~contextgem.public.sentences.Sentence`]
     - List of sentence objects that contain the extracted aspect content (only if ``reference_depth="sentences"``)
   * - ``reference_page_numbers``
     - list[int]
     - Numbers of the pages that the reference paragraphs and sentences are on, e.g. in a document created from a file converted with Docling (see :doc:`../documents/docling`). Empty if the references have no page numbers

.. note::
   To find the position of a reference paragraph or sentence in the document, e.g. to cite it or to retrieve its surrounding context, see :ref:`locating-paras-and-sents`.


🚀 Advanced Usage
-------------------

Aspects with Concepts
~~~~~~~~~~~~~~~~~~~~~~

Combining aspect extraction with detailed concept extraction:

.. literalinclude:: ../../../dev/usage_examples/docs/aspects/aspect_with_concepts.py
   :language: python

Complex Hierarchical Structure
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Creating a comprehensive document analysis structure with aspects, sub-aspects and concepts:

.. literalinclude:: ../../../dev/usage_examples/docs/aspects/complex_hierarchy.py
   :language: python

Justifications for Extraction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

With ``add_justifications=True``, each extracted item includes an explanation of why the text segment was considered a part of the aspect:

.. literalinclude:: ../../../dev/usage_examples/docs/aspects/aspect_with_justifications.py
   :language: python


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
* **Reference tracking**: Extracted items of aspects always reference their paragraphs. For the concepts of an aspect, set ``add_references=True`` to trace the extracted data back to its source


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
* **Key Performance Indicators**: Identify the sections that report metrics like EBITDA margins, debt-to-equity ratios, and year-over-year percentage changes

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
