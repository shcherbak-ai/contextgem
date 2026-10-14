:og:image: https://contextgem.dev/_static/docs_preview_image_rating_concept.png

.. meta::
   :description: How to rate a document or an aspect on a numerical scale that you define with RatingConcept.

RatingConcept
==============

:class:`~contextgem.public.concepts.RatingConcept` is a specialized concept type that calculates, infers, and derives rating values from documents within a clearly defined numerical scale.


📝 Overview
-------------

:class:`~contextgem.public.concepts.RatingConcept` enables sophisticated rating analysis from documents, allowing you to:

* **Derive implicit ratings**: Calculate ratings based on sentiment analysis, key criteria, or contextual evaluation
* **Generate evaluative scores**: Produce numerical assessments that quantify quality, relevance, or performance
* **Normalize diverse signals**: Convert qualitative assessments into consistent numerical ratings within your defined scale
* **Synthesize overall scores**: Combine multiple factors or opinions into comprehensive rating assessments

This concept type is particularly valuable for generating evaluative information from documents such as:

* Product and service reviews where sentiment must be quantified on a standardized scale
* Performance assessments requiring numerical quality or satisfaction scoring
* Risk evaluations needing severity or probability measurements
* Content analyses where subjective characteristics must be rated objectively


💻 Usage Example
------------------

Here's a simple example of how to use :class:`~contextgem.public.concepts.RatingConcept` to extract a product rating:

.. literalinclude:: ../../../dev/usage_examples/docs/concepts/rating_concept/rating_concept.py
   :language: python


⚙️ Parameters
--------------

When creating a :class:`~contextgem.public.concepts.RatingConcept`, you can specify the following parameters:

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
     - A clear description of what should be evaluated and rated, including the criteria for assigning different values within the rating scale (e.g., "Evaluate product quality based on features, durability, and performance where 1 represents poor quality and 10 represents exceptional quality"). The more specific the description, the more consistent and accurate the ratings will be.
   * - ``rating_scale``
     - ``tuple[int, int]``
     - (Required)
     - Defines the boundaries for valid ratings as a tuple of (start, end) values (e.g., ``(1, 5)`` for a 1-5 star rating, or ``(0, 100)`` for a percentage-based evaluation). This parameter establishes the numerical range within which all ratings must fall, ensuring consistency across evaluations.
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
     - Whether to include justifications for extracted items. Justifications provide explanations of why the LLM assigned specific rating values and the reasoning behind the evaluation, which is especially useful for understanding the factors that influenced the rating. For example, a justification might explain that a smartphone received an 8/10 quality rating based on its premium build materials, advanced camera system, and long battery life, despite lacking expandable storage.
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
     - Whether to include source references for extracted items. References indicate the specific locations in the document that provided information or evidence used to determine the rating. This is particularly useful for understanding which parts of the document influenced the rating assessment, allowing to trace back evaluations to relevant content that supports the numerical value assigned.
   * - ``reference_depth``
     - ``str``
     - ``"paragraphs"``
     - Source reference granularity. Available values: ``"paragraphs"``, ``"sentences"``.
   * - ``singular_occurrence``
     - ``bool``
     - ``False``
     - Whether this concept is restricted to having only one extracted item. If ``True``, only a single rating will be extracted. For rating concepts, this parameter is particularly useful when you want to extract a single overall score (e.g., *"overall product quality"*) rather than identifying multiple ratings throughout the document for different features. This helps distinguish between a global evaluation versus component-specific ratings. Note that with advanced LLMs, this constraint may not be required as they can often infer the appropriate number of ratings to extract from the concept's name, description, and rating context. With a decision model (:class:`~contextgem.public.decision_models.DocumentDecisionModel`), a concept gets at most one item with or without this parameter, unless the text is sent in parts, i.e. ``max_paragraphs_to_analyze_per_call`` is passed to the extraction method: a decision model then returns an item for each part, and rejects a concept with ``singular_occurrence=True`` with ``ValueError`` before any request is sent. See :ref:`sending a long text in parts <decision-models-parts-label>`.
   * - ``custom_data``
     - ``dict``
     - ``{}``
     - Optional. Dictionary for storing any additional data that you want to associate with the concept. This data must be JSON-serializable. This data is not used for extraction but can be useful for custom processing or downstream tasks.


🚀 Advanced Usage
--------------------

🔍 References and Justifications for Extraction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When extracting a :class:`~contextgem.public.concepts.RatingConcept`, it's often useful to include justifications, to understand the reasoning behind the score, and references to the text that informed it:

.. literalinclude:: ../../../dev/usage_examples/docs/concepts/rating_concept/refs_and_justifications.py
   :language: python

⭐⭐ Multiple Rating Categories
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can extract multiple rating categories from a document by creating separate rating concepts:

.. literalinclude:: ../../../dev/usage_examples/docs/concepts/rating_concept/multiple_ratings.py
   :language: python


📊 Extracted Items
--------------------

When a :class:`~contextgem.public.concepts.RatingConcept` is extracted, it is populated with **a list of extracted items** accessible through the ``.extracted_items`` property. Each item is an instance of the ``_IntegerItem`` class with the following attributes:

.. list-table::
   :widths: 20 20 60
   :header-rows: 1

   * - Attribute
     - Type
     - Description
   * - ``value``
     - int
     - The extracted rating value as an integer within the defined rating scale
   * - ``justification``
     - str
     - Explanation of why this rating was extracted (only if ``add_justifications=True``)
   * - ``probabilities``
     - dict[int, float]
     - Probability of each integer of the rating scale, e.g. ``{1: 0.05, 2: 0.25, 3: 0.7}`` (only if the concept is extracted by a decision model, ``None`` otherwise; see :ref:`decision-models-probabilities-label`)
   * - ``reference_paragraphs``
     - list[:class:`~contextgem.public.paragraphs.Paragraph`]
     - List of paragraph objects that influenced the rating determination (only if ``add_references=True``)
   * - ``reference_sentences``
     - list[:class:`~contextgem.public.sentences.Sentence`]
     - List of sentence objects that influenced the rating determination (only if ``add_references=True`` and ``reference_depth="sentences"``)

.. note::
   To find the position of a reference paragraph or sentence in the document, e.g. to cite it or to retrieve its surrounding context, see :ref:`locating-paras-and-sents`.


💡 Best Practices
-------------------

- Create descriptive names for your rating concepts that clearly indicate what is being evaluated (e.g., *"Product Usability Rating"* rather than just *"Rating"*).
- Enhance extraction quality by including clear definitions of what each point on the scale represents in your concept description (e.g., *"1 = poor, 3 = average, 5 = excellent"*).
- Provide specific evaluation criteria in your concept description to guide the LLM's assessment process. For example, when rating software usability, specify that factors like interface intuitiveness, learning curve, and navigation efficiency should be considered.
- Enable justifications (using ``add_justifications=True``) when you need to understand the reasoning behind a rating, which is particularly valuable for evaluations that involve complex criteria where the rationale may not be immediately obvious from the score alone.
- Enable references (using ``add_references=True``) to trace ratings back to specific evidence in the document that informed the evaluation.
- Apply ``singular_occurrence=True`` for concepts that should yield a single comprehensive rating (like an overall product score) rather than multiple ratings throughout the document.
