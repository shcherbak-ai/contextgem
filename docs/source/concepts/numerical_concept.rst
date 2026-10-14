:og:image: https://contextgem.dev/_static/docs_preview_image_numerical_concept.png

.. meta::
   :description: How to extract or calculate numbers (integers, floats or both) from a document or an aspect with NumericalConcept.

NumericalConcept
=================

:class:`~contextgem.public.concepts.NumericalConcept` is a specialized concept type that extracts, calculates, or derives numerical values (integers, floats, or both) from document content.


📝 Overview
-------------

:class:`~contextgem.public.concepts.NumericalConcept` enables powerful numerical data extraction and analysis from documents, such as:

* **Direct extraction**: retrieving explicitly stated values like prices, percentages, or measurements
* **Calculated values**: computing sums, averages, growth rates, or other derived metrics
* **Quantitative assessments**: determining counts, frequencies, totals, or numerical scores


💻 Usage Example
------------------

Here's a simple example of how to use :class:`~contextgem.public.concepts.NumericalConcept` to extract a price from a document:

.. literalinclude:: ../../../dev/usage_examples/docs/concepts/numerical_concept/numerical_concept.py
   :language: python

⚙️ Parameters
--------------

When creating a :class:`~contextgem.public.concepts.NumericalConcept`, you can specify the following parameters:

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
     - A clear description of what numerical value to extract, which can include explicit values to find, calculations to perform, or quantitative assessments to derive from the document content
   * - ``numeric_type``
     - ``str``
     - ``"any"``
     - The type of numerical values to extract. Available values: ``"int"``, ``"float"``, ``"any"``. When ``"any"`` is specified, the system will automatically determine whether to use an integer or floating-point representation based on the extracted value, choosing the most appropriate type for each numerical item.
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
     - Whether to include justifications for extracted items. Justifications provide explanations of why the LLM extracted specific numerical values and the reasoning behind the extraction, which is especially useful for complex calculations, inferred values, or when debugging results.
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
     - Whether to include source references for extracted items. References indicate the specific locations in the document where the numerical values were either directly found or from which they were calculated or inferred, helping to trace back extracted values to their source content even when the extraction involves complex calculations or mathematical reasoning.
   * - ``reference_depth``
     - ``str``
     - ``"paragraphs"``
     - Source reference granularity. Available values: ``"paragraphs"``, ``"sentences"``.
   * - ``singular_occurrence``
     - ``bool``
     - ``False``
     - Whether this concept is restricted to having only one extracted item. If ``True``, only a single numerical value will be extracted. For numerical concepts, this parameter is particularly useful when you want to extract a single specific value rather than identifying multiple numerical values throughout the document. This helps distinguish between single-value concepts versus multi-value concepts (e.g., *"total contract value"* vs *"all payment amounts"*). Note that with advanced LLMs, this constraint may not be required as they can often infer the appropriate number of items to extract from the concept's name, description, and type.
   * - ``custom_data``
     - ``dict``
     - ``{}``
     - Optional. Dictionary for storing any additional data that you want to associate with the concept. This data must be JSON-serializable. This data is not used for extraction but can be useful for custom processing or downstream tasks.


🚀 Advanced Usage
--------------------

🔍 References and Justifications for Extraction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can configure a :class:`~contextgem.public.concepts.NumericalConcept` to include justifications and references. Justifications explain the reasoning behind the extracted values, while references point to the parts of the document where the values were found or from which they were calculated or inferred:

.. literalinclude:: ../../../dev/usage_examples/docs/concepts/numerical_concept/refs_and_justifications.py
   :language: python


📊 Extracted Items
--------------------

When a :class:`~contextgem.public.concepts.NumericalConcept` is extracted, it is populated with **a list of extracted items** accessible through the ``.extracted_items`` property. Each item is an instance of the ``_IntegerItem``, ``_FloatItem`` or ``_IntegerOrFloatItem`` class, for ``numeric_type`` set to ``"int"``, ``"float"`` or ``"any"`` (default) respectively, with the following attributes:

.. list-table::
   :widths: 20 20 60
   :header-rows: 1

   * - Attribute
     - Type
     - Description
   * - ``value``
     - int or float
     - The extracted numerical value, either an integer or floating-point number depending on the ``numeric_type`` setting
   * - ``justification``
     - str
     - Explanation of why this numerical value was extracted (only if ``add_justifications=True``)
   * - ``reference_paragraphs``
     - list[:class:`~contextgem.public.paragraphs.Paragraph`]
     - List of paragraph objects where the numerical value was found or from which it was calculated or inferred (only if ``add_references=True``)
   * - ``reference_sentences``
     - list[:class:`~contextgem.public.sentences.Sentence`]
     - List of sentence objects where the numerical value was found or from which it was calculated or inferred (only if ``add_references=True`` and ``reference_depth="sentences"``)

.. note::
   To find the position of a reference paragraph or sentence in the document, e.g. to cite it or to retrieve its surrounding context, see :ref:`locating-paras-and-sents`.


💡 Best Practices
-------------------

Here are some best practices to optimize your use of :class:`~contextgem.public.concepts.NumericalConcept`:

- Provide a clear and specific description that helps the LLM understand exactly what numerical values to extract, using precise and unambiguous language in your concept names and descriptions. For numerical concepts, be explicit about the exact values you're seeking (e.g., *"the total contract value in USD"* rather than just *"contract value"*). Avoid vague terms that could lead to incorrect extractions—for example, use *"quarterly revenue figures in millions"* instead of *"revenue numbers"* to ensure consistent and accurate extractions.
- Use the appropriate ``numeric_type`` based on what you expect to extract or calculate:

  - Use ``"int"`` for counts, quantities, or whole numbers
  - Use ``"float"`` for prices, measurements, or values that may have decimal points
  - Use ``"any"`` when you're not sure or need to extract both types

- Break down complex numerical extractions into multiple simpler numerical concepts when appropriate. Instead of one concept extracting *"all financial metrics,"* consider separate concepts for *"revenue figures,"* *"expense amounts,"* and *"profit margins."* This provides more structured data and makes it easier to process the results for specific purposes.
- Enable justifications (using ``add_justifications=True``) when you need to understand the reasoning behind the LLM's numerical extractions, especially when calculations or conversions are involved.
- Enable references (using ``add_references=True``) when you need to trace back to specific parts of the document that contained the numerical values or were used to calculate derived values.
- Use ``singular_occurrence=True`` to enforce only a single numerical value extraction. This is particularly useful for concepts that should yield a unique value, such as *"total contract value"* or *"effective interest rate,"* rather than identifying multiple numerical values throughout the document.
