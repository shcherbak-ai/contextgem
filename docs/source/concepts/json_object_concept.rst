:og:image: https://contextgem.shcherbak.ai/_static/docs_preview_image_json_object_concept.png

.. meta::
   :description: How to extract structured data as JSON objects with JsonObjectConcept, with a structure that you define as a dictionary or as classes, including nested ones.

JsonObjectConcept
==================

:class:`~contextgem.public.concepts.JsonObjectConcept` is a powerful concept type that extracts structured data in the form of JSON objects from documents, enabling sophisticated information organization and retrieval.


📝 Overview
-------------

:class:`~contextgem.public.concepts.JsonObjectConcept` is used when you need to extract complex, structured information from unstructured text, including:

* **Nested data structures**: objects with multiple fields, hierarchical information, and related attributes
* **Standardized formats**: consistent data extraction following predefined schemas for reliable downstream processing
* **Complex entity extraction**: comprehensive extraction of entities with multiple attributes and relationships


💻 Usage Example
------------------

Here's a simple example of how to use :class:`~contextgem.public.concepts.JsonObjectConcept` to extract product information:

.. literalinclude:: ../../../dev/usage_examples/docs/concepts/json_object_concept/json_object_concept.py
   :language: python


⚙️ Parameters
--------------

When creating a :class:`~contextgem.public.concepts.JsonObjectConcept`, you can specify the following parameters:

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
   * - ``structure``
     - ``type | dict[str, Any]``
     - (Required)
     - JSON object schema defining the data structure to be extracted. Can be specified as a Python class with type annotations or a dictionary with field names as keys and their corresponding types as values. This schema can represent simple flat structures or complex nested hierarchies with multiple levels of organization. The LLM will attempt to extract data that conforms to this structure, enabling precise and consistent extraction of complex information patterns.
   * - ``examples``
     - ``list[JsonObjectExample]``
     - ``[]``
     - Optional. Example JSON objects illustrating the concept usage. Such examples must conform to the ``structure`` schema. Examples significantly improve extraction accuracy by showing the LLM concrete instances of the expected output format and content patterns. This is particularly valuable for complex schemas with nested structures or when there are specific formatting conventions that should be followed (e.g., how dates, identifiers, or specialized fields should be represented). Examples also help clarify how to handle edge cases or ambiguous information in the source document.
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
     - Whether to include justifications for extracted items. Justifications provide explanations of why the LLM extracted specific JSON structures and the reasoning behind field values. This is especially valuable for complex structures where the extraction process involves inference or when multiple data points must be synthesized. For example, a justification might explain how the LLM determined a product's category based on various features mentioned across different paragraphs, or why certain optional fields were populated or left empty based on available information in the document.
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
     - Whether to include source references for extracted items. References indicate the specific locations in the document that informed the extraction of the JSON structure. This is particularly valuable for complex objects where field values may be calculated or inferred from multiple scattered pieces of information throughout the document. References help trace back extracted values to their source evidence, validate the extraction reasoning, and understand which parts of the document contributed to the synthesis of structured data, especially for fields requiring interpretation, not only direct extraction.
   * - ``reference_depth``
     - ``str``
     - ``"paragraphs"``
     - Source reference granularity. Available values: ``"paragraphs"``, ``"sentences"``.
   * - ``singular_occurrence``
     - ``bool``
     - ``False``
     - Whether this concept is restricted to having only one extracted item. If ``True``, only a single JSON object will be extracted. For JSON object concepts, this parameter is particularly useful when you want to extract a comprehensive structured representation of a single entity (e.g., "product specifications" or "company profile") rather than multiple instances of structured data scattered throughout the document. This is especially valuable when extracting complex nested objects that aggregate information from different parts of the document into a cohesive whole. Note that with advanced LLMs, this constraint may not be required as they can often infer the appropriate number of objects to extract from the concept's name, description, and schema structure.
   * - ``custom_data``
     - ``dict``
     - ``{}``
     - Optional. Dictionary for storing any additional data that you want to associate with the concept. This data must be JSON-serializable. This data is not used for extraction but can be useful for custom processing or downstream tasks.


🏗️ Defining Structure
-----------------------

The ``structure`` parameter defines the schema for the data you want to extract. JsonObjectConcept uses Pydantic models internally to validate all structures, ensuring type safety and data integrity. Dictionary-based definitions provide a simpler abstraction for defining JSON object structures, while still benefiting from Pydantic's robust validation system under the hood.

You can define the structure in several ways:

1. **Using a dictionary with type annotations:**

.. literalinclude:: ../../../dev/usage_examples/docs/concepts/json_object_concept/structure/simple_structure.py
   :language: python

2. **Using nested dictionaries for complex structures:**

.. literalinclude:: ../../../dev/usage_examples/docs/concepts/json_object_concept/structure/nested_structure.py
   :language: python

3. **Using a Python class with type annotations:**

While dictionary structures provide the simplest way to define JSON schemas, you may prefer to use class definitions if that better fits your codebase style:

.. literalinclude:: ../../../dev/usage_examples/docs/concepts/json_object_concept/structure/simple_class_structure.py
   :language: python

4. **Using nested classes for complex structures:**

For hierarchical data structures, you can also use nested class definitions, e.g. when your application code already has dataclasses or Pydantic models for the data.

When using nested class definitions, all classes in the structure must inherit from the ``JsonObjectClassStruct`` utility class to enable automatic conversion of the whole class hierarchy to a dictionary structure:

.. literalinclude:: ../../../dev/usage_examples/docs/concepts/json_object_concept/structure/nested_class_structure.py
   :language: python


🚀 Advanced Usage
--------------------

✏️ Adding Examples
~~~~~~~~~~~~~~~~~~~~

You can provide examples of structured JSON objects to improve extraction accuracy, especially for complex schemas or when there might be ambiguity in how to organize or format the extracted information:

.. literalinclude:: ../../../dev/usage_examples/docs/concepts/json_object_concept/adding_examples.py
   :language: python

🔍 References and Justifications for Extraction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can configure a :class:`~contextgem.public.concepts.JsonObjectConcept` to include justifications and references, which provide transparency into the extraction process. Justifications explain the reasoning behind the extracted values, while references point to the specific parts of the document that were used as sources for the extraction:

.. literalinclude:: ../../../dev/usage_examples/docs/concepts/json_object_concept/refs_and_justifications.py
   :language: python

.. note::
   To find the position of a reference paragraph or sentence in the document, e.g. to cite it or to retrieve its surrounding context, see :ref:`locating-paras-and-sents`.


💡 Best Practices
-------------------

- Keep your JSON structures simple yet comprehensive, focusing on the essential fields needed for your use case to avoid LLM prompt overloading.
- Include realistic examples (using :class:`~contextgem.public.examples.JsonObjectExample`) that precisely match your schema to guide extraction, especially for ambiguous or specialized data formats.
- Provide detailed descriptions for your JsonObjectConcept that specify exactly what structured data to extract and how fields should be interpreted.
- For complex JSON objects, use nested dictionaries or class hierarchies to organize related fields logically.
- Enable justifications (using ``add_justifications=True``) when interpretation rationale is important, especially for extractions that involve judgment or qualitative assessment, such as sentiment analysis (positive/negative), priority assignment (high/medium/low), or data categorization where the LLM must make interpretive decisions rather than extract explicit facts.
- Enable references (using ``add_references=True``) when you need to verify the document source of extracted values for compliance or verification purposes. This is especially valuable when the LLM is not just directly extracting explicit text, but also interpreting or inferring information from context. For example, in legal document analysis where traceability of information is essential for auditing or validation, references help track both explicit statements and the implicit information the model has derived from them.
- Use ``singular_occurrence=True`` when you expect exactly one instance of the structured data in the document (e.g., a single product specification, one patient medical record, or a unique customer complaint). This is useful for documents with a clear singular focus. Conversely, omit this parameter (``False`` is the default) when you need to extract multiple instances of the same structure from a document, such as multiple product listings in a catalog, several patient records in a hospital report, or various customer complaints in a feedback compilation.
