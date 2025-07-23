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

:og:description: ContextGem: LLM Extraction Methods


Extraction Methods
==================

This guide documents the extraction methods provided by the :class:`~contextgem.public.llms.DocumentLLM` and :class:`~contextgem.public.llms.DocumentLLMGroup` classes for extracting aspects and concepts from documents using large language models.

|

📄🧠 Complete Document Processing
-----------------------------------

:meth:`~contextgem.public.llms.DocumentLLM.extract_all`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Performs comprehensive extraction by processing a :class:`~contextgem.public.documents.Document` for all :class:`~contextgem.public.aspects.Aspect` and :class:`~contextgem.internal.base.concepts._Concept` instances. This is the most commonly used method for complete document analysis.

.. note::
   See supported concept types in :doc:`../concepts/supported_concepts`. All public concept types inherit from the internal :class:`~contextgem.internal.base.concepts._Concept` base class.

**Method Signature:**

.. code-block:: python

   def extract_all(
       self,
       document: Document,
       overwrite_existing: bool = False,
       max_items_per_call: int = 0,
       use_concurrency: bool = False,
       max_paragraphs_to_analyze_per_call: int = 0,
       max_images_to_analyze_per_call: int = 0,
   ) -> Document

.. note::
   An async equivalent :meth:`~contextgem.public.llms.DocumentLLM.extract_all_async` is also available.

**Parameters:**

.. list-table::
   :header-rows: 1
   :widths: 15 15 10 60

   * - Parameter
     - Type
     - Default
     - Description
   * - ``document``
     - ``Document``
     - (Required)
     - The document with attached :class:`~contextgem.public.aspects.Aspect` and/or :class:`~contextgem.internal.base.concepts._Concept` instances to extract.
   * - ``overwrite_existing``
     - ``bool``
     - ``False``
     - Whether to overwrite already processed :class:`~contextgem.public.aspects.Aspect` and :class:`~contextgem.internal.base.concepts._Concept` instances with newly extracted information. This is particularly useful when reprocessing documents with updated LLMs or extraction parameters.
   * - ``max_items_per_call``
     - ``int``
     - ``0``
     - Maximum number of :class:`~contextgem.public.aspects.Aspect` and/or :class:`~contextgem.internal.base.concepts._Concept` instances with the same extraction parameters to process in a single LLM call (single LLM prompt). ``0`` means all aspect and/or concept instances with same extraction params in a one call. This is particularly useful for complex tasks or long documents to prevent prompt overloading and allow the LLM to focus on a smaller set of extraction tasks at once.
   * - ``use_concurrency``
     - ``bool``
     - ``False``
     - Enable concurrent processing of multiple :class:`~contextgem.public.aspects.Aspect` and/or :class:`~contextgem.internal.base.concepts._Concept` instances. Can significantly reduce processing time by executing multiple extraction tasks in parallel, especially beneficial for documents with many aspects and concepts. However, it might cause rate limit errors with LLM providers. When enabled, adjust the ``async_limiter`` on your :class:`~contextgem.public.llms.DocumentLLM` to control request frequency (default is 3 acquisitions per 10 seconds). For optimal results, combine with ``max_items_per_call=1`` to maximize concurrency, although this would cause increase in LLM API costs as each aspect/concept will be processed in a separate LLM call (LLM prompt). See :doc:`../optimizations/optimization_speed` for examples of concurrency configuration.
   * - ``max_paragraphs_to_analyze_per_call``
     - ``int``
     - ``0``
     - Maximum paragraphs to include in a single LLM call (single LLM prompt). ``0`` means all paragraphs. This parameter is crucial when working with long documents that exceed the LLM's context window. By limiting the number of paragraphs per call, you can ensure the LLM processes the document in manageable segments while maintaining semantic coherence. This prevents token limit errors and often improves extraction quality by allowing the model to focus on smaller portions of text at a time. For more details on handling long documents, see :doc:`../optimizations/optimization_long_docs`.
   * - ``max_images_to_analyze_per_call``
     - ``int``
     - ``0``
     - Maximum :class:`~contextgem.public.images.Image` instances to analyze in a single LLM call (single LLM prompt). ``0`` means all images. This parameter is crucial when working with documents containing multiple images that might exceed the LLM's context window. By limiting the number of images per call, you can ensure the LLM processes the document's visual content in manageable batches. Relevant only when extracting document-level concepts from document images. See :ref:`vision-concept-extraction-label` for an example of extracting concepts from document images.
   * - ``raise_exception_on_extraction_error``
     - ``bool``
     - ``True``
     - Whether to raise an exception if the extraction fails due to invalid data returned by an LLM or an error in the LLM API. If True (default): if the LLM returns invalid data, ``LLMExtractionError`` will be raised, and if the LLM API call fails, ``LLMAPIError`` will be raised. If False, a warning will be issued instead, and no extracted items will be returned.

|

**Return Value:**

Returns the same :class:`~contextgem.public.documents.Document` instance passed as input, but with all attached :class:`~contextgem.public.aspects.Aspect` and :class:`~contextgem.internal.base.concepts._Concept` instances populated with their extracted items. The document's aspects and concepts will have their ``extracted_items`` field populated with the extracted information, and if applicable, ``reference_paragraphs``/ ``reference_sentences`` will be set based on the extraction parameters. The exact structure of references depends on the ``reference_depth`` setting of each aspect and concept.

**Example Usage:**

.. literalinclude:: ../../../dev/usage_examples/docs/llms/llm_extraction_methods/extract_all.py
   :language: python
   :caption: Extracting all aspects and concepts from a document

.. raw:: html

   <a target="_blank" href="https://colab.research.google.com/github/shcherbak-ai/contextgem/blob/main/dev/notebooks/docs/llms/llm_extraction_methods/extract_all.ipynb">
     <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
   </a>

|

📄 Aspect Extraction Methods
-----------------------------

:meth:`~contextgem.public.llms.DocumentLLM.extract_aspects_from_document`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Extracts :class:`~contextgem.public.aspects.Aspect` instances from a :class:`~contextgem.public.documents.Document`.

**Method Signature:**

.. code-block:: python

   def extract_aspects_from_document(
       self,
       document: Document,
       from_aspects: list[Aspect] | None = None,
       overwrite_existing: bool = False,
       max_items_per_call: int = 0,
       use_concurrency: bool = False,
       max_paragraphs_to_analyze_per_call: int = 0,
   ) -> list[Aspect]

.. note::
   An async equivalent :meth:`~contextgem.public.llms.DocumentLLM.extract_aspects_from_document_async` is also available.

**Parameters:**

.. list-table::
   :header-rows: 1
   :widths: 15 15 10 60

   * - Parameter
     - Type
     - Default
     - Description
   * - ``document``
     - ``Document``
     - (Required)
     - The document with attached :class:`~contextgem.public.aspects.Aspect` instances to be extracted.
   * - ``from_aspects``
     - ``list[Aspect] | None``
     - ``None``
     - Specific aspects to extract from the document. If ``None``, extracts all aspects attached to the document. This allows you to selectively process only certain aspects rather than the entire set.
   * - ``overwrite_existing``
     - ``bool``
     - ``False``
     - Whether to overwrite already processed aspects with newly extracted information. This is particularly useful when reprocessing documents with updated LLMs or extraction parameters.
   * - ``max_items_per_call``
     - ``int``
     - ``0``
     - Maximum number of :class:`~contextgem.public.aspects.Aspect` instances with the same extraction parameters to process in a single LLM call (single LLM prompt). ``0`` means all aspect instances with same extraction params in a one call. This is particularly useful for complex tasks or long documents to prevent prompt overloading and allow the LLM to focus on a smaller set of extraction tasks at once.
   * - ``use_concurrency``
     - ``bool``
     - ``False``
     - Enable concurrent processing of multiple :class:`~contextgem.public.aspects.Aspect` instances. Can significantly reduce processing time by executing multiple extraction tasks concurrently, especially beneficial for documents with many aspects. However, it might cause rate limit errors with LLM providers. When enabled, adjust the ``async_limiter`` on your :class:`~contextgem.public.llms.DocumentLLM` to control request frequency (default is 3 acquisitions per 10 seconds). For optimal results, combine with ``max_items_per_call=1`` to maximize concurrency, although this would cause increase in LLM API costs as each aspect will be processed in a separate LLM call (LLM prompt). See :doc:`../optimizations/optimization_speed` for examples of concurrency configuration.
   * - ``max_paragraphs_to_analyze_per_call``
     - ``int``
     - ``0``
     - Maximum paragraphs to include in a single LLM call (single LLM prompt). ``0`` means all paragraphs. This parameter is crucial when working with long documents that exceed the LLM's context window. By limiting the number of paragraphs per call, you can ensure the LLM processes the document in manageable segments while maintaining semantic coherence. This prevents token limit errors and often improves extraction quality by allowing the model to focus on smaller portions of text at a time. For more details on handling long documents, see :doc:`../optimizations/optimization_long_docs`.
   * - ``raise_exception_on_extraction_error``
     - ``bool``
     - ``True``
     - Whether to raise an exception if the extraction fails due to invalid data returned by an LLM or an error in the LLM API. If True (default): if the LLM returns invalid data, ``LLMExtractionError`` will be raised, and if the LLM API call fails, ``LLMAPIError`` will be raised. If False, a warning will be issued instead, and no extracted items will be returned.

|

**Return Value:**

Returns a list of :class:`~contextgem.public.aspects.Aspect` instances that were processed during extraction. If ``from_aspects`` was specified, returns only those aspects; otherwise returns all aspects attached to the document. Each aspect in the returned list will have its ``extracted_items`` field populated with the extracted information, and its ``reference_paragraphs`` field will always be set. The ``reference_sentences`` field will only be populated when the aspect's ``reference_depth`` is set to ``"sentences"``.

**Example Usage:**

.. literalinclude:: ../../../dev/usage_examples/docs/llms/llm_extraction_methods/extract_aspects_from_document.py
   :language: python
   :caption: Extracting aspects from a document

.. raw:: html

   <a target="_blank" href="https://colab.research.google.com/github/shcherbak-ai/contextgem/blob/main/dev/notebooks/docs/llms/llm_extraction_methods/extract_aspects_from_document.ipynb">
     <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
   </a>

|

🧠 Concept Extraction Methods
------------------------------

:meth:`~contextgem.public.llms.DocumentLLM.extract_concepts_from_document`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Extracts :class:`~contextgem.internal.base.concepts._Concept` instances from a :class:`~contextgem.public.documents.Document` object.

.. note::
   See supported concept types in :doc:`../concepts/supported_concepts`. All public concept types inherit from the internal :class:`~contextgem.internal.base.concepts._Concept` base class.

**Method Signature:**

.. code-block:: python

   def extract_concepts_from_document(
       self,
       document: Document,
       from_concepts: list[_Concept] | None = None,
       overwrite_existing: bool = False,
       max_items_per_call: int = 0,
       use_concurrency: bool = False,
       max_paragraphs_to_analyze_per_call: int = 0,
       max_images_to_analyze_per_call: int = 0,
   ) -> list[_Concept]

.. note::
   An async equivalent :meth:`~contextgem.public.llms.DocumentLLM.extract_concepts_from_document_async` is also available.

**Parameters:**

.. list-table::
   :header-rows: 1
   :widths: 15 15 10 60

   * - Parameter
     - Type
     - Default
     - Description
   * - ``document``
     - ``Document``
     - (Required)
     - The document from which concepts are to be extracted.
   * - ``from_concepts``
     - ``list[_Concept] | None``
     - ``None``
     - Specific concepts to extract from the document. If ``None``, extracts all concepts attached to the document. This allows you to selectively process only certain concepts rather than the entire set.
   * - ``overwrite_existing``
     - ``bool``
     - ``False``
     - Whether to overwrite already processed concepts with newly extracted information. This is particularly useful when reprocessing documents with updated LLMs or extraction parameters.
   * - ``max_items_per_call``
     - ``int``
     - ``0``
     - Maximum number of :class:`~contextgem.internal.base.concepts._Concept` instances with the same extraction parameters to process in a single LLM call (single LLM prompt). ``0`` means all concept instances with same extraction params in a one call. This is particularly useful for complex tasks or long documents to prevent prompt overloading and allow the LLM to focus on a smaller set of extraction tasks at once.
   * - ``use_concurrency``
     - ``bool``
     - ``False``
     - Enable concurrent processing of multiple :class:`~contextgem.internal.base.concepts._Concept` instances. Can significantly reduce processing time by executing multiple extraction tasks concurrently, especially beneficial for documents with many concepts. However, it might cause rate limit errors with LLM providers. When enabled, adjust the ``async_limiter`` on your :class:`~contextgem.public.llms.DocumentLLM` to control request frequency (default is 3 acquisitions per 10 seconds). For optimal results, combine with ``max_items_per_call=1`` to maximize concurrency, although this would cause increase in LLM API costs as each concept will be processed in a separate LLM call (LLM prompt). See :doc:`../optimizations/optimization_speed` for examples of concurrency configuration.
   * - ``max_paragraphs_to_analyze_per_call``
     - ``int``
     - ``0``
     - Maximum paragraphs to include in a single LLM call (single LLM prompt). ``0`` means all paragraphs. This parameter is crucial when working with long documents that exceed the LLM's context window. By limiting the number of paragraphs per call, you can ensure the LLM processes the document in manageable segments while maintaining semantic coherence.
   * - ``max_images_to_analyze_per_call``
     - ``int``
     - ``0``
     - Maximum images to include in a single LLM call (single LLM prompt). ``0`` means all images. This parameter is crucial when extracting concepts from documents with multiple images using vision-capable LLMs. It helps prevent overwhelming the model with too many visual inputs at once, manages token usage more effectively, and enables more focused concept extraction from visual content. See :ref:`vision-concept-extraction-label` for an example of extracting concepts from document images.
   * - ``raise_exception_on_extraction_error``
     - ``bool``
     - ``True``
     - Whether to raise an exception if the extraction fails due to invalid data returned by an LLM or an error in the LLM API. If True (default): if the LLM returns invalid data, ``LLMExtractionError`` will be raised, and if the LLM API call fails, ``LLMAPIError`` will be raised. If False, a warning will be issued instead, and no extracted items will be returned.

|

**Return Value:**

Returns a list of :class:`~contextgem.internal.base.concepts._Concept` instances that were processed during extraction. If ``from_concepts`` was specified, returns only those concepts; otherwise returns all concepts attached to the document. Each concept in the returned list will have its ``extracted_items`` field populated with the extracted information, and if applicable, ``reference_paragraphs``/ ``reference_sentences`` will be set based on the extraction parameters.

**Example Usage:**

.. literalinclude:: ../../../dev/usage_examples/docs/llms/llm_extraction_methods/extract_concepts_from_document.py
   :language: python
   :caption: Extracting concepts from a document

.. raw:: html

   <a target="_blank" href="https://colab.research.google.com/github/shcherbak-ai/contextgem/blob/main/dev/notebooks/docs/llms/llm_extraction_methods/extract_concepts_from_document.ipynb">
     <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
   </a>

:meth:`~contextgem.public.llms.DocumentLLM.extract_concepts_from_aspect`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Extracts :class:`~contextgem.internal.base.concepts._Concept` instances associated with a given :class:`~contextgem.public.aspects.Aspect` in a :class:`~contextgem.public.documents.Document`.

The aspect must be previously processed before concept extraction can occur. This means that the aspect should have already gone through extraction, which identifies the relevant context (text segments) in the document that match the aspect's description. This extracted context is then used as the foundation for concept extraction, allowing concepts to be identified specifically within the scope of the aspect.

.. note::
   See supported concept types in :doc:`../concepts/supported_concepts`. All public concept types inherit from the internal :class:`~contextgem.internal.base.concepts._Concept` base class.

**Method Signature:**

.. code-block:: python

   def extract_concepts_from_aspect(
       self,
       aspect: Aspect,
       document: Document,
       from_concepts: list[_Concept] | None = None,
       overwrite_existing: bool = False,
       max_items_per_call: int = 0,
       use_concurrency: bool = False,
       max_paragraphs_to_analyze_per_call: int = 0,
   ) -> list[_Concept]

.. note::
   An async equivalent :meth:`~contextgem.public.llms.DocumentLLM.extract_concepts_from_aspect_async` is also available.

**Parameters:**

.. list-table::
   :header-rows: 1
   :widths: 15 15 10 60

   * - Parameter
     - Type
     - Default
     - Description
   * - ``aspect``
     - ``Aspect``
     - (Required)
     - The aspect from which to extract concepts. Must be previously processed through aspect extraction before concepts can be extracted.
   * - ``document``
     - ``Document``
     - (Required)
     - The document that contains the aspect with the attached concepts to be extracted.
   * - ``from_concepts``
     - ``list[_Concept] | None``
     - ``None``
     - Specific concepts to extract from the aspect. If ``None``, extracts all concepts attached to the aspect. This allows you to selectively process only certain concepts rather than the entire set.
   * - ``overwrite_existing``
     - ``bool``
     - ``False``
     - Whether to overwrite already processed concepts with newly extracted information. This is particularly useful when reprocessing documents with updated LLMs or extraction parameters.
   * - ``max_items_per_call``
     - ``int``
     - ``0``
     - Maximum number of :class:`~contextgem.internal.base.concepts._Concept` instances with the same extraction parameters to process in a single LLM call (single LLM prompt). ``0`` means all concept instances with same extraction params in one call. This is particularly useful for complex tasks to prevent prompt overloading and allow the LLM to focus on a smaller set of extraction tasks at once.
   * - ``use_concurrency``
     - ``bool``
     - ``False``
     - Enable concurrent processing of multiple :class:`~contextgem.internal.base.concepts._Concept` instances. Can significantly reduce processing time by executing multiple extraction tasks concurrently, especially beneficial for aspects with many concepts. However, it might cause rate limit errors with LLM providers. When enabled, adjust the ``async_limiter`` on your :class:`~contextgem.public.llms.DocumentLLM` to control request frequency (default is 3 acquisitions per 10 seconds). For optimal results, combine with ``max_items_per_call=1`` to maximize concurrency, although this would cause increase in LLM API costs as each concept will be processed in a separate LLM call (LLM prompt). See :doc:`../optimizations/optimization_speed` for examples of concurrency configuration.
   * - ``max_paragraphs_to_analyze_per_call``
     - ``int``
     - ``0``
     - Maximum number of the aspect's paragraphs to analyze in a single LLM call (single LLM prompt). ``0`` means all the aspect's paragraphs. This parameter is crucial when working with long documents or aspects that cover extensive portions of text that might exceed the LLM's context window. By limiting the number of paragraphs per call, you can break down analysis into manageable chunks or allow the LLM to focus more deeply on smaller sections of text at a time. For more details on handling long documents, see :doc:`../optimizations/optimization_long_docs`.
   * - ``raise_exception_on_extraction_error``
     - ``bool``
     - ``True``
     - Whether to raise an exception if the extraction fails due to invalid data returned by an LLM or an error in the LLM API. If True (default): if the LLM returns invalid data, ``LLMExtractionError`` will be raised, and if the LLM API call fails, ``LLMAPIError`` will be raised. If False, a warning will be issued instead, and no extracted items will be returned.

|

**Return Value:**

Returns a list of :class:`~contextgem.internal.base.concepts._Concept` instances that were processed during extraction from the specified aspect. If ``from_concepts`` was specified, returns only those concepts; otherwise returns all concepts attached to the aspect. Each concept in the returned list will have its ``extracted_items`` field populated with the extracted information, and if applicable, ``reference_paragraphs``/ ``reference_sentences`` will be set based on the extraction parameters.

**Example Usage:**

.. literalinclude:: ../../../dev/usage_examples/docs/llms/llm_extraction_methods/extract_concepts_from_aspect.py
   :language: python
   :caption: Extracting concepts from an aspect

.. raw:: html

   <a target="_blank" href="https://colab.research.google.com/github/shcherbak-ai/contextgem/blob/main/dev/notebooks/docs/llms/llm_extraction_methods/extract_concepts_from_aspect.ipynb">
     <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
   </a>
