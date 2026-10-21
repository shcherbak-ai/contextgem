.. meta::
   :description: Complete, self-contained examples of extracting aspects and concepts from a document with ContextGem, with an LLM or a decision model, step by step.

Quickstart examples
====================

Below are complete, self-contained examples showing how to extract data from a document using ContextGem.


🔄 Extraction Process
----------------------

ContextGem follows a simple extraction process:

1. Create a :class:`~contextgem.public.documents.Document` instance with your content, or, for a PDF, DOCX or other file converted with Docling, with :meth:`~contextgem.public.documents.Document.from_docling` (see :doc:`documents/docling`)
2. Define :class:`~contextgem.public.aspects.Aspect` instances for sections of interest
3. Define concept instances (:class:`~contextgem.public.concepts.StringConcept`, :class:`~contextgem.public.concepts.BooleanConcept`, :class:`~contextgem.public.concepts.NumericalConcept`, :class:`~contextgem.public.concepts.DateConcept`, :class:`~contextgem.public.concepts.JsonObjectConcept`, :class:`~contextgem.public.concepts.RatingConcept`, :class:`~contextgem.public.concepts.LabelConcept`) for specific data points to extract, and attach them to :class:`~contextgem.public.aspects.Aspect` (for aspect context) or :class:`~contextgem.public.documents.Document` (for document context).
4. Use :class:`~contextgem.public.llms.DocumentLLM`, :class:`~contextgem.public.decision_models.DocumentDecisionModel` (for the aspects and concepts that a decision model extracts, see :doc:`decision_models/decision_models`) or :class:`~contextgem.public.models.DocumentModelGroup` to perform the extraction
5. Access the extracted data in the document object


📋 Aspect Extraction from Document
-----------------------------------

.. tip::
   Aspect extraction is useful for identifying and extracting specific sections or topics from documents. Common use cases include:

   * Extracting specific clauses from legal contracts
   * Identifying specific sections from financial reports
   * Isolating relevant topics from research papers
   * Extracting product features from technical documentation

.. literalinclude:: ../../dev/usage_examples/docs/quickstart/quickstart_aspect.py
   :language: python


🌳 Extracting Aspect with Sub-Aspects
--------------------------------------

.. tip::
   Sub-aspect extraction helps organize complex topics into logical components. Common use cases include:

   * Breaking down termination clauses in employment contracts into company rights, employee rights, and severance terms
   * Dividing financial report sections into revenue streams, expenses, and forecasts
   * Organizing product specifications into technical details, compatibility, and maintenance requirements

.. literalinclude:: ../../dev/usage_examples/docs/quickstart/quickstart_sub_aspect.py
   :language: python


🔍 Concept Extraction from Aspect
----------------------------------

.. tip::
   Concept extraction from aspects helps identify specific data points within already extracted sections or topics. Common use cases include:

   * Extracting payment amounts from a contract's payment terms
   * Extracting liability cap from a contract's liability section
   * Isolating timelines from delivery terms
   * Extracting a list of features from a product description
   * Identifying programming languages from a CV's experience section

.. literalinclude:: ../../dev/usage_examples/docs/quickstart/quickstart_concept_aspect.py
   :language: python


📝 Concept Extraction from Document (text)
-------------------------------------------

.. tip::
   Concept extraction from text documents locates specific information directly from text. Common use cases include:

   * Extracting anomalies from entire legal documents
   * Identifying financial figures across multiple report sections
   * Extracting citations and references from academic papers
   * Identifying product specifications from technical manuals
   * Extracting contact information from business documents

.. literalinclude:: ../../dev/usage_examples/docs/quickstart/quickstart_concept_document_text.py
   :language: python


.. _vision-concept-extraction-label:

🖼️ Concept Extraction from Document (vision)
---------------------------------------------

.. tip::
   Concept extraction using vision capabilities processes documents with complex layouts or images. Common use cases include:

   * Extracting data from scanned contracts or receipts
   * Identifying information from charts and graphs in reports
   * Identifying visual product features from marketing materials

.. literalinclude:: ../../dev/usage_examples/docs/quickstart/quickstart_concept_document_vision.py
   :language: python


⚖️ Aspect and Concept Extraction with a Decision Model
-------------------------------------------------------

.. tip::
   A decision model answers closed questions about a text (yes/no, one of your labels, a level of a scale) with the probability of each possible answer, and generates no text. Common use cases include:

   * Finding the paragraphs on a topic, e.g. the termination clauses of a contract
   * Checking whether a contract has a clause, e.g. termination for convenience
   * Classifying documents or clauses by type
   * Rating a document on a scale, e.g. how detailed its payment terms are
   * Sending the less certain answers to a human review, by their probabilities

.. literalinclude:: ../../dev/usage_examples/readme/quickstart_decision_model.py
   :language: python

.. seealso::
   A decision model also extracts the concepts of an aspect, from the paragraphs extracted for it, and works together with LLMs in a model group: see :doc:`decision_models/decision_models`.


📄 Extraction from a File Converted with Docling
-------------------------------------------------

.. tip::
   `Docling <https://github.com/docling-project/docling>`_ converts files in many formats, e.g. PDF, DOCX, PPTX, XLSX and HTML (see the :ref:`supported formats <docling-formats-label>`), and :meth:`~contextgem.public.documents.Document.from_docling` creates a document from the result, whose paragraphs keep their place in the structure of the file and the pages they are on. Common use cases include:

   * Extracting the values of the tables of a PDF file, e.g. the fees of a contract
   * Citing the page of each extracted item, e.g. for a reviewer to check it in the file
   * Extracting from presentations and spreadsheets, with the slide or the sheet of each paragraph
   * Extracting from legacy Office files, e.g. DOC, which Docling converts with LibreOffice

.. literalinclude:: ../../dev/usage_examples/readme/quickstart_docling.py
   :language: python

.. seealso::
   Installing Docling with the ``docling`` extra, what each part of a file becomes, and the pictures of a file: see :doc:`documents/docling`.


💬 Lightweight LLM Chat Interface
----------------------------------

.. note::
   While ContextGem is primarily designed for advanced structured data extraction, it also provides a lightweight, unified interface for interacting with LLMs via natural language - across both text and vision - with built-in fallback support.

.. tip::
   To preserve message history across turns, pass a :class:`~contextgem.public.llms.ChatSession` instance via ``chat_session=...`` to ``DocumentLLM.chat(...)`` (or ``.chat_async(...)``). Without a session, each ``chat(...)`` call is treated as a one-off message → response interaction.

.. literalinclude:: ../../dev/usage_examples/readme/llm_chat.py
   :language: python


.. _llm-chat-with-tools-label:

🛠️ Chat with Tools
~~~~~~~~~~~~~~~~~~~~

Use the ``@register_tool`` decorator to enable LLM tool calling. Schemas are auto-generated from type hints and docstrings.

.. literalinclude:: ../../dev/usage_examples/docs/llm_config/chat_with_tools_simple.py
   :language: python

.. seealso::
   For advanced usage including ``TypedDict`` for complex parameters, parameter descriptions and constraints with ``Annotated``, and supported type hints, see :doc:`llms/chat_with_tools`.
