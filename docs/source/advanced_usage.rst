.. meta::
   :description: Complete examples of advanced ContextGem usage: aspects that contain concepts, extracting aspects and concepts together, and a pipeline that uses several LLMs over several documents.

Advanced usage examples
=======================

Below are complete, self-contained examples demonstrating advanced usage of ContextGem.


🔍 Extracting Aspects Containing Concepts
------------------------------------------

.. tip::
   Concept extraction is useful for extracting specific data points from a document or an aspect. For example, a "Payment terms" aspect in a contract may have multiple concepts:

   * "Payment amount"
   * "Payment due date"
   * "Payment method"

.. literalinclude:: ../../dev/usage_examples/docs/advanced/advanced_aspects_with_concepts.py
   :language: python


📊 Extracting Aspects and Concepts from a Document
----------------------------------------------------

.. tip::
   This example demonstrates how to extract both document-level concepts and aspect-specific concepts from a document with references. Using concurrency can significantly speed up extraction when working with multiple aspects and concepts.

   Document-level concepts apply to the entire document (like "Is Privacy Policy" or "Last Updated Date"), while aspect-specific concepts are tied to particular sections or themes within the document.

.. literalinclude:: ../../dev/usage_examples/docs/advanced/advanced_aspects_and_concepts_document.py
   :language: python


.. _multi-llm-pipeline-label:

🔄 Using a Multi-LLM Pipeline to Extract Data from Several Documents
---------------------------------------------------------------------

.. tip::
   A pipeline is a reusable configuration of extraction steps. You can use the same pipeline to extract data from multiple documents.

   For example, if your app extracts data from invoices, you can configure a pipeline once, and then use it for each incoming invoice.

.. literalinclude:: ../../dev/usage_examples/docs/advanced/advanced_multiple_docs_pipeline.py
   :language: python
