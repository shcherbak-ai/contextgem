:og:description: ContextGem: Dealing with Long Documents

Dealing with Long Documents
============================

ContextGem offers specialized configuration options for efficiently processing lengthy documents.

✂️ Segmentation Approach
--------------------------

Unlike many systems that rely on chunking (e.g. RAG), ContextGem intelligently segments documents into natural semantic units like paragraphs and sentences. This preserves the contextual integrity of the content while allowing you to configure:

- Maximum number of paragraphs per LLM call
- Maximum number of aspects/concepts to analyze per LLM call
- Maximum number of images per LLM call (if the document contains images)

⚙️ Effective Optimization Strategies
--------------------------------------

- **🔄 Use Long-Context Models**: Select models with large context windows. (See :doc:`optimization_choosing_llm` for guidance on choosing the right model.)
- **📏 Limit Paragraphs Per Call**: This will reduce each prompt's length and ensure a more focused analysis.
- **🔢 Limit Aspects/Concepts Per Call**: Process a smaller number of aspects or concepts in each LLM call, preventing prompt overloading.
- **⚠️ Use Sentence-Level Reference Depth Sparingly**: Only use sentence-level reference depth for aspects or concepts when absolutely necessary, as it requires loading a SaT model and running sentence segmentation on text, which can be slow for long documents.
- **⚡ Optional: Enable Concurrency**: Enable running extractions concurrently if your API setup permits. This will reduce the overall processing time. (See :doc:`optimization_speed` for guidance on configuring concurrency.)

Since each use case has unique requirements, experiment with different configurations to find your optimal setup.

.. literalinclude:: ../../../dev/usage_examples/docs/optimizations/optimization_long_docs.py
    :language: python
    :caption: Example of configuring LLM extraction for long documents
