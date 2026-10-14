:og:description: ContextGem: Optimizing for Speed

.. meta::
   :description: How to speed up extraction: concurrency, smaller models, a fallback model, default parameters, and sparing use of justifications and sentence-level references.

Optimizing for Speed
=====================

For large-scale processing or time-sensitive applications, optimize your pipeline for speed:

- **🚀 Enable and Configure Concurrency**: Process multiple extractions concurrently. Adjust the async limiter to adapt to your model API setup.
- **📦 Use Smaller Models**: Select smaller/distilled LLMs that perform faster. (See :doc:`optimization_choosing_llm` for guidance on choosing the right model.)
- **🔄 Use a Fallback Model**: Configure a fallback model to retry extractions that failed due to rate limits.
- **⚙️ Use Default Parameters**: Without concurrency, all the extractions are processed in as few model calls as possible. With concurrency enabled, each aspect/concept is processed in a call of its own by default.
- **📉 Enable Justifications Only When Necessary**: Do not use justifications for simple aspects or concepts. This will reduce the number of tokens generated.
- **⚠️ Use Sentence-Level Reference Depth Sparingly**: Only use sentence-level reference depth for aspects or concepts when absolutely necessary, as it requires loading a SaT model and running sentence segmentation on text, which can be slow for long documents.


.. literalinclude:: ../../../dev/usage_examples/docs/optimizations/optimization_speed.py
    :language: python
    :caption: Example of optimizing extraction for speed
