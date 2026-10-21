.. meta::
   :description: How to improve extraction accuracy: a capable model, larger segmentation models, examples, justifications, fewer paragraphs, aspects and concepts per call, and a fallback model.

Optimizing for Accuracy
========================

When accuracy is paramount, ContextGem offers several techniques to improve extraction quality:

- **🚀 Use a Capable LLM**: Choose a powerful LLM model for extraction.
- **🪄 Use Larger Segmentation Models**: Select a larger SaT model for intelligent segmentation of paragraphs or sentences, to ensure the highest segmentation accuracy in complex documents (e.g. contracts).
- **💡 Provide Examples**: For most complex concepts, add examples to guide the LLM's extraction format and style.
- **🧠 Request Justifications**: For most complex aspects/concepts, enable justifications to understand the LLM's reasoning and instruct the LLM to "think" when giving an answer.
- **📏 Limit Paragraphs Per Call**: This will reduce each prompt's length and ensure a more focused analysis.
- **🔢 Limit Aspects/Concepts Per Call**: Process a smaller number of aspects or concepts in each LLM call, preventing prompt overloading.
- **🔄 Use a Fallback Model**: Configure a fallback model to retry failed extractions with a different model.


.. literalinclude:: ../../../dev/usage_examples/docs/optimizations/optimization_accuracy.py
    :language: python
    :caption: Example of optimizing extraction for accuracy
