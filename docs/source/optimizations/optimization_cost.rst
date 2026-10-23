.. meta::
   :description: How to lower the cost of extraction: cost-efficient models, decision models for closed questions, default parameters, sparing justifications, usage and cost monitoring, and what each request to an LLM sends.

Optimizing for Cost
====================

ContextGem offers several strategies to optimize for cost efficiency while maintaining extraction quality:

- **💸 Select Cost-Efficient Models**: Use smaller/distilled LLMs for extracting aspects and basic concepts (e.g. titles, payment amounts, dates).
- **⚖️ Consider a Decision Model for Closed Questions**: Boolean, label and rating concepts, as well as aspects, can be extracted with a decision model, which generates no text. (See :doc:`../decision_models/decision_models` for when a decision model is a good fit, and for what its requests cost.)
- **⚙️ Use Default Parameters**: All the extractions will be processed in as few model calls as possible.
- **📉 Enable Justifications Only When Necessary**: Do not use justifications for simple aspects or concepts. This will reduce the number of tokens generated.
- **📊 Monitor Usage and Cost**: Track model calls, token consumption, and cost to identify optimization opportunities.


.. literalinclude:: ../../../dev/usage_examples/docs/optimizations/optimization_cost.py
    :language: python
    :caption: Example of optimizing extraction for cost


.. _optimization-cost-request-label:

📨 What a Request Sends
------------------------

Each request to an LLM sends the system message, the instructions of the task with the aspects or concepts of the request, and the content:

- **The system message**: the default one is written for extraction from documents, and you can replace it, or send none, with ``system_message`` (see :doc:`../llms/llm_config`).
- **The instructions**: fixed text for the kind of request, with more of it for a reasoning model, and the descriptions of the aspects or concepts, with their structures, labels and examples.
- **The content**: for concepts without references, the text of the document or of the aspect as it is. For aspects, and for concepts with references, each paragraph with its ID in markup, and its additional context, if any; with references to sentences, each sentence with its own ID. For a document of many short paragraphs, e.g. lines of a log, the markup is a noticeable part of the request.

The system message and the instructions are sent again with each request: the more requests an extraction takes, e.g. with ``max_items_per_call`` or ``max_paragraphs_to_analyze_per_call`` (see :doc:`../llms/llm_extraction_methods`), the more often they are paid for. To see the requests, and count their tokens, before any is paid for, capture them with a stand-in for the LLM (see :doc:`../testing`). Count with the tokenizer of the model where you have it: ``litellm.token_counter``, with which ContextGem checks a request against the context window, counts with ``tiktoken`` for a model whose tokenizer it does not have, so that its count for such a model, e.g. a Claude model, is an estimate.
