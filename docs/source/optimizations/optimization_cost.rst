:og:description: ContextGem: Optimizing for Cost

.. meta::
   :description: How to lower the cost of extraction: cost-efficient models, decision models for closed questions, default parameters, sparing justifications, and usage and cost monitoring.

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
