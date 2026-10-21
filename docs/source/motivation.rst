.. meta::
   :description: Why ContextGem: the problems that extracting structured data from documents with LLMs involves, and how the framework solves them so that you describe only what to extract.

Why ContextGem?
================

ContextGem is a framework for extracting structured data and insights from documents using LLMs and decision models.

**Describe, don't prompt.** You describe *what* to extract in natural language, and ContextGem handles *how* — prompt construction, data modeling, output parsing, source reference mapping, justifications, and pipeline orchestration are all built in.


🧩 What Document Extraction Requires
--------------------------------------

Building reliable structured-extraction workflows from documents typically involves solving a stack of recurring problems:

.. grid:: 2
    :gutter: 3

    .. grid-item-card:: 📝 Prompting and Modeling
        :class-card: sd-border-0

        * Crafting extraction prompts for each new task and adapting them when requirements change
        * Defining data models with validation logic for every output shape
        * Embedding few-shot examples that calibrate extraction behavior

    .. grid-item-card:: 🔧 Document and Pipeline Plumbing
        :class-card: sd-border-0

        * Mapping outputs back to precise locations in the source document
        * Handling nested context (*e.g. document > sections > paragraphs > entities*)
        * Orchestrating multi-step, multi-model workflows
        * Tracking usage and costs across providers
        * Configuring concurrent I/O to keep extraction fast


💎 The ContextGem Approach
----------------------------

ContextGem provides a "batteries included" answer to each of these problems, with simple, declarative syntax:

- **Prompting and modeling.** ContextGem auto-generates dynamic prompts and validated data models from your natural-language extraction targets, with optional few-shot calibration via attached examples.
- **Document and pipeline plumbing.** Aspects map back to precise paragraph- or sentence-level locations in the source, and so can the concepts that an LLM extracts. Nested context (document → aspects → sub-aspects → concepts) is handled automatically. The whole workflow is a single declarative, reusable, serializable pipeline. Concurrent I/O, fallback/retry, and cross-provider usage and cost tracking are built in.
- **The right model for each question.** In a model group, each aspect and concept goes to the model that suits it: an LLM where it takes reasoning, or a :doc:`decision model <decision_models/decision_models>` where a fast, low-cost answer is enough, and any model can fall back on others. A decision model answers closed questions (yes/no, one of your labels, a level of a scale) with the probability of each possible answer, so that you set the threshold.

For details on each capability, see the relevant feature pages in the sidebar.


🎯 Focused Approach
---------------------

ContextGem is built for in-depth analysis of individual documents. There is no search step that picks the chunks that look relevant, as in RAG: the model reads every paragraph of the text it extracts from, so no fact is missed because a search ranked its passage low (see :doc:`how_it_works`).

The trade-off: ContextGem does not query across documents or search a corpus. For these, RAG frameworks (e.g. LlamaIndex, Haystack) remain more appropriate.
