:og:description: Why ContextGem?

.. meta::
   :description: Why ContextGem: the problems that extracting structured data from documents with LLMs involves, and how the framework solves them so that you describe only what to extract.

Why ContextGem?
================

ContextGem is a framework for extracting structured data and insights from documents with LLMs and decision models, designed to strike the right balance between ease of use, customizability, and accuracy.

You describe *what* to extract in natural language, and ContextGem handles *how* — prompt construction, data modelling, output parsing, source reference mapping, justifications, and pipeline orchestration are all built in.


🧩 What Document Extraction Requires
--------------------------------------

Building reliable structured-extraction workflows from documents typically involves solving a stack of recurring problems:

.. grid:: 2
    :gutter: 3

    .. grid-item-card:: 📝 Prompting and Modelling
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

- **Prompting and modelling.** ContextGem auto-generates dynamic prompts and validated data models from your natural-language extraction targets, with optional few-shot calibration via attached examples.
- **Document and pipeline plumbing.** Outputs map back to precise paragraph- or sentence-level locations in the source. Nested context (document → aspects → sub-aspects → concepts) is handled automatically. The whole workflow is a single declarative, reusable, serializable pipeline. Concurrent I/O, fallback/retry, and cross-provider usage tracking are built in.
- **The right model for each question.** Each aspect and concept can be routed to the model that suits it: an LLM where it takes reasoning, or a :doc:`decision model <decision_models/decision_models>`, which answers closed questions (yes/no, pick one option, rate on a scale) with a probability for each possible answer, so that you set the threshold.

For details on each capability, see the relevant feature pages in the sidebar.


🎯 Focused Approach
---------------------

ContextGem is intentionally optimized for **in-depth single-document analysis** to deliver maximum extraction accuracy and precision. While this focused approach enables superior results for individual documents, ContextGem currently does not support cross-document querying or corpus-wide information retrieval. For these use cases, modern RAG frameworks (e.g. LlamaIndex) remain more appropriate.
