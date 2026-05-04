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

:og:description: Why ContextGem?

Why ContextGem?
================

ContextGem is an LLM framework designed to strike the right balance between ease of use, customizability, and accuracy for structured data and insights extraction from documents.

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
        * Orchestrating multi-step, multi-LLM workflows
        * Tracking usage and costs across providers
        * Configuring concurrent I/O to keep extraction fast

ContextGem provides a "batteries included" approach to all of these, with simple, declarative syntax.


💎 The ContextGem Approach
----------------------------

ContextGem provides a "batteries included" answer to each of these problems:

- **Prompting and modelling.** ContextGem auto-generates dynamic prompts and validated data models from your natural-language extraction targets, with optional few-shot calibration via attached examples.
- **Document and pipeline plumbing.** Outputs map back to precise paragraph- or sentence-level locations in the source. Nested context (document → aspects → sub-aspects → concepts) is handled automatically. The whole workflow is a single declarative, reusable, serializable pipeline. Concurrent I/O, fallback/retry, and cross-provider usage tracking are built in.

For details on each capability, see the relevant feature pages in the sidebar.


🎯 Focused Approach
---------------------

ContextGem is intentionally optimized for **in-depth single-document analysis** to deliver maximum extraction accuracy and precision. While this focused approach enables superior results for individual documents, ContextGem currently does not support cross-document querying or corpus-wide information retrieval. For these use cases, modern RAG frameworks (e.g. LlamaIndex) remain more appropriate.
