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
- **⚡ Optional: Enable Concurrency**: Enable running extractions concurrently if your API setup permits. This will reduce the overall processing time. (See :doc:`optimization_speed` for guidance on configuring concurrency.)

Since each use case has unique requirements, experiment with different configurations to find your optimal setup.

.. literalinclude:: ../../../dev/usage_examples/docs/optimizations/optimization_long_docs.py
    :language: python
    :caption: Example of configuring LLM extraction for long documents
