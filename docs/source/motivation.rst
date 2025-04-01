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

Why ContextGem?
================

ContextGem is an LLM framework designed to strike the right balance between ease of use, customizability, and accuracy for structured data and insights extraction from documents.

ContextGem offers the **easiest and fastest way** to build LLM extraction workflows for document analysis through powerful abstractions of most time consuming parts.


⏱️ Development Overhead of Other Frameworks
--------------------------------------------

Most popular LLM frameworks for extracting structured data from documents require extensive boilerplate code to extract even basic information. As a developer using these frameworks, you're typically expected to:

.. grid:: 2
    :gutter: 3

    .. grid-item-card:: 📝 Prompt Engineering
        :class-card: sd-border-0
        
        * Write custom prompts from scratch for each extraction scenario
        * Maintain different prompt templates for different extraction workflows
        * Adapt prompts manually when extraction requirements change

    .. grid-item-card:: 🔧 Technical Implementation
        :class-card: sd-border-0
        
        * Define your own data models and implement validation logic
        * Implement complex chaining for multi-LLM workflows
        * Implement nested context extraction logic (*e.g. document > sections > paragraphs > entities*)
        * Configure text segmentation logic for correct reference mapping
        * Configure concurrent I/O processing logic to speed up complex extraction workflows

**Result:** All these limitations significantly increase development time and complexity.


💡 The ContextGem Solution
---------------------------

ContextGem addresses these challenges by providing a flexible, intuitive framework that extracts structured data and insights from documents with minimal effort. Complex, most time-consuming parts are handled with **powerful abstractions**, eliminating boilerplate code and reducing development overhead.

With ContextGem, you benefit from a "batteries included" approach, coupled with simple, intuitive syntax.


.. list-table:: ContextGem and Other Open-Source LLM Frameworks
   :header-rows: 1
   :widths: 3 45 10 20

   * - 
     - Key built-in abstractions
     - **ContextGem**
     - Other frameworks*

   * - 💎 
     - **Automated dynamic prompts**
       
       Automatically constructs comprehensive prompts for your specific extraction needs.
     - ✅
     - ❌

   * - 💎 
     - **Automated data modelling and validators**
    
       Automatically creates data models and validation logic.
     - ✅
     - ❌
     
   * - 💎 
     - **Precise granular reference mapping (paragraphs & sentences)**
    
       Automatically maps extracted data to the relevant parts of the document, which will always match in the source document, with customizable granularity.
     - ✅
     - ❌

   * - 💎 
     - **Justifications (reasoning backing the extraction)**
     
       Automatically provides justifications for each extraction, with customizable granularity.
     - ✅
     - ❌
     
   * - 💎 
     - **Neural segmentation (SaT)**
     
       Automatically segments the document into paragraphs and sentences using state-of-the-art SaT models, compatible with many languages.
     - ✅
     - ❌
     
   * - 💎 
     - **Multilingual support (I/O without prompting)**
       
       Supports multiple languages in input and output without additional prompting.
     - ✅
     - ❌

   * - 💎 
     - **Single, unified extraction pipeline (declarative, reusable, fully serializable)**
       
       Allows to define a complete extraction workflow in a single, unified, reusable pipeline, using simple declarative syntax.
     - ✅
     - ⚠️

   * - 💎 
     - **Grouped LLMs with role-specific tasks**
     
       Allows to easily group LLMs with different roles to process role-specific tasks in the pipeline.
     - ✅
     - ⚠️

   * - 💎 
     - **Nested context extraction**
    
       Automatically manages nested context based on the pipeline definition (e.g. document > aspects > sub-aspects > concepts).
     - ✅
     - ⚠️

   * - 💎 
     - **Unified, fully serializable results storage model (document)**
    
       All extraction results are stored on the document object, including aspects, sub-aspects, and concepts. This object is fully serializable, and all the extraction results can be restored, with just one line of code.
     - ✅
     - ⚠️

   * - 💎 
     - **Extraction task calibration with examples**
    
       Allows to easily define and attach output examples that guide the LLM's extraction behavior, without manually modifying prompts.
     - ✅
     - ⚠️

   * - 💎 
     - **Built-in concurrent I/O processing**
    
       Automatically manages concurrent I/O processing to speed up complex extraction workflows, with a simple switch (``use_concurrency=True``).
     - ✅
     - ⚠️

   * - 💎 
     - **Automated usage & costs tracking**
    
       Automatically tracks usage (calls, tokens, costs) of all LLM calls.
     - ✅
     - ⚠️

   * - 💎 
     - **Fallback and retry logic**
     
       Built-in retry logic and easily attachable fallback LLMs.
     - ✅
     - ✅

   * - 💎 
     - **Multiple LLM providers**

       Compatible with a wide range of commercial and locally hosted LLMs.
     - ✅
     - ✅

| ✅ - fully supported - no additional setup required
| ⚠️ - partially supported - requires additional setup
| ❌ - not supported - requires custom logic


    \* See :doc:`vs_other_frameworks` for specific implementation examples comparing ContextGem with other popular open-source LLM frameworks. (Comparison as of 24 March 2025.)


🎯 Focused Approach
---------------------

ContextGem is intentionally optimized for **in-depth single-document analysis** to deliver maximum extraction accuracy and precision. While this focused approach enables superior results for individual documents, ContextGem currently does not support cross-document querying or corpus-wide information retrieval. For these use cases, traditional RAG (Retrieval-Augmented Generation) systems over document collections (e.g. LlamaIndex) remain more appropriate.
