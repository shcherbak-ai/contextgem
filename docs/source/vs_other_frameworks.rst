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

:og:description: ContextGem: Comparison with other frameworks

ContextGem and other frameworks
================================

Due to ContextGem's powerful abstractions, it is the **easiest and fastest way** to build LLM extraction workflows for document analysis.


✏️ Basic Example
------------------

Below is a basic example of an extraction workflow - *extraction of anomalies from a document* - implemented side-by-side in ContextGem and other frameworks. (All implementations are self-contained. Comparison as of 24 March 2025.)

Even implementing this basic extraction workflow requires significantly more effort in other frameworks:

* 🔧 **Manual model definition**: Developers must define Pydantic validation models for structured output
* 📝 **Prompt engineering**: Crafting comprehensive prompts that guide the LLM effectively
* 🔄 **Output parsing logic**: Setting up parsers to handle the LLM's response
* 📄 **Reference mapping**: Writing custom logic for mapping references in the source document

In contrast, ContextGem handles all these complexities automatically. Users simply describe what to extract in natural language, provide basic configuration parameters, and the framework takes care of the rest.


.. tab-set::

    .. tab-item:: **ContextGem**

        :bdg-success:`⚡ Fastest way`

        ContextGem is the fastest and easiest way to implement an LLM extraction workflow. All the boilerplate code is handled behind the scenes.

        **Major time savers:**

        * ⌨️ **Simple syntax**: ContextGem uses a simple, intuitive API that requires minimal code
        * 📝 **Automatic prompt engineering**: ContextGem automatically constructs a prompt tailored to the extraction task
        * 🔄 **Automatic model definition**: ContextGem automatically defines the Pydantic model for structured output
        * 🧩 **Automatic output parsing**: ContextGem automatically parses the LLM's response
        * 🔍 **Automatic reference tracking**: Precise references are automatically extracted and mapped to the original document
        * 📏 **Flexible reference granularity**: References can be tracked at different levels (paragraphs, sentences)

        .. literalinclude:: ../../dev/usage_examples/readme/quickstart_concept.py
            :language: python
            :caption: Anomaly extraction example (ContextGem)

    .. tab-item:: LangChain

        LangChain is a popular and versatile framework for building LLM applications through composable components. It offers excellent flexibility and a rich ecosystem of integrations. While powerful, feature-rich, and widely adopted in the industry, it requires more manual configuration and setup work for structured data extraction tasks compared to ContextGem's streamlined approach.

        **Development overhead:**

        * 📝 **Manual prompt engineering**: Crafting comprehensive prompts that guide the LLM effectively
        * 🔧 **Manual model definition**: Developers must define Pydantic validation models for structured output
        * 🧩 **Manual output parsing**: Setting up parsers to handle the LLM's response
        * 🔍 **Manual reference mapping**: Writing custom logic for mapping references

        .. literalinclude:: ../../dev/usage_examples/vs_other_frameworks/basic/langchain.py
            :language: python
            :caption: Anomaly extraction example (LangChain)

    .. tab-item:: LlamaIndex

        LlamaIndex is a powerful and versatile framework for building LLM applications with data, particularly excelling at RAG workflows and document retrieval. It offers a comprehensive set of tools for data indexing and querying. While highly effective for its intended use cases, for structured data extraction tasks (non-RAG setup), it requires more manual configuration and setup work compared to ContextGem's streamlined approach.

        **Development overhead:**

        * 📝 **Manual prompt engineering**: Crafting comprehensive prompts that guide the LLM effectively
        * 🔧 **Manual model definition**: Developers must define Pydantic validation models for structured output
        * 🧩 **Manual output parsing**: Setting up parsers to handle the LLM's response
        * 🔍 **Manual reference mapping**: Writing custom logic for mapping references

        .. literalinclude:: ../../dev/usage_examples/vs_other_frameworks/basic/llama_index.py
            :language: python
            :caption: Anomaly extraction example (LlamaIndex)

    .. tab-item:: LlamaIndex (RAG)

        LlamaIndex with RAG setup is a powerful and sophisticated framework for document retrieval and analysis, offering exceptional capabilities for knowledge-intensive applications. Its comprehensive architecture excels at handling complex document interactions and information retrieval tasks across large document collections. While it provides robust and versatile capabilities for building advanced document-based applications, it does require more manual configuration and specialized setup for structured extraction tasks compared to ContextGem's streamlined and intuitive approach.

        **Development overhead:**

        * 📝 **Manual prompt engineering**: Crafting comprehensive prompts that guide the LLM effectively
        * 🔧 **Manual model definition**: Developers must define Pydantic validation models for structured output
        * 🧩 **Manual output parsing**: Setting up parsers to handle the LLM's response
        * 🔍 **Complex reference mapping**: Getting precise references correctly requires additional config, such as setting up a sentence splitter,  CitationQueryEngine, adjusting chunk sizes, etc.

        .. literalinclude:: ../../dev/usage_examples/vs_other_frameworks/basic/llama_index_rag.py
            :language: python
            :caption: Anomaly extraction example (LlamaIndex RAG)

    .. tab-item:: Instructor

        Instructor is a popular framework that specializes in structured data extraction with LLMs using Pydantic. It offers excellent type safety and validation capabilities, making it a solid choice for many extraction tasks. While powerful for structured outputs, Instructor requires more manual setup for document analysis workflows.

        **Development overhead:**

        * 📝 **Manual prompt engineering**: Crafting comprehensive prompts that guide the LLM effectively
        * 🔧 **Manual model definition**: Developers must define Pydantic validation models for structured output
        * 🔍 **Manual reference mapping**: Writing custom logic for mapping references

        .. literalinclude:: ../../dev/usage_examples/vs_other_frameworks/basic/instructor.py
            :language: python
            :caption: Anomaly extraction example (Instructor)


🔬 Advanced Example
---------------------

As use cases grow more complex, the development overhead of alternative frameworks becomes increasingly evident, while ContextGem's abstractions deliver substantial time savings. As extraction steps stack up, the implementation with other frameworks quickly becomes *non-scalable*:

* 📝 **Manual prompt engineering**: Crafting comprehensive prompts for each extraction step
* 🔧 **Manual model definition**: Defining Pydantic validation models for each element of extraction
* 🧩 **Manual output parsing**: Setting up parsers to handle the LLM's response
* 🔍 **Manual reference mapping**: Writing custom logic for mapping references
* 📄 **Complex pipeline configuration**: Writing custom logic for pipeline configuration and extraction components
* 📊 **Implementing usage and cost tracking callbacks**, which quickly increases in complexity when multiple LLMs are used in the pipeline
* 🔄 **Complex concurrency setup**: Implementing complex concurrency logic with asyncio
* 📝 **Embedding examples in prompts**: Writing output examples directly in the custom prompts
* 📋 **Manual result aggregation**: Need to write code to collect and organize results

Below is a more advanced example of an extraction workflow - *using an extraction pipeline for multiple documents, with concurrency and cost tracking* - implemented side-by-side in ContextGem and other frameworks. (All implementations are self-contained. Comparison as of 24 March 2025.)

.. tab-set::

    .. tab-item:: **ContextGem**

        :bdg-success:`⚡ Fastest way`

        ContextGem is the fastest and easiest way to implement an LLM extraction workflow. All the boilerplate code is handled behind the scenes.

        **Major time savers:**

        * ⌨️ **Simple syntax**: ContextGem uses a simple, intuitive API that requires minimal code
        * 🔄 **Automatic model definition**: ContextGem automatically defines the Pydantic model for structured output
        * 📝 **Automatic prompt engineering**: ContextGem automatically constructs a prompt tailored to the extraction task
        * 🧩 **Automatic output parsing**: ContextGem automatically parses the LLM's response
        * 🔍 **Automatic reference tracking**: Precise references are automatically extracted and mapped to the original document
        * 📏 **Flexible reference granularity**: References can be tracked at different levels (paragraphs, sentences)
        * 📄 **Easy pipeline definition**: Simple, declarative syntax for defining the extraction pipeline involving multiple LLMs, in a few lines of code
        * 💰 **Automated usage and cost tracking**: Built-in token counting and cost calculation without additional setup
        * 🔄 **Built-in concurrency**: Concurrent execution of extraction steps with a simple switch ``use_concurrency=True``
        * 📊 **Easy example definition**: Output examples can be easily defined without modifying any prompts
        * 📋 **Built-in result aggregation**: Results are automatically collected and organized in a unified storage model (document)

        .. literalinclude:: ../../dev/usage_examples/docs/advanced/advanced_multiple_docs_pipeline.py
            :language: python
            :caption: Extraction pipeline example (ContextGem)

    .. tab-item:: LangChain

        LangChain provides a powerful and flexible framework for building LLM applications with excellent composability and a rich ecosystem of integrations. While it offers great versatility for many use cases, it does require additional manual setup and configuration for complex extraction workflows.

        **Development overhead:**

        * 📝 **Manual prompt engineering**: Must craft detailed prompts for each extraction step
        * 🔧 **Manual model definition**: Need to define Pydantic models and output parsers for structured data
        * 🧩 **Complex chain configuration**: Requires manual setup of chains and their connections involving multiple LLMs
        * 🔍 **Manual reference mapping**: Must implement custom logic to track source references
        * 🔄 **Complex concurrency setup**: Implementing concurrent processing requires additional setup with asyncio
        * 💰 **Cost tracking setup**: Requires custom logic for cost tracking for each LLM
        * 💾 **No unified storage model**: Need to write additional code to collect and organize results

        .. literalinclude:: ../../dev/usage_examples/vs_other_frameworks/advanced/langchain.py
            :language: python
            :caption: Extraction pipeline example (LangChain)

    .. tab-item:: LlamaIndex

        LlamaIndex provides a robust data framework for LLM applications with excellent capabilities for knowledge retrieval and RAG. It offers powerful tools for working with documents and structured data, though implementing complex extraction workflows may require some additional configuration to fully leverage its capabilities.

        **Development overhead:**

        * 📝 **Manual prompt engineering**: Must craft detailed prompts for each extraction task
        * 🔧 **Manual model definition**: Need to define Pydantic models and output parsers for structured data
        * 🧩 **Pipeline setup**: Requires manual configuration of extraction pipeline components involving multiple LLMs
        * 🔍 **Limited reference tracking**: Basic source tracking, but requires additional work for fine-grained references
        * 📊 **Embedding examples in prompts**: Examples must be manually incorporated into prompts
        * 🔄 **Complex concurrency setup**: Implementing concurrent processing requires additional setup with asyncio
        * 💰 **Cost tracking setup**: Requires custom logic for cost tracking for each LLM
        * 💾 **No unified storage model**: Need to write additional code to collect and organize results

        .. literalinclude:: ../../dev/usage_examples/vs_other_frameworks/advanced/llama_index.py
            :language: python
            :caption: Extraction pipeline example (LlamaIndex)

    .. tab-item:: Instructor

        Instructor is a powerful library focused on structured outputs from LLMs with strong typing support through Pydantic. It excels at extracting structured data with validation, but requires additional work to build complex extraction pipelines.

        **Development overhead:**

        * 📝 **Manual prompt engineering**: Crafting comprehensive prompts that guide the LLM effectively
        * 🔧 **Manual model definition**: Developers must define Pydantic validation models for structured output
        * 🧩 **Manual pipeline assembly**: Requires custom code to connect extraction components involving multiple LLMs
        * 🔍 **Manual reference mapping**: Must implement custom logic to track source references
        * 📊 **Embedding examples in prompts**: Examples must be manually incorporated into prompts
        * 🔄 **Complex concurrency setup**: Implementing concurrent processing requires additional setup with asyncio
        * 💰 **Cost tracking setup**: Requires custom logic for cost tracking for each LLM

        .. literalinclude:: ../../dev/usage_examples/vs_other_frameworks/advanced/instructor.py
            :language: python
            :caption: Extraction pipeline example (Instructor)
