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

:og:description: ContextGem: Effortless LLM extraction from documents

.. image:: _static/contextgem_readme_header.png
   :width: 100%
   :alt: ContextGem logo
   :target: #
   :align: center

|

Welcome to ContextGem Documentation!
=====================================

ContextGem is a free, open-source LLM framework that makes it radically easier to extract structured data and insights from documents — with minimal code.

|

.. grid:: 1 1 2 2
    :gutter: 3
    :padding: 0
    
    .. grid-item-card:: 📚 Project Description
        :link: motivation
        :link-type: doc
        :class-card: sd-border-0 sd-shadow-sm sd-card-hover
        
        Learn about the motivation, comparisons with other frameworks, and how ContextGem works.
        
        +++
        .. button-ref:: motivation
            :ref-type: doc
            :click-parent:
            :color: primary
            :outline:
            :expand:
            
            Learn More
            
    .. grid-item-card:: 🚀 Getting Started
        :link: installation
        :link-type: doc
        :class-card: sd-border-0 sd-shadow-sm sd-card-hover
        
        Instructions to install ContextGem and quickly start using it.
        
        +++
        .. button-ref:: installation
            :ref-type: doc
            :click-parent:
            :color: primary
            :outline:
            :expand:
            
            Get Started

    .. grid-item-card:: 📄 Documents
        :link: documents/document_config
        :link-type: doc
        :class-card: sd-border-0 sd-shadow-sm sd-card-hover
        
        Learn how to create and configure documents to extract information (aspects and concepts) from.
        
        +++
        .. button-ref:: documents/document_config
            :ref-type: doc
            :click-parent:
            :color: primary
            :outline:
            :expand:
            
            Create Documents

    .. grid-item-card:: 🔄 Document Converters
        :link: converters/docx
        :link-type: doc
        :class-card: sd-border-0 sd-shadow-sm sd-card-hover

        Learn how to use ContextGem's built-in document converters for files such as DOCX.

        +++
        .. button-ref:: converters/docx
            :ref-type: doc
            :click-parent:
            :color: primary
            :outline:
            :expand:

            Convert Files
    
    .. grid-item-card:: 📋 Extracting Aspects
        :link: aspects/aspects
        :link-type: doc
        :class-card: sd-border-0 sd-shadow-sm sd-card-hover

        Learn how to identify and extract specific document sections like clauses, chapters, or terms using ContextGem's Aspects API.

        +++
        .. button-ref:: aspects/aspects
            :ref-type: doc
            :click-parent:
            :color: primary
            :outline:
            :expand:

            Extract Aspects

    .. grid-item-card:: 🎯 Extracting Concepts
        :link: concepts/supported_concepts
        :link-type: doc
        :class-card: sd-border-0 sd-shadow-sm sd-card-hover

        Learn how to extract and infer structured data like JSON objects, strings, numbers, dates, booleans, ratings, and labels from documents using ContextGem's Concepts API.

        +++
        .. button-ref:: concepts/supported_concepts
            :ref-type: doc
            :click-parent:
            :color: primary
            :outline:
            :expand:

            Extract Concepts

    .. grid-item-card:: 🔀 Extraction Pipelines
        :link: pipelines/extraction_pipelines
        :link-type: doc
        :class-card: sd-border-0 sd-shadow-sm sd-card-hover

        Learn how to create reusable extraction pipelines that combine aspects and concepts for consistent document analysis across multiple files.

        +++
        .. button-ref:: pipelines/extraction_pipelines
            :ref-type: doc
            :click-parent:
            :color: primary
            :outline:
            :expand:

            Build Pipelines

    .. grid-item-card:: 🤖 Large Language Models
        :link: llms/supported_llms
        :link-type: doc
        :class-card: sd-border-0 sd-shadow-sm sd-card-hover
        
        Learn about supported cloud LLM providers and local models, and how to configure and use them for extraction.
        
        +++
        .. button-ref:: llms/supported_llms
            :ref-type: doc
            :click-parent:
            :color: primary
            :outline:
            :expand:
            
            Explore LLMs

    .. grid-item-card:: ⚡ Advanced Usage
        :link: advanced_usage
        :link-type: doc
        :class-card: sd-border-0 sd-shadow-sm sd-card-hover
        
        Explore advanced features and techniques for extracting data from documents.
        
        +++
        .. button-ref:: advanced_usage
            :ref-type: doc
            :click-parent:
            :color: primary
            :outline:
            :expand:
            
            Learn More

    .. grid-item-card:: ⚙️ Optimization Guide
        :link: optimizations/optimization_choosing_llm
        :link-type: doc
        :class-card: sd-border-0 sd-shadow-sm sd-card-hover
        
        Learn how to optimize your extraction pipeline for accuracy, cost, and performance.
        
        +++
        .. button-ref:: optimizations/optimization_choosing_llm
            :ref-type: doc
            :click-parent:
            :color: primary
            :outline:
            :expand:
            
            Optimize
    
    .. grid-item-card:: 💾 Serialization
        :link: serialization
        :link-type: doc
        :class-card: sd-border-0 sd-shadow-sm sd-card-hover
        
        Learn how to serialize and deserialize ContextGem objects for storage and transfer.
        
        +++
        .. button-ref:: serialization
            :ref-type: doc
            :click-parent:
            :color: primary
            :outline:
            :expand:
            
            Save & Load
    
    .. grid-item-card:: 📖 API Reference
        :link: api/documents
        :link-type: doc
        :class-card: sd-border-0 sd-shadow-sm sd-card-hover
        
        Complete API documentation for all ContextGem modules and classes.
        
        +++
        .. button-ref:: api/documents
            :ref-type: doc
            :click-parent:
            :color: primary
            :outline:
            :expand:
            
            View API

|

Indices and tables
-------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. Navigation structure (hidden from page, visible in sidebar)

.. toctree::
   :maxdepth: 2
   :caption: Project Description
   :hidden:
   
   motivation
   vs_other_frameworks
   how_it_works

.. toctree::
   :maxdepth: 2
   :caption: Getting Started
   :hidden:
   
   installation
   quickstart

.. toctree::
   :maxdepth: 2
   :caption: Documents
   :hidden:
   
   documents/document_config

.. toctree::
   :maxdepth: 2
   :caption: Document Converters
   :hidden:
   
   converters/docx

.. toctree::
   :maxdepth: 2
   :caption: Extracting Aspects
   :hidden:
   
   aspects/aspects

.. toctree::
   :maxdepth: 2
   :caption: Extracting Concepts
   :hidden:
   
   concepts/supported_concepts
   concepts/string_concept
   concepts/boolean_concept
   concepts/numerical_concept
   concepts/date_concept
   concepts/rating_concept
   concepts/json_object_concept
   concepts/label_concept

.. toctree::
   :maxdepth: 2
   :caption: Extraction Pipelines
   :hidden:
   
   pipelines/extraction_pipelines

.. toctree::
   :maxdepth: 2
   :caption: Large Language Models
   :hidden:
   
   llms/supported_llms
   llms/llm_config
   llms/llm_extraction_methods

.. toctree::
   :maxdepth: 2
   :caption: Advanced Usage
   :hidden:
   
   advanced_usage
   logging_config

.. toctree::
   :maxdepth: 2
   :caption: Optimization Guide
   :hidden:
   
   optimizations/optimization_choosing_llm
   optimizations/optimization_accuracy
   optimizations/optimization_speed
   optimizations/optimization_cost
   optimizations/optimization_long_docs
   optimizations/optimization_small_llm_troubleshooting

.. toctree::
   :maxdepth: 2
   :caption: Serialization
   :hidden:
   
   serialization

.. toctree::
   :maxdepth: 2
   :caption: API Reference
   :hidden:
   
   api/documents
   api/converters
   api/aspects
   api/concepts
   api/examples
   api/llms
   api/data_models
   api/utils
   api/images
   api/paragraphs
   api/sentences
   api/pipelines
   api/decorators
