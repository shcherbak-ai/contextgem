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

:og:description: ContextGem: Extraction Pipelines for reusable document analysis workflows

Extraction Pipelines
====================

:class:`~contextgem.public.pipelines.ExtractionPipeline` is a powerful component that enables you to create reusable collections of predefined aspects and concepts for consistent document analysis. Pipelines serve as templates that can be applied to multiple documents, ensuring standardized data extraction across your application.


📝 Overview
-------------

Extraction pipelines package common extraction patterns into reusable units, allowing you to:

* **Standardize document processing**: Define a consistent set of aspects and concepts once, then apply them to multiple documents
* **Create reusable templates**: Build domain-specific pipelines (e.g., contract analysis, invoice processing, report analysis)
* **Ensure consistent analysis**: Maintain uniform extraction criteria across document batches
* **Simplify workflow management**: Organize complex extraction workflows into manageable, reusable components

Pipelines are particularly valuable when processing multiple documents of the same type, where you need to extract the same categories of information consistently.


⭐ Key Features
-----------------

Template-Based Extraction
~~~~~~~~~~~~~~~~~~~~~~~~~~

Pipelines act as extraction templates that define what information to extract from documents. Once created, a pipeline can be assigned to any number of documents, ensuring consistent analysis criteria.

Aspect and Concept Organization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Pipelines can contain both:

* **Aspects**: For extracting document sections and organizing content hierarchically
* **Concepts**: For extracting specific data points with intelligent inference

This allows you to create comprehensive extraction workflows that combine broad content organization with detailed data extraction.

Reusability and Scalability
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A single pipeline can be applied to multiple documents, making it ideal for batch processing, automated workflows, and applications that need to process similar document types repeatedly.


💻 Basic Usage
--------------

Simple Pipeline Creation
~~~~~~~~~~~~~~~~~~~~~~~~

Here's how to create and use a basic extraction pipeline:

.. literalinclude:: ../../../dev/usage_examples/docstrings/pipelines/def_pipeline.py
   :language: python

Pipeline Assignment to Documents
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Once created, pipelines can be easily assigned to documents:

.. code-block:: python

   from contextgem import Document, ExtractionPipeline
   
   # Create your pipeline
   my_pipeline = ExtractionPipeline(aspects=[...], concepts=[...])
   
   # Create documents
   doc1 = Document(raw_text="First document content...")
   doc2 = Document(raw_text="Second document content...")
   
   # Assign the same pipeline to multiple documents
   doc1.assign_pipeline(my_pipeline)
   doc2.assign_pipeline(my_pipeline)
   
   # Now both documents have the same extraction configuration


⚙️ Parameters
--------------

When creating an :class:`~contextgem.public.pipelines.ExtractionPipeline`, you can configure the following parameters:

.. list-table::
   :widths: 20 15 15 50
   :header-rows: 1

   * - Parameter
     - Type
     - Default Value
     - Description
   * - ``aspects``
     - ``list[Aspect]``
     - ``[]``
     - *Optional*. List of :class:`~contextgem.public.aspects.Aspect` instances to extract from documents. Aspects represent structural categories of information and can contain their own sub-aspects and concepts for detailed analysis. See :doc:`../aspects/aspects` for more information.
   * - ``concepts``
     - ``list[_Concept]``
     - ``[]``
     - *Optional*. List of ``_Concept`` instances to identify within or infer from documents. These are document-level concepts that apply to the entire document content. See supported concept types in :doc:`../concepts/supported_concepts`.


📊 Pipeline Assignment
------------------------

The :meth:`~contextgem.public.documents.Document.assign_pipeline` method is used to apply a pipeline to a document. This method:

* **Assigns aspects and concepts**: Transfers the pipeline's aspects and concepts to the document
* **Validates compatibility**: Ensures no conflicts with existing document configuration

Assignment Options
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Basic assignment (will raise error if document already has aspects/concepts)
   document.assign_pipeline(my_pipeline)
   
   # Overwrite existing configuration
   document.assign_pipeline(my_pipeline, overwrite_existing=True)


🚀 Advanced Usage
-------------------

Multi-Document Processing
~~~~~~~~~~~~~~~~~~~~~~~~~

Pipelines excel at processing multiple documents of the same type. Here's a comprehensive example:

.. literalinclude:: ../../../dev/usage_examples/docs/advanced/advanced_multiple_docs_pipeline.py
   :language: python

.. raw:: html

   <a target="_blank" href="https://colab.research.google.com/github/shcherbak-ai/contextgem/blob/main/dev/notebooks/docs/advanced/advanced_multiple_docs_pipeline.ipynb">
     <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
   </a>

Pipeline Serialization
~~~~~~~~~~~~~~~~~~~~~~

Pipelines can be serialized for storage and later reuse:

.. code-block:: python

   # Serialize the pipeline
   pipeline_json = pipeline.to_json()  # or to_dict() / to_disk()

   # Deserialize the pipeline
   pipeline_deserialized = ExtractionPipeline.from_json(
       pipeline_json
   )  # or from_dict() / from_disk()


💡 Best Practices
-------------------

Pipeline Design
~~~~~~~~~~~~~~~~

* **Domain-specific organization**: Create pipelines tailored to specific document types (contracts, invoices, reports, etc.)
* **Logical grouping**: Group related aspects and concepts together for coherent analysis
* **Reusable templates**: Design pipelines to be generic enough for reuse across similar documents

Concept Placement Strategy
~~~~~~~~~~~~~~~~~~~~~~~~~~

* **Document-level concepts**: Place concepts that apply to the entire document in the pipeline's ``concepts`` list
* **Aspect-level concepts**: Place concepts that are specific to particular document sections within the relevant aspects
* **Avoid duplication**: Don't create similar concepts at both document and aspect levels


🎯 Example Use Cases
----------------------

Invoice Processing Pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   invoice_pipeline = ExtractionPipeline(
       concepts=[
           StringConcept(name="Vendor Name", description="Name of the vendor/supplier"),
           StringConcept(name="Invoice Number", description="Unique invoice identifier"),
           DateConcept(name="Invoice Date", description="Date the invoice was issued"),
           DateConcept(name="Due Date", description="Payment due date"),
           NumericalConcept(name="Total Amount", description="Total invoice amount"),
           StringConcept(name="Currency", description="Currency of the invoice"),
       ]
   )

Research Paper Analysis Pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   research_pipeline = ExtractionPipeline(
       aspects=[
           Aspect(name="Abstract", description="Paper abstract and summary"),
           Aspect(name="Methodology", description="Research methods and approach"),
           Aspect(name="Results", description="Findings and outcomes"),
           Aspect(name="Conclusions", description="Conclusions and implications"),
       ],
       concepts=[
           StringConcept(name="Research Field", description="Primary research domain"),
           StringConcept(name="Keywords", description="Paper keywords and topics"),
           DateConcept(name="Publication Date", description="When the paper was published"),
           RatingConcept(name="Novelty Score", description="Novelty of the research", rating_scale=(1, 10)),
       ]
   )


⚡ Pipeline Reuse Benefits
-----------------------------

* **Consistency**: Ensures all documents are processed with identical extraction criteria
* **Efficiency**: Eliminates the need to recreate aspects and concepts for each document
* **Maintainability**: Changes to extraction logic only need to be made in one place


📚 Related Documentation
-------------------------

* :doc:`../aspects/aspects` - Learn about aspect extraction
* :doc:`../concepts/supported_concepts` - Explore available concept types and how to use them
* :doc:`../advanced_usage` - See advanced pipeline usage examples
* :doc:`../llms/llm_extraction_methods` - Understand LLM extraction methods
* :doc:`../serialization` - Learn about pipeline serialization and storage