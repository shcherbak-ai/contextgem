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

:og:description: ContextGem: Creating Documents

Creating Documents
===================

This guide explains how to create and configure :class:`~contextgem.public.documents.Document` instances to process textual and visual content for analysis.

Documents serve as the container for the content from which information (aspects and concepts) can be extracted.


⚙️ Configuration Parameters
----------------------------

The minimum configuration for a document requires either ``raw_text``, ``paragraphs``, or ``images``:

.. literalinclude:: ../../../dev/usage_examples/docstrings/documents/def_document.py
   :language: python
   :caption: Document creation

|

The :class:`~contextgem.public.documents.Document` class accepts the following parameters:

.. list-table::
   :header-rows: 1
   :widths: 25 15 15 45

   * - Parameter
     - Type
     - Default Value
     - Description
   * - ``raw_text``
     - ``str | None``
     - ``None``
     - The main text of the document as a single string.
   * - ``paragraphs``
     - ``list[Paragraph]``
     - ``[]``
     - List of :class:`~contextgem.public.paragraphs.Paragraph` instances in consecutive order as they appear in the document. Normally auto-populated from ``raw_text``.
   * - ``images``
     - ``list[Image]``
     - ``[]``
     - List of :class:`~contextgem.public.images.Image` instances attached to or representing the document. Used for visual content analysis.
   * - ``aspects``
     - ``list[Aspect]``
     - ``[]``
     - List of :class:`~contextgem.public.aspects.Aspect` instances associated with the document for focused analysis. Must have unique names and descriptions. See :doc:`../aspects/aspects` for more details.
   * - ``concepts``
     - ``list[_Concept]``
     - ``[]``
     - List of ``_Concept`` instances associated with the document for information extraction. Must have unique names and descriptions. See supported concept types in :doc:`../concepts/supported_concepts`.
   * - ``paragraph_segmentation_mode``
     - ``Literal["newlines", "sat"]``
     - ``"newlines"``
     - Mode for paragraph segmentation. ``"newlines"`` splits on newline characters, ``"sat"`` uses a SaT (Segment Any Text) model for intelligent segmentation.
   * - ``sat_model_id``
     - ``SaTModelId``
     - ``"sat-3l-sm"``
     - SaT model ID for paragraph/sentence segmentation or a local path to a SaT model. See `wtpsplit models <https://github.com/segment-any-text/wtpsplit>`_ for available options.
   * - ``pre_segment_sentences``
     - ``bool``
     - ``False``
     - Whether to pre-segment sentences during Document initialization. When ``False``, sentence segmentation is deferred until sentences are actually needed, improving initialization performance.


🔄 DOCX Document Conversion
----------------------------

ContextGem provides a built-in :class:`~contextgem.public.converters.docx.DocxConverter` to easily transform DOCX files into LLM-ready :class:`~contextgem.public.documents.Document` instances.

For detailed usage examples and configuration options, see :doc:`../converters/docx`.


🎯 Adding Aspects and Concepts for Extraction
-----------------------------------------------

Before extracting information from a document with an LLM, you must define and add **aspects** and **concepts** to your document instance. These components serve as the foundation for targeted analysis and structured information extraction.

**Aspects** define the text segments (sections, topics, themes) to be extracted from the document. They can be combined with concepts for comprehensive analysis.

**Concepts** define specific data points to be extracted or inferred from the document content: entities, insights, structured objects, classifications, numerical calculations, dates, ratings, and assessments.

For detailed guidance on creating and configuring these components, see:

- :doc:`../aspects/aspects` - Complete guide to defining and using aspects
- :doc:`../concepts/supported_concepts` - All available concept types and how to use them
