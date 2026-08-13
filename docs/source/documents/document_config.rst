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


🎯 Adding Aspects and Concepts for Extraction
-----------------------------------------------

Before extracting information from a document with an LLM, you must define and add **aspects** and **concepts** to your document instance. These components serve as the foundation for targeted analysis and structured information extraction.

**Aspects** define the text segments (sections, topics, themes) to be extracted from the document. They can be combined with concepts for comprehensive analysis.

**Concepts** define specific data points to be extracted or inferred from the document content: entities, insights, structured objects, classifications, numerical calculations, dates, ratings, and assessments.

For detailed guidance on creating and configuring these components, see:

- :doc:`../aspects/aspects` - Complete guide to defining and using aspects
- :doc:`../concepts/supported_concepts` - All available concept types and how to use them


.. _locating-paras-and-sents:

📍 Locating Paragraphs and Sentences
--------------------------------------

Extraction results reference the document's own :class:`~contextgem.public.paragraphs.Paragraph` and :class:`~contextgem.public.sentences.Sentence` objects (see e.g. :doc:`../aspects/aspects` for details on references). To find the position of such an object in the document, use :meth:`~contextgem.public.documents.Document.get_paragraph_index` and :meth:`~contextgem.public.documents.Document.get_sentence_index`:

.. code-block:: python

   # 0-based position of a paragraph in document.paragraphs
   para_index = document.get_paragraph_index(paragraph)

   # (paragraph_index, sentence_index) position of a sentence,
   # where sentence_index is 0-based within the paragraph's sentences
   para_index, sent_index = document.get_sentence_index(sentence)

Lookups are keyed by each object's unique ID rather than text equality, so paragraphs or sentences with identical text (e.g. duplicate clauses in a contract) resolve to their specific occurrences. Unique IDs are preserved by serialization, so lookups also work across ``to_dict()``/``from_dict()`` round-trips.

This makes the methods suitable for working with extraction references, e.g. citing where in the document extracted information was found, ordering references combined from multiple extracted items, or retrieving surrounding context:

.. code-block:: python

   # Human-readable citation for a reference paragraph
   ref_para = concept.extracted_items[0].reference_paragraphs[0]
   citation = f"Found in paragraph {document.get_paragraph_index(ref_para) + 1}"

   # Order references combined from multiple extracted items
   # (references within a single extracted item are already in document order)
   combined_refs = [
       para
       for item in concept.extracted_items
       for para in item.reference_paragraphs
   ]
   ordered_refs = sorted(combined_refs, key=document.get_paragraph_index)

   # Surrounding context of a reference paragraph
   ref_index = document.get_paragraph_index(ref_para)
   context_window = document.paragraphs[max(0, ref_index - 2) : ref_index + 3]
