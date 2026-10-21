.. meta::
   :description: How to create a Document from text, paragraphs or images, add aspects and concepts to it, give paragraphs additional context, locate paragraphs and sentences in it, use SaT models without network access, and release the memory of a SaT model.

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
     - List of :class:`~contextgem.public.paragraphs.Paragraph` instances in consecutive order as they appear in the document. Normally auto-populated from ``raw_text``. See :ref:`paragraph-additional-context` for providing your own paragraphs with additional context.
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
     - SaT model ID for paragraph/sentence segmentation or a local path to a folder with a SaT model. See `wtpsplit models <https://github.com/segment-any-text/wtpsplit>`_ for available options. A model given by its ID is downloaded from Hugging Face (``huggingface.co``, ``hf.co`` and their subdomains) when it is first used, and cached. See :ref:`sat-models-offline` for using a model where Hugging Face cannot be reached.
   * - ``pre_segment_sentences``
     - ``bool``
     - ``False``
     - Whether to pre-segment sentences during Document initialization. When ``False``, sentence segmentation is deferred until sentences are actually needed, improving initialization performance.
   * - ``custom_data``
     - ``dict``
     - ``{}``
     - *Optional*. Dictionary for storing any additional data that you want to associate with the document. This data must be JSON-serializable. It is not used for extraction.


🎯 Adding Aspects and Concepts for Extraction
-----------------------------------------------

Before extracting information from a document with a model, you must define and add **aspects** and **concepts** to your document instance.

**Aspects** define the text segments (sections, topics, themes) to be extracted from the document. They can be combined with concepts for comprehensive analysis.

**Concepts** define specific data points to be extracted or inferred from the document content: entities, insights, structured objects, classifications, numerical calculations, dates, ratings, and assessments.

For detailed guidance on creating and configuring these components, see:

- :doc:`../aspects/aspects` - Complete guide to defining and using aspects
- :doc:`../concepts/supported_concepts` - All available concept types and how to use them


.. _paragraph-additional-context:

🧩 Providing Additional Context for Paragraphs
-----------------------------------------------

When you create the paragraphs yourself, e.g. from the structured output of a document conversion tool, you can describe each paragraph in its ``additional_context``: its structural role, formatting, or position in the document. This lets the model take the structure of the document into account, while the text of the paragraphs, which is returned in references, stays clean:

.. literalinclude:: ../../../dev/usage_examples/docstrings/paragraphs/def_paragraph.py
   :language: python
   :caption: Paragraphs with additional context

Pass such paragraphs to the document via ``paragraphs``. For a file converted with Docling, :meth:`~contextgem.public.documents.Document.from_docling` creates the paragraphs and their additional context from the structure that Docling found in the file (see :doc:`docling`).

The additional context:

- should be a single-line string: it is cleaned before it is sent to a model, as the text of the paragraph is, i.e. control characters are removed, and whitespace, line breaks included, is collapsed to single spaces. A warning is logged if it has line breaks, as they are not kept. Additional context of which nothing is left is not sent;
- is passed to the LLM together with the paragraph text in the prompts that list paragraphs, i.e. when aspects are extracted, or when concepts are extracted with references (``add_references=True``). It is not used when concepts are extracted from the document text as a whole;
- is sent to a decision model together with the paragraph text when aspects are extracted, also with each sentence of the paragraph that the model is asked about, for an aspect with ``reference_depth="sentences"`` (see :ref:`decision-models-aspects-label`);
- is inherited by the sentences segmented from the paragraph. When sentence-level references are used (``reference_depth="sentences"``), the LLM gets it once, with the paragraph, as a decision model does: the additional context of a sentence is passed with the sentence only if it differs from that of its paragraph, e.g. for a paragraph that you create with its sentences;
- is preserved when the document is serialized.


.. _locating-paras-and-sents:

📍 Locating Paragraphs and Sentences
--------------------------------------

Extraction results reference the document's own :class:`~contextgem.public.paragraphs.Paragraph` and :class:`~contextgem.public.sentences.Sentence` objects (see e.g. :doc:`../aspects/aspects` for details on references): a paragraph in ``reference_paragraphs`` of an aspect, of a sub-aspect or of an extracted item is the same object as the paragraph in ``document.paragraphs``, and a sentence in ``reference_sentences`` is the same object as the sentence in the ``sentences`` of its paragraph. This holds after each extraction method, and in a document restored from its serialized form (see :doc:`../serialization`). What you change in such an object, e.g. its ``custom_data``, is therefore changed in the document, and the other way round.

To find the position of such an object in the document, use :meth:`~contextgem.public.documents.Document.get_paragraph_index` and :meth:`~contextgem.public.documents.Document.get_sentence_index`:

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

Paragraphs and sentences can also hold the numbers of the pages that they are on, in ``page_numbers``, which :meth:`~contextgem.public.documents.Document.from_docling` sets for a file with pages, e.g. a PDF file, and which you can set on the paragraphs that you create yourself. Sentences segmented from a paragraph inherit them. The ``reference_page_numbers`` of an aspect or of an extracted item are then the pages of its references:

.. code-block:: python

   # Pages of the references of an extracted item, e.g. [4, 5]
   pages = concept.extracted_items[0].reference_page_numbers


.. _sat-models-offline:

🔌 Using SaT Models Without Network Access
-------------------------------------------

A SaT model splits the text of a document into sentences, and into paragraphs with ``paragraph_segmentation_mode="sat"``. Sentences are segmented when they are needed: when an aspect or a concept is extracted with references to sentences (``reference_depth="sentences"``), or when the document is created with ``pre_segment_sentences=True``. Only the paragraphs without sentences are segmented: a paragraph created with its sentences keeps them, as does one that :meth:`~contextgem.public.documents.Document.from_docling` makes one sentence, e.g. a table row. A document whose paragraphs are split on newlines (the default) and that needs no sentences uses no SaT model.

A model given by its ID, e.g. the default ``"sat-3l-sm"``, is downloaded from Hugging Face the first time it is used, together with its tokenizer: several hundred megabytes in all. ``huggingface_hub`` keeps the files in its cache (``~/.cache/huggingface``, or the folder set in the ``HF_HOME`` environment variable), and reads them from there afterwards. Where Hugging Face cannot be reached when the application runs, e.g. in a container or on a server without internet access, prepare the model in one of these ways:

- **Fill the cache in advance**, where Hugging Face can be reached, e.g. when the container image is built, and keep the cache folder with the application. If the application uses another model than the default one, add its ``sat_model_id`` to the ``Document`` in the command:

  .. code-block:: bash

     # Download the default model and its tokenizer into the cache folder
     export HF_HOME=/opt/huggingface
     uv run python -c "from contextgem import Document; Document(raw_text='Text.', pre_segment_sentences=True)"

  When the application runs, set the same ``HF_HOME``, and ``HF_HUB_OFFLINE=1``, so that the files are read from the cache without asking Hugging Face whether they have changed. See the `environment variables of huggingface_hub <https://huggingface.co/docs/huggingface_hub/package_reference/environment_variables>`_.

- **Keep the model in a folder** of its own, and pass the path of the folder as ``sat_model_id``. The folder holds the files of the model (``config.json`` and ``model_optimized.onnx``) and its tokenizer (``tokenizer.json``), which you can download with the ``hf`` command of ``huggingface_hub``, installed with ContextGem:

  .. code-block:: bash

     uv run hf download segment-any-text/sat-3l-sm config.json model_optimized.onnx --local-dir models/sat-3l-sm
     uv run hf download facebookAI/xlm-roberta-base tokenizer.json --local-dir models/sat-3l-sm

  .. code-block:: python

     document = Document(raw_text=text, sat_model_id="models/sat-3l-sm")

  A model in such a folder is loaded without network access, and without the cache. If the folder has no ``tokenizer.json``, the tokenizer is downloaded from Hugging Face, or read from the cache, as for a model given by its ID.


.. _sat-models-memory:

♻️ Releasing a SaT Model
------------------------

A SaT model is loaded when a document first needs it, which takes about a second for the default model, and is kept loaded for the documents that follow, for as long as the process runs. Documents processed one after another, e.g. a batch of documents whose aspects or concepts are extracted with references to sentences, thus load the model once. Documents segmented at the same time, e.g. in threads, share one model. The async extraction methods, e.g. ``extract_all_async()``, run the model in a worker thread, so that the event loop goes on while it segments a document.

The model used last is kept: a document with another ``sat_model_id`` loads its model, which is then kept in place of the one before. The path of a folder with a model refers to the same model in any form, e.g. relative or absolute. A loaded model holds a large amount of memory: about a gigabyte for the default model, and more once it has segmented a long document, or several documents at the same time, as it keeps the memory that it used for them. To free it, e.g. in a long-running process that segments no more documents, call :func:`~contextgem.public.utils.release_sat_model`. A document that needs the model then loads it again.

.. literalinclude:: ../../../dev/usage_examples/docstrings/utils/release_sat_model.py
   :language: python
   :caption: Release the SaT model
