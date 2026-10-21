:og:description: ContextGem: Converting Files with Docling

.. meta::
   :description: How to create a Document from a PDF, DOCX, PPTX, XLSX, HTML or scanned file converted with Docling, with the structure of the file in the additional context of the paragraphs and the pages of each paragraph.

Converting Files with Docling
==============================

`Docling <https://github.com/docling-project/docling>`_ converts PDF, DOCX, PPTX, XLSX, HTML, scanned images and other files into a ``DoclingDocument``. :meth:`~contextgem.public.documents.Document.from_docling` creates a ContextGem document from it, in which each paragraph keeps what plain text loses:

* **its place in the structure of the file**, e.g. the headings of its section, or the column headers of its table row, which the models get together with the paragraph;
* **the pages it is on**, which give the pages of the items that refer to it.

.. figure:: ../_static/contextgem_docling.svg
   :width: 100%
   :alt: A two-page service agreement in PDF, converted with Docling, and what an LLM extracts from it. From the table of fees on page 1, the concept Fees gives three JSON objects, one for each row, e.g. the service Hosting with a monthly fee of 1200.0 EUR billed in advance, each cited on page 1. From section 3 on page 2, the aspect Termination gives back the heading and clauses 3.1 and 3.2, cited on page 2.
   :align: center
   :class: dark-light


📦 Installation
----------------

.. code-block:: bash

   uv add "contextgem[docling]"

Or using pip:

.. code-block:: bash

   pip install -U "contextgem[docling]"

The ``docling`` extra installs Docling, which runs its layout and table recognition models on PDF files and images, on PyTorch, which it installs. On Linux, PyTorch comes with the CUDA libraries of NVIDIA by default; for a CPU-only PyTorch, see the `installation guide of Docling <https://docling-project.github.io/docling/installation/>`_. The extra does not install on an Intel Mac (see :doc:`../installation`).

.. tip::
   A Docling document converted elsewhere, e.g. by a Docling Serve instance, or saved by Docling as JSON and loaded with ``DoclingDocument.load_from_json()``, needs only the document model of Docling, not Docling itself: ``uv add contextgem docling-core``.


🚀 From a File to Cited Facts
------------------------------

Convert the file with Docling, create the document with ``from_docling()``, and extract from it as from any other document. Below, an LLM extracts a concept and an aspect from a `two-page service agreement <https://github.com/shcherbak-ai/contextgem/blob/main/tests/other_files/docling/service_agreement.pdf>`_:

.. literalinclude:: ../../../dev/usage_examples/readme/quickstart_docling.py
   :language: python
   :caption: Extracting from a PDF file converted with Docling

Each fee comes from a row of the table, which the LLM gets with the header of each column, and the termination clauses from section 3. ``reference_page_numbers`` gives the pages that an aspect or an extracted item was found on (see :ref:`docling-pages-label`).


🧱 What the Document Is Made Of
--------------------------------

Each piece of text in the body of the file becomes a paragraph, in reading order. Its additional context says what the paragraph is, and where it is in the file:

.. figure:: ../_static/contextgem_docling_paragraphs.svg
   :width: 100%
   :alt: The heading, clause 2.1 and the table of section 2, on page 1 of the service agreement, and the paragraphs that from_docling() makes of them. The heading "2. Fees and Payment" is a paragraph with the additional context "Heading, level 1". Clause 2.1 has the context "Section: 2. Fees and Payment". The table caption "Table 1: Monthly fees" is a paragraph of its own. The table becomes a paragraph with its column headers, "Service; Monthly fee; Billing", then a paragraph for each row, with each value after the header of its column, e.g. "Service: Hosting; Monthly fee: 1,200 EUR; Billing: In advance", with the context "Table row", the caption of the table and the section. The page footer is left out. Each paragraph is on page 1.
   :align: center
   :class: dark-light

.. list-table::
   :header-rows: 1
   :widths: 22 48 30

   * - In the file
     - Paragraph
     - Additional context
   * - Title, heading
     - The text
     - ``Title``, ``Heading, level 2``
   * - Body text
     - The text
     - Only the context below, e.g. its section
   * - List item
     - The text, after the number of a numbered item, e.g. ``1. Invoices are payable within 30 days.``, or after its position in its list where the file gives no number, e.g. in HTML
     - ``List item``, ``List item, level 2``
   * - Table
     - A paragraph with the headers of the columns, if the table has them, then a paragraph for each row, with each value after the header of its column, e.g. ``Service: Hosting; Monthly fee: 1,200 EUR``
     - ``Table header``, ``Table row``, with the caption of the table
   * - Caption, footnote, code, formula
     - The text
     - e.g. ``Table caption``, ``Picture caption``, ``Footnote``, ``Code``
   * - Form field, key-value pair
     - The key and the value, e.g. ``Full name: Ada Lovelace`` or ``I agree to the terms: checked``
     - ``Form field``, ``Key-value pair``

Where the file has them, the additional context also gives:

* **the headings of the section** of the paragraph, e.g. ``Section: 3. Term and Termination > 3.2 Notice``;
* **the sheet or the slide**, in a spreadsheet or a presentation, e.g. ``Sheet: Q3 figures`` or ``Slide 2``;
* **the speaker** of a cue, in a subtitle file, e.g. WebVTT: ``Speaker: Ann Berg``;
* **the titles**, in a file with several: each slide of a presentation can have its own, and Docling makes a title of the heading of each chapter of an EPUB file and of each top-level heading (``#``) of a Markdown file. The titles are then the top level of the sections, e.g. ``Slide 2; Section: Results``. The single title of a file is in the section of no paragraph, as it would be in the additional context of all of them.

**Left out:** page headers and footers, comments, the speaker notes of a presentation, hidden slides and sheets, and the text that Docling finds inside pictures.

.. literalinclude:: ../../../dev/usage_examples/docs/documents/docling_conversion.py
   :language: python
   :caption: The paragraphs of a document created from a file converted with Docling

The additional context is sent to the models together with the paragraph, in the requests that list paragraphs: to an LLM, for aspects and for concepts with references (``add_references=True``), and to a decision model, which reads each paragraph on its own, for aspects (see :ref:`paragraph-additional-context`). A concept without references, which is how a decision model extracts concepts, is extracted from the text of the paragraphs only, and its items have no pages.

.. note::
   **Headings of a PDF file have no levels.** Docling detects the headings of a PDF file, but not their levels: each heading is at level 1, and the section of a paragraph is the heading before it. The headings of DOCX and HTML files keep their levels relative to each other, e.g. in HTML, Docling makes ``<h1>`` the title and ``<h2>`` a heading of level 1.

   **A scanned PDF file or image** is converted the same way: Docling reads its text with OCR, and finds its structure with its layout and table models. Its paragraphs have pages too.

.. tip::
   Docling can also export its document as Markdown, which can be passed to a document as ``raw_text``. A document created with ``from_docling()`` differs in that its references hold no Markdown syntax, each table row keeps the headers of its table, and the page of each paragraph is known.


.. _docling-pages-label:

📍 Pages and Positions
-----------------------

The ``page_numbers`` of a paragraph are the numbers of the pages that its text is on, e.g. ``[4, 5]`` for a paragraph that continues on the next page, and its sentences have them too. The ``reference_page_numbers`` of an aspect or an extracted item are the pages of its reference paragraphs and sentences, as in the example above.

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Format
     - Page numbers
   * - PDF, scanned image
     - The pages
   * - PPTX
     - The slides
   * - XLSX
     - The sheets
   * - Formats without pages, e.g. DOCX, and HTML unless Docling renders it into pages (``render_page=True`` in the options of its backend)
     - Empty

The page numbers are not sent to the models, and are kept when the document is serialized (see :doc:`../serialization`).

For more than the page, e.g. to highlight a paragraph in the file, the custom data of each paragraph holds, under the key ``"docling"``, where the paragraph comes from:

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - Key
     - Value
   * - ``self_ref``
     - The reference of the Docling item that the paragraph is made from, e.g. ``"#/texts/12"``, to look the item up in the Docling document.
   * - ``label``
     - The label of that item, e.g. ``"list_item"`` or ``"table"``.
   * - ``prov``
     - For each part of the item, the number of its page (``page_no``) and its bounding box on the page (``bbox``, with the coordinates ``l``, ``t``, ``r`` and ``b`` from the origin ``coord_origin``). The bounding box of a table row is that of its table.
   * - ``row_index``
     - For a table row, the index of the row in the table, whose header rows count, e.g. ``1`` for the first row under a header row.
   * - ``source``
     - For a paragraph of a subtitle file, a list with an entry for each cue it comes from: its ``kind`` (``"track"``), its start and end times in seconds (``start_time``, ``end_time``), and its identifier and speaker (``identifier``, ``voice``) where it has them, e.g. to cite the time of an extracted item. The times are not sent to the models.

E.g. the custom data of the first row of the table of fees, with the coordinates rounded:

.. code-block:: python

   {
       "docling": {
           "self_ref": "#/tables/0",
           "label": "table",
           "prov": [
               {
                   "page_no": 1,
                   "bbox": {
                       "l": 62.3,
                       "t": 477.0,
                       "r": 439.9,
                       "b": 377.3,
                       "coord_origin": "BOTTOMLEFT",
                   },
               }
           ],
           "row_index": 1,
       }
   }


🖼️ Pictures
------------

The pictures of the Docling document are attached to the document as images, if the Docling document has image data for them, e.g. for extraction with a vision model (see :ref:`vision-concept-extraction-label`). A picture that repeats an earlier one, e.g. a logo on each page, is attached once. The custom data of each image holds the same Docling data as that of a paragraph, and the captions of a picture are paragraphs.

Whether the Docling document has image data for a picture depends on how the file was converted:

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Source
     - Image data
   * - DOCX, PPTX or XLSX file
     - Kept by default: the pictures are embedded in the file
   * - PDF file
     - Kept with ``generate_picture_images=True`` in the PDF pipeline options of Docling (below), or cropped from the images of the pages with ``generate_page_images=True``
   * - HTML or Markdown file
     - Read with ``fetch_images=True`` in the options of the backend of Docling, which reads the images in data URIs, and with ``enable_local_fetch=True`` those in local files, or ``enable_remote_fetch=True`` those at URLs
   * - Docling document saved as JSON with its images in separate files (``image_mode=ImageRefMode.REFERENCED``)
     - Read from the files only for a Docling document from a trusted source, which docling-core leaves to the calling code: ``DoclingDocument.load_from_json(path, allow_local_images=True)``, or ``docling_document.load_local_images(base_dir=...)`` with the folder of the JSON file, which the paths are relative to
   * - Image at a remote URL in the Docling document
     - Not fetched

.. code-block:: python
   :caption: Keeping the images of the pictures of a PDF file

   from docling.datamodel.base_models import InputFormat
   from docling.datamodel.pipeline_options import PdfPipelineOptions
   from docling.document_converter import DocumentConverter, PdfFormatOption

   pipeline_options = PdfPipelineOptions(generate_picture_images=True)
   converter = DocumentConverter(
       format_options={
           InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
       }
   )

The pictures that are not attached as images are counted, with the reason, in a message logged at the ``INFO`` level.

.. note::
   A concept with the ``"image"`` or ``"multimodal"`` modality is extracted from all the images of the document in one request by default, and many providers accept only a limited number of images in a request. For a file with many pictures, pass ``max_images_to_analyze_per_call`` to the extraction method, below the limit of your provider, to send the images in parts, one request per part (see :doc:`../optimizations/optimization_long_docs`). Each request then sees only the images of its part.


⚙️ Other Settings of the Document
----------------------------------

The other arguments of ``from_docling()`` are passed to the document, e.g. ``aspects``, ``concepts``, ``sat_model_id`` or ``pre_segment_sentences`` (see :doc:`document_config`). Its paragraphs and images are made from the Docling document, and its ``raw_text`` is the text of its paragraphs.
