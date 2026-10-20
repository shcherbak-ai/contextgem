:og:description: ContextGem: Converting Files with Docling

.. meta::
   :description: How to create a Document from a PDF, DOCX, PPTX, XLSX, HTML or scanned file converted with Docling, with the structure of the file in the additional context of the paragraphs and the page of each paragraph in its custom data.

Converting Files with Docling
==============================

`Docling <https://github.com/docling-project/docling>`_ converts files such as PDF, DOCX, PPTX, XLSX, HTML and scanned images into a ``DoclingDocument``: the text, the tables and the pictures of the file, with their structure and their place on each page. :meth:`~contextgem.public.documents.Document.from_docling` creates a ContextGem document from it. The models get the structure of the file together with its text, and each extracted item can be traced to the page it was found on.


📦 Installation
----------------

The ``docling`` extra of ContextGem installs Docling, which converts the files:

.. code-block:: bash

   uv add "contextgem[docling]"

Or using pip:

.. code-block:: bash

   pip install "contextgem[docling]"

Docling runs layout and table recognition models on PDF files and images, on PyTorch, which it installs. On Linux, PyTorch comes with the CUDA libraries of NVIDIA by default; for a CPU-only PyTorch, see the `installation guide of Docling <https://docling-project.github.io/docling/installation/>`_. The extra does not install on an Intel Mac (see :doc:`../installation`).

A Docling document converted elsewhere, e.g. by a Docling Serve instance, or saved by Docling as JSON and loaded with ``DoclingDocument.load_from_json()``, needs only the document model of Docling, ``docling-core``, not Docling itself: ``uv add contextgem docling-core``.


📄 Creating a Document
-----------------------

.. literalinclude:: ../../../dev/usage_examples/docs/documents/docling_conversion.py
   :language: python
   :caption: Creating a document from a file converted with Docling

Each piece of text in the body of the Docling document becomes a paragraph, in reading order:

.. list-table::
   :header-rows: 1
   :widths: 25 45 30

   * - Docling item
     - Text of the paragraph
     - Additional context
   * - Title, heading
     - The text
     - ``Title``, ``Heading, level 2``
   * - Body text
     - The text
     - The section only
   * - List item
     - The text, after the number of a numbered item, e.g. ``2.1 Invoices are payable within 30 days.``
     - ``List item``, ``List item, level 2``
   * - Table
     - A paragraph with the headers of the columns, then a paragraph for each row, with each value after the header of its column, e.g. ``Service: Hosting; Monthly fee: 1,200 EUR``
     - ``Table header``, ``Table row``, with the caption of the table
   * - Caption, footnote, code, formula
     - The text
     - e.g. ``Table caption``, ``Picture caption``, ``Footnote``
   * - Form field, key-value pair
     - The key and the value, e.g. ``Full name: Ada Lovelace`` or ``I agree to the terms: checked``
     - ``Form field``, ``Key-value pair``

The additional context also gives the headings of the section of the paragraph, e.g. ``Section: 3. Term and Termination > 3.2 Notice``, and, in spreadsheets and presentations, its sheet or slide, e.g. ``Sheet: Q3 figures`` or ``Slide 2``. It is sent to the models together with the paragraph (see :ref:`paragraph-additional-context`): to an LLM, which gets the structure of the file without Markdown syntax in the text of the paragraphs, and to a decision model, which reads each paragraph on its own.

Page headers and footers, comments, and the text that Docling finds inside pictures are not converted.

A scanned PDF file or image is converted the same way: Docling reads its text with OCR, and finds its structure with its layout and table models. Its paragraphs have page numbers too.

.. note::
   In a PDF file, Docling detects the headings, but not their levels: each heading of a PDF file is at level 1, and the section of a paragraph is the heading before it. The heading levels of DOCX and HTML files are kept.

.. tip::
   Docling can also export its document as Markdown, which can be passed to a document as ``raw_text``. A document created with ``from_docling()`` differs in that its references hold no Markdown syntax, a table row keeps the headers of its table, and the page of each paragraph is known (see below).


📍 Citing Pages
----------------

The ``page_numbers`` of each paragraph are the numbers of the pages that its text is on, e.g. ``[4, 5]`` for a paragraph that continues on the next page, and its sentences have them too. Aspects and extracted items give the pages of their reference paragraphs and sentences in ``reference_page_numbers``:

.. literalinclude:: ../../../dev/usage_examples/docs/documents/docling_cited_pages.py
   :language: python
   :caption: Citing the pages of extracted items

For a presentation, the page numbers are the numbers of the slides, and for a spreadsheet, of the sheets. They are empty for the formats without pages, e.g. DOCX and HTML. They are not sent to the models, and are kept when the document is serialized (see :doc:`../serialization`).

For more than the page, e.g. to highlight a paragraph in the file, the custom data of each paragraph holds, under the key ``"docling"``, where the paragraph comes from:

- ``self_ref``: the reference of the Docling item that the paragraph is made from, e.g. ``"#/texts/12"``, to look the item up in the Docling document.
- ``label``: the label of that item, e.g. ``"list_item"`` or ``"table"``.
- ``prov``: for each part of the item, the number of its page (``page_no``) and its bounding box on the page (``bbox``, with the coordinates ``l``, ``t``, ``r`` and ``b`` from the origin ``coord_origin``).
- ``row_index``: for a table row, the index of the row in the table.

E.g. the custom data of the first row of the fee table in the example above, with the coordinates rounded:

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

The bounding box of a table row is that of its table.


🖼️ Pictures
------------

The pictures of the Docling document are attached to the document as images, if the Docling document has image data for them, e.g. for extraction with a vision model (see :ref:`vision-concept-extraction-label`). A picture that repeats an earlier one, e.g. a logo on each page, is attached once. The custom data of each image holds the same Docling data as that of a paragraph. The captions of a picture are paragraphs.

Docling keeps the images of the pictures of a PDF file only when asked to:

.. code-block:: python

   from docling.datamodel.base_models import InputFormat
   from docling.datamodel.pipeline_options import PdfPipelineOptions
   from docling.document_converter import DocumentConverter, PdfFormatOption

   pipeline_options = PdfPipelineOptions(generate_picture_images=True)
   converter = DocumentConverter(
       format_options={
           InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
       }
   )


⚙️ Other Settings of the Document
----------------------------------

The other arguments of ``from_docling()`` are passed to the document, e.g. ``aspects``, ``concepts``, ``sat_model_id`` or ``pre_segment_sentences`` (see :doc:`document_config`). Its paragraphs and images are made from the Docling document, and its ``raw_text`` is the text of its paragraphs.
