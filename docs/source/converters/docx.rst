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

:og:description: ContextGem: DOCX Converter

DOCX Converter
===============

ContextGem provides built-in converter to easily transform DOCX files into LLM-ready ContextGem document objects.

* 📑 **Comprehensive extraction of document elements**: paragraphs, headings, lists, tables, comments, footnotes, textboxes, headers/footers, links, embedded images, and inline formatting
* 🧩 **Document structure preservation** with rich metadata for improved LLM analysis
* 🛠️ **Built-in converter** that directly processes Word XML


🚀 Usage
----------

.. literalinclude:: ../../../dev/usage_examples/readme/docx_converter.py
   :language: python


🔄 Conversion Process
----------------------

The :class:`~contextgem.public.converters.DocxConverter` performs the following operations when converting a DOCX file to a ContextGem Document with :meth:`~contextgem.public.converters.DocxConverter.convert` method:

.. list-table::
   :header-rows: 1
   :widths: 25 50 25

   * - Elements
     - Extraction Details
     - Control Parameter (Default)
   * - **Text**
     - Extracts the full document text as raw text, and optionally applies markdown processing and formatting while preserving raw text separately
     - ``apply_markdown=True``
   * - **Paragraphs**
     - Extracts :class:`~contextgem.public.paragraphs.Paragraph` objects with rich metadata serving as additional context for LLM (e.g., *"Style: Normal, Table: 3, Row: 1, Column: 3, Table Cell"*)
     - *Always included*
   * - **Headings**
     - Preserves heading levels and formats as markdown headings when in markdown mode
     - *Always included*
   * - **Lists**
     - Maintains list hierarchy, numbering, and formatting with proper indentation and list type information
     - *Always included*
   * - **Tables**
     - Preserves table structure and formats tables in markdown mode
     - ``include_tables=True``
   * - **Headers & Footers**
     - Captures document headers and footers with appropriate metadata
     - ``include_headers=True`` / ``include_footers=True``
   * - **Footnotes**
     - Extracts footnotes with references and preserves connection to original text
     - ``include_footnotes=True``
   * - **Comments**
     - Preserves document comments with author information and timestamps
     - ``include_comments=True``
   * - **Links**
     - Processes and formats hyperlinks, preserving both link text and target URLs
     - ``include_links=True``
   * - **Text Boxes**
     - Extracts text from various text box formats
     - ``include_textboxes=True``
   * - **Inline Formatting**
     - Applies inline formatting such as bold, italic, underline, etc. when in markdown mode
     - ``include_inline_formatting=True``
   * - **Images**
     - Extracts embedded images and converts them to :class:`~contextgem.public.images.Image` objects for further processing with vision models
     - ``include_images=True``

ℹ️ Current Limitations
------------------------

DocxConverter has the following limitations:

* Drawings such as charts are skipped as it is challenging to represent them in text format.
* Inline markdown formatting (bold, italic, etc.) and hyperlink formatting are not supported in specially marked sections (headers, footers, footnotes, comments).
* Extraction of generated table of contents (ToC) is not supported. (A ToC is an automatically generated list of document headings with page numbers that Word creates based on heading styles.)
* When converting very long DOCX files with complex formatting, performance may be very slow. (Our goal is to improve this in the future.)
