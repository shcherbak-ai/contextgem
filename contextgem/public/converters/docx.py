#
# ContextGem
#
# Copyright 2025 Shcherbak AI AS. All rights reserved. Developed by Sergii Shcherbak.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""
DOCX document conversion module for ContextGem.

Provides functionality for converting Microsoft Word DOCX files into ContextGem document objects,
preserving text, structure, tables, footnotes, headers, footers, and embedded images.
Implemented through the DocxConverter class.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import BinaryIO, cast

from contextgem.internal.converters.docx.base import _DocxConverterBase
from contextgem.internal.converters.docx.package import _DocxPackage
from contextgem.internal.exceptions import DocxConverterError
from contextgem.internal.loggers import logger
from contextgem.internal.typings.types import TextMode
from contextgem.public.documents import Document
from contextgem.public.paragraphs import Paragraph


class DocxConverter(_DocxConverterBase):
    """
    Converter for DOCX files into ContextGem documents.

    This class handles extraction of text, formatting, tables, images, footnotes,
    comments, and other elements from DOCX files by directly parsing Word XML.

    The converter is read-only and does not modify the source DOCX file
    in any way. It only extracts content for conversion to ContextGem document object
    or text formats.

    The resulting ContextGem document is populated with the following:

    - Raw text: The raw text of the DOCX file.

    - Paragraphs: Paragraph objects with the following metadata:

      - Raw text: The raw text of the paragraph.
      - Additional context: Metadata about the paragraph's style, list level,
        table cell position, being part of a footnote or comment, etc. This context
        provides additional information that is useful for LLM analysis and extraction.

    - Images: Image objects constructed from embedded images in the DOCX file.

    Example:
        .. literalinclude:: ../../../dev/usage_examples/readme/docx_converter.py
            :language: python
            :caption: DocxConverter usage example
    """

    def convert_to_text_format(
        self,
        docx_path_or_file: str | Path | BinaryIO,
        output_format: TextMode = "markdown",
        include_tables: bool = True,
        include_comments: bool = True,
        include_footnotes: bool = True,
        include_headers: bool = True,
        include_footers: bool = True,
        include_textboxes: bool = True,
        include_links: bool = True,
        include_inline_formatting: bool = True,
        strict_mode: bool = False,
    ) -> str:
        """
        Converts a DOCX file directly to text without creating a ContextGem Document.

        :param docx_path_or_file: Path to the DOCX file (as string or Path object) or a file-like object
        :param output_format: Output format ("markdown" or "raw") (default: "markdown")
        :param include_tables: If True, include tables in the output (default: True)
        :param include_comments: If True, include comments in the output (default: True)
        :param include_footnotes: If True, include footnotes in the output (default: True)
        :param include_headers: If True, include headers in the output (default: True)
        :param include_footers: If True, include footers in the output (default: True)
        :param include_textboxes: If True, include textbox content (default: True)
        :param include_links: If True, process and format hyperlinks (default: True)
        :param include_inline_formatting: If True, apply inline formatting (bold, italic, etc.)
            in markdown mode (default: True)
        :param strict_mode: If True, raise exceptions for any processing error
            instead of skipping problematic elements (default: False)
        :return: Text in the specified format

        .. note::
           When using markdown output format, the following conditions apply:

           * Document structure elements (headings, lists, tables) are preserved
           * Headings are converted to markdown heading syntax (# Heading 1, ## Heading 2, etc.)
           * Lists are converted to markdown list syntax, preserving numbering and hierarchy
           * Tables are formatted using markdown table syntax
           * Footnotes, comments, headers, and footers are included as specially marked sections
        """
        # Validate file extension first
        self._validate_file_extension(docx_path_or_file)

        package = None

        try:
            package = _DocxPackage(docx_path_or_file)

            if output_format.lower() == "markdown":
                # Process document elements into markdown lines
                markdown_lines = self._process_docx_elements(
                    package,
                    markdown_mode=True,
                    include_tables=include_tables,
                    include_comments=include_comments,
                    include_footnotes=include_footnotes,
                    include_headers=include_headers,
                    include_footers=include_footers,
                    include_textboxes=include_textboxes,
                    include_links=include_links,
                    strict_mode=strict_mode,
                    include_inline_formatting=include_inline_formatting,
                )

                # Join all lines and return as a single string
                # Safe cast: markdown_lines is returned as a list of strings
                return "\n".join(cast(list[str], markdown_lines))
            elif output_format.lower() == "raw":
                # Process document elements
                paragraphs = self._process_docx_elements(
                    package,
                    markdown_mode=False,
                    include_tables=include_tables,
                    include_comments=include_comments,
                    include_footnotes=include_footnotes,
                    include_headers=include_headers,
                    include_footers=include_footers,
                    include_textboxes=include_textboxes,
                    include_links=include_links,
                    strict_mode=strict_mode,
                    include_inline_formatting=include_inline_formatting,
                )

                # Combine all paragraph texts
                # Safe cast: paragraphs is returned as a list of Paragraph objects
                return "\n\n".join(
                    para.raw_text for para in cast(list[Paragraph], paragraphs)
                )
            else:
                raise DocxConverterError(f"Invalid output format: {output_format}")
        except DocxConverterError:
            # Re-raise specific converter errors
            raise
        except Exception as e:
            # Convert generic exceptions to DocxConverterError
            logger.error(f"Error converting DOCX to {output_format}: {e}")
            raise DocxConverterError(
                f"Error converting DOCX to {output_format}: {e}"
            ) from e
        finally:
            # Ensure the package is closed even if an exception occurs
            if package:
                package.close()

    def convert(
        self,
        docx_path_or_file: str | Path | BinaryIO,
        apply_markdown: bool = True,
        raw_text_to_md: bool | None = None,  # TODO: remove this parameter in v1.0.0.
        include_tables: bool = True,
        include_comments: bool = True,
        include_footnotes: bool = True,
        include_headers: bool = True,
        include_footers: bool = True,
        include_textboxes: bool = True,
        include_images: bool = True,
        include_links: bool = True,
        include_inline_formatting: bool = True,
        strict_mode: bool = False,
    ) -> Document:
        """
        Converts a DOCX file into a ContextGem Document object.

        :param docx_path_or_file: Path to the DOCX file (as string or Path object) or a file-like object
        :param apply_markdown: If True, applies markdown processing and formatting to the document content
            while preserving raw text separately (default: True)
        :param raw_text_to_md: [DEPRECATED] Use apply_markdown instead. Will be removed in v1.0.0.
            Note: This parameter previously controlled whether raw_text would contain raw or markdown text.
            The new apply_markdown parameter instead controls whether to apply markdown processing
            while keeping raw text and processed text separate.
        :param include_tables: If True, include tables in the output (default: True)
        :param include_comments: If True, include comments in the output (default: True)
        :param include_footnotes: If True, include footnotes in the output (default: True)
        :param include_headers: If True, include headers in the output (default: True)
        :param include_footers: If True, include footers in the output (default: True)
        :param include_textboxes: If True, include textbox content (default: True)
        :param include_images: If True, extract and include images (default: True)
        :param include_links: If True, process and format hyperlinks (default: True)
        :param include_inline_formatting: If True, apply inline formatting (bold, italic, etc.)
            in markdown mode (default: True)
        :param strict_mode: If True, raise exceptions for any processing error
            instead of skipping problematic elements (default: False)
        :return: A populated Document object
        """
        # Validate file extension first
        self._validate_file_extension(docx_path_or_file)

        # Handle deprecated parameter
        if raw_text_to_md is not None:
            warnings.warn(
                "The 'raw_text_to_md' parameter is deprecated and will be removed in v1.0.0. "
                "Please use 'apply_markdown' instead. Note: This change affects how text processing "
                "is handled - the Document now maintains separate raw and processed text representations.",
                DeprecationWarning,
                stacklevel=2,
            )
            apply_markdown = raw_text_to_md

        package = None
        try:
            # Get file name or descriptor for logging
            file_desc = (
                docx_path_or_file
                if isinstance(docx_path_or_file, str | Path)
                else "file object"
            )
            logger.info(f"Converting DOCX: {file_desc} (strict mode: {strict_mode})")

            # Create _DocxPackage
            package = _DocxPackage(docx_path_or_file)

            # Process document elements and get paragraphs
            logger.debug("Processing document elements")
            paragraphs = self._process_docx_elements(
                package,
                markdown_mode=False,  # Always get Paragraph objects, but we'll handle text formatting separately
                include_tables=include_tables,
                include_comments=include_comments,
                include_footnotes=include_footnotes,
                include_headers=include_headers,
                include_footers=include_footers,
                include_textboxes=include_textboxes,
                include_links=include_links,
                include_inline_formatting=include_inline_formatting,
                use_markdown_text_in_paragraphs=apply_markdown,
                populate_md_text=apply_markdown,
                strict_mode=strict_mode,
            )
            logger.debug(f"Extracted {len(paragraphs)} paragraphs")

            # Generate raw text from the paragraph objects we already have
            # Safe cast: paragraphs is returned as a list of Paragraph objects
            raw_text = "\n\n".join(
                para.raw_text for para in cast(list[Paragraph], paragraphs)
            )
            doc_kwargs = {
                "raw_text": raw_text,
                "paragraphs": paragraphs,
            }

            # Create the document object
            context_doc = Document(**doc_kwargs)

            if apply_markdown:
                # Generate markdown text from the same paragraphs we extracted
                markdown_lines = self._process_docx_elements(
                    package,
                    markdown_mode=True,
                    include_tables=include_tables,
                    include_comments=include_comments,
                    include_footnotes=include_footnotes,
                    include_headers=include_headers,
                    include_footers=include_footers,
                    include_textboxes=include_textboxes,
                    include_links=include_links,
                    include_inline_formatting=include_inline_formatting,
                    strict_mode=strict_mode,
                )
                # Safe cast: markdown_lines is returned as a list of strings
                md_text = "\n".join(cast(list[str], markdown_lines))

                # When markdown mode is requested, populate _md_text
                context_doc._md_text = md_text

            # Process images from DOCX if requested
            if include_images:
                logger.debug("Processing images")
                images = self._extract_images(package, strict_mode=strict_mode)
                # Attach images to the document
                context_doc.images = images
                logger.debug(f"Added {len(images)} images to document")

            logger.info(
                f"DOCX conversion completed successfully: {len(paragraphs)} paragraphs, "
                f"{len(context_doc.images) if include_images else 0} images"
            )
            return context_doc
        except DocxConverterError:
            # Re-raise specific converter errors
            raise
        except Exception as e:
            # Catch any other exceptions and convert to DocxConverterError
            logger.error(f"Error converting DOCX file: {e}")
            raise DocxConverterError(f"Error converting DOCX file: {e}") from e
        finally:
            # Ensure the package is closed even if an exception occurs
            if package:
                package.close()
