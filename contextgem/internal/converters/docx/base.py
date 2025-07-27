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
Base class for the DocxConverter.

This module contains the _DocxConverterBase base class which implements all
the low-level functionality for parsing DOCX files, extracting text, formatting,
tables, images, footnotes, comments, and other elements.
"""

import base64
import re
from collections.abc import Callable
from pathlib import Path
from typing import BinaryIO, cast

from lxml import etree

from contextgem.internal.converters.docx.package import _DocxPackage
from contextgem.internal.converters.docx.utils import (
    NUMBERED_LIST_FORMATS,
    PAGE_FIELD_KEYWORDS,
    _docx_get_namespaced_attr,
    _docx_xpath,
    _extract_comment_id_from_context,
    _extract_footnote_id_from_context,
    _join_text_parts,
)
from contextgem.internal.exceptions import (
    DocxContentError,
    DocxConverterError,
    DocxXmlError,
)
from contextgem.internal.loggers import logger
from contextgem.internal.utils import _is_text_content_empty
from contextgem.public.images import Image
from contextgem.public.paragraphs import Paragraph


class _DocxConverterBase:
    """
    Base class for the DOCX converter.
    """

    def _validate_file_extension(
        self, docx_path_or_file: str | Path | BinaryIO
    ) -> None:
        """
        Validates that the provided file has a valid DOCX extension.

        :param docx_path_or_file: Path to the file or file-like object
        :raises DocxConverterError: If the file doesn't have a valid DOCX extension
        """
        # Skip validation for file-like objects (BinaryIO)
        if not isinstance(docx_path_or_file, str | Path):
            return

        file_path = Path(docx_path_or_file)
        file_extension = file_path.suffix.lower()

        # Only accept .docx files - the standard XML-based Word document format
        if not file_extension:
            raise DocxConverterError(
                f"File '{file_path.name}' has no extension. Expected a .docx file."
            )
        elif file_extension != ".docx":
            raise DocxConverterError(
                f"File '{file_path.name}' has extension '{file_extension}'. "
                f"Only .docx files are supported by this converter."
            )

    def _process_paragraph(
        self,
        para_element: etree._Element,
        package: _DocxPackage,
        markdown_mode: bool = False,
        include_textboxes: bool = True,
        include_links: bool = True,
        include_inline_formatting: bool = True,
        apply_text_formatting: bool | None = None,
        populate_md_text: bool = False,
        list_counters: dict | None = None,
        strict_mode: bool = False,
    ) -> str | Paragraph | None:
        """
        Processes a paragraph element and returns appropriate content based on mode.

        :param para_element: Paragraph XML element
        :param package: _DocxPackage object
        :param markdown_mode: If True, return markdown formatted text,
            otherwise return a Paragraph object (default: False)
        :param include_textboxes: If True, include textbox content (default: True)
        :param include_links: If True, process and format hyperlinks (default: True)
        :param include_inline_formatting: If True, apply inline formatting (bold, italic, etc.)
            in markdown mode (default: True)
        :param apply_text_formatting: If provided, use this flag for text formatting
            instead of markdown_mode (default: None)
        :param populate_md_text: If True, populate the _md_text field in Paragraph objects
            with markdown representation (default: False)
        :param list_counters: Dictionary to track list numbering counters (default: None)
        :param strict_mode: If True, raise exceptions for any processing error
            instead of skipping problematic elements (default: False)
        :return: * **None** -- When paragraph should be skipped:

                    - Empty paragraph (no text content after processing)
                    - Textbox paragraph when ``include_textboxes=False``
                    - Paragraph within tracked changes (move source or deletion)
                    - Processing error when ``strict_mode=False`` (error logged)

                * **str** -- When ``markdown_mode=True`` and paragraph has content:

                    - Markdown-formatted text with appropriate styling

                * **Paragraph** -- When ``markdown_mode=False`` and paragraph has content:

                    - Structured paragraph object with ``raw_text`` and ``additional_context``
                    - Optionally has ``_md_text`` attribute populated when ``populate_md_text=True``
        :rtype: str | Paragraph | None
        """
        try:
            # Check if this is a text box paragraph and we should skip it
            if not include_textboxes and self._is_text_box_paragraph(para_element):
                return None

            # Check if this entire paragraph is within a move source or deletion -
            # skip these to avoid duplicates
            is_para_in_move_from = _docx_xpath(para_element, "ancestor::w:moveFrom")
            is_para_in_deletion = _docx_xpath(para_element, "ancestor::w:del")
            if is_para_in_move_from or is_para_in_deletion:
                return None

            # Extract raw text content (always without markdown formatting for consistency)
            raw_text = self._extract_paragraph_text(
                para_element=para_element,
                package=package,
                markdown_mode=False,  # Always extract raw text first
                strict_mode=strict_mode,
                include_textboxes=include_textboxes,
                include_links=include_links,
                include_inline_formatting=False,  # Never apply inline formatting for raw text
            ).strip()
            if _is_text_content_empty(raw_text):
                return None

            # Get style information
            style_id = self._get_paragraph_style(para_element)
            style_name = self._get_style_name(style_id, package)
            style_info = f"Style: {style_name}"

            # Get list information
            is_list, list_level, _, list_type, is_numbered = self._get_list_info(
                para_element, package
            )

            # Extract markdown text if needed (either for return or for _md_text field)
            markdown_text = None
            if markdown_mode or populate_md_text:
                # Use apply_text_formatting if provided, otherwise use True for markdown
                actual_formatting_mode = (
                    apply_text_formatting if apply_text_formatting is not None else True
                )
                if actual_formatting_mode:
                    # Get base markdown text
                    base_markdown_text = self._extract_paragraph_text(
                        para_element=para_element,
                        package=package,
                        markdown_mode=True,  # Extract with markdown formatting
                        strict_mode=strict_mode,
                        include_textboxes=include_textboxes,
                        include_links=include_links,
                        include_inline_formatting=include_inline_formatting,  # Use the parameter
                    ).strip()

                    # Apply document-level markdown formatting
                    markdown_text = self._apply_markdown_formatting(
                        text=base_markdown_text,
                        style_name=style_name,
                        is_list=is_list,
                        list_level=list_level,
                        is_numbered=is_numbered,
                        para_element=para_element,
                        list_counters=list_counters,
                    )

            # Use the appropriate text for processing based on mode
            text = markdown_text if markdown_mode and markdown_text else raw_text

            # Get footnote reference information
            footnote_info = ""
            footnote_ids = self._extract_footnote_references(para_element)
            if footnote_ids:
                footnote_info = f", Footnote References: {','.join(footnote_ids)}"

            # Get comment reference information
            comment_info = ""
            comment_ids = self._extract_comment_references(para_element)
            if comment_ids:
                comment_info = f", Comment References: {','.join(comment_ids)}"

            # Check if this is a text box paragraph
            text_box_info = ""
            if self._is_text_box_paragraph(para_element):
                text_box_info = ", Text Box"

            # Get hyperlink information
            hyperlink_info = ""
            if include_links:
                hyperlinks = self._extract_hyperlink_references(para_element, package)
                if hyperlinks:
                    # For paragraph mode, add hyperlink information to metadata without changing text
                    hyperlink_urls = [link["url"] for link in hyperlinks if link["url"]]
                    if hyperlink_urls:
                        hyperlink_info = (
                            f", Link, Link URL: {', '.join(hyperlink_urls)}"
                        )

            if markdown_mode:
                # Return the formatted markdown text
                return text

            else:
                # Return a Paragraph instance with metadata
                metadata = style_info

                # Add list information with more details
                if is_list:
                    list_type_info = "Numbered" if is_numbered else "Bullet"
                    metadata += (
                        f", List Type: {list_type_info}, List Level: {list_level}"
                    )
                    if list_type:
                        # Capitalize the format name for consistency
                        formatted_list_type = list_type.capitalize()
                        metadata += f", List Format: {formatted_list_type}"

                    # Extract List ID from original _get_list_info results
                    p_pr_elements = _docx_xpath(para_element, "w:pPr")
                    if p_pr_elements:
                        p_pr = p_pr_elements[0]
                        num_pr_elements = _docx_xpath(p_pr, "w:numPr")
                        if num_pr_elements:
                            num_pr = num_pr_elements[0]
                            num_id_elements = _docx_xpath(num_pr, "w:numId")
                            if num_id_elements:
                                list_id = _docx_get_namespaced_attr(
                                    num_id_elements[0], "val"
                                )
                                metadata += f", List ID: {list_id}"

                metadata += (
                    footnote_info + comment_info + text_box_info + hyperlink_info
                )

                # Create paragraph with _md_text if requested
                paragraph = Paragraph(
                    raw_text=raw_text,
                    additional_context=metadata,
                )
                if populate_md_text:
                    # Use the properly formatted markdown text
                    paragraph._md_text = markdown_text if markdown_text else raw_text

                return paragraph

        except DocxXmlError:
            # Re-raise specific XML errors
            raise
        except Exception as e:
            if strict_mode:
                raise DocxContentError(f"Error processing paragraph: {e}") from e
            else:
                logger.warning(f"Error processing paragraph: {e}")
                return None

    def _extract_paragraph_text(
        self,
        para_element: etree._Element,
        package: _DocxPackage,
        markdown_mode: bool = False,
        include_textboxes: bool = True,
        include_links: bool = True,
        include_inline_formatting: bool = True,
        strict_mode: bool = False,
    ) -> str:
        """
        Extracts the text content from a paragraph element.

        Page numbers are automatically excluded from extraction.

        :param para_element: Paragraph XML element
        :param package: _DocxPackage object containing hyperlink relationships
        :param markdown_mode: If True, format hyperlinks as markdown links
        :param include_textboxes: If True, include textbox content (default: True)
        :param include_links: If True, process and format hyperlinks (default: True)
        :param include_inline_formatting: If True, apply inline formatting (bold, italic, etc.)
            in markdown mode (default: True)
        :param strict_mode: If True, raise exceptions for any processing error
            instead of skipping problematic elements (default: False)
        :return: Text content of the paragraph
        """
        # Use a dictionary to track text content by location
        text_by_location = {}
        ordered_runs = []

        try:
            # First, collect hyperlink information to handle them properly during run processing
            # Skip this entirely if include_links is False to avoid processing overhead
            hyperlink_info_by_run = {}  # Maps run elements to their hyperlink info
            if include_links:
                hyperlink_elements = _docx_xpath(para_element, ".//w:hyperlink")
                for hyperlink_elem in hyperlink_elements:
                    # Get the relationship ID
                    rel_id = _docx_get_namespaced_attr(hyperlink_elem, "id", "r")

                    # Get the hyperlink URL from relationships
                    url = package.hyperlinks.get(rel_id, "")

                    # Find all runs within this hyperlink
                    hyperlink_runs = _docx_xpath(hyperlink_elem, ".//w:r")
                    for run in hyperlink_runs:
                        hyperlink_info_by_run[id(run)] = {"url": url, "rel_id": rel_id}

            # Process regular paragraph text (including runs inside hyperlinks, content controls, etc.)
            run_idx = 0
            runs = _docx_xpath(para_element, ".//w:r")
            for run in runs:
                run_element_id = id(run)
                is_hyperlink_run = run_element_id in hyperlink_info_by_run

                # Skip page number fields
                if self._is_page_number_field(run, para_element):
                    continue

                # Extract formatting for this run if inline formatting is enabled
                run_formatting = {}
                if include_inline_formatting and markdown_mode:
                    run_formatting = self._extract_run_formatting(run)

                # Process text elements in this run (don't skip runs with drawings)
                text_elements = _docx_xpath(run, ".//w:t")
                for text_elem in text_elements:
                    # Get all text including from child elements
                    text_content = (
                        # Attribute is defined in _Element
                        text_elem.text_content()  # type: ignore[attr-defined]
                        if hasattr(text_elem, "text_content")
                        else (text_elem.text or "")
                    )
                    # For regular paragraph text, don't use content deduplication to allow
                    # intentional duplicates but check if this text element is part of a textbox
                    # to avoid cross-section duplication
                    if text_content:
                        # Check if this text element is inside a textbox/drawing context
                        is_in_textbox = (
                            _docx_xpath(text_elem, "ancestor::v:textbox")
                            or _docx_xpath(text_elem, "ancestor::w:txbxContent")
                            or _docx_xpath(text_elem, "ancestor::a:p")
                            or _docx_xpath(text_elem, "ancestor::w:drawing")
                        )

                        # Check if this text element is in a move source (w:moveFrom) - skip these
                        # to avoid duplicates
                        is_in_move_from = _docx_xpath(text_elem, "ancestor::w:moveFrom")

                        # Check if this text element is in a deletion (w:del) - skip these
                        is_in_deletion = _docx_xpath(text_elem, "ancestor::w:del")

                        # Only process if NOT in textbox (textbox content will be handled in
                        # specialized sections) and NOT in a move source or deletion (to avoid
                        # duplicated/deleted content)
                        if (
                            not is_in_textbox
                            and not is_in_move_from
                            and not is_in_deletion
                        ):
                            # Start with the base text content
                            formatted_text = text_content

                            # Apply inline formatting if enabled and in markdown mode
                            if (
                                include_inline_formatting
                                and markdown_mode
                                and run_formatting
                            ):
                                formatted_text = self._apply_inline_formatting(
                                    formatted_text, run_formatting
                                )

                            # Apply hyperlink formatting only if include_links=True and
                            # markdown_mode=True
                            if is_hyperlink_run and markdown_mode and include_links:
                                # For hyperlink runs in markdown mode, format as markdown link
                                url = hyperlink_info_by_run[run_element_id]["url"]
                                if url:
                                    # If text already has formatting, preserve it inside the link
                                    text_by_location[run_idx] = (
                                        f"[{formatted_text}]({url})"
                                    )
                                else:
                                    text_by_location[run_idx] = formatted_text
                            else:
                                # Regular text or hyperlink when include_links=False
                                text_by_location[run_idx] = formatted_text
                            ordered_runs.append(run_idx)
                            run_idx += 1

                # Process line breaks in this run
                br_elements = _docx_xpath(run, ".//w:br")
                for _br in br_elements:
                    text_by_location[run_idx] = "\n"
                    ordered_runs.append(run_idx)
                    run_idx += 1

                # Add footnote reference marker if this run contains a footnote reference
                footnote_refs = _docx_xpath(run, ".//w:footnoteReference")
                for footnote_ref in footnote_refs:
                    footnote_id = _docx_get_namespaced_attr(footnote_ref, "id")
                    if footnote_id:
                        if markdown_mode:
                            # In markdown mode, use descriptive marker
                            text_by_location[run_idx] = f"[Footnote {footnote_id}]"
                        else:
                            # In raw text mode, use just the footnote number
                            text_by_location[run_idx] = footnote_id
                        ordered_runs.append(run_idx)
                        run_idx += 1

                # Add comment reference marker if this run contains a comment reference
                # Note: Comments are only included in markdown mode, not in raw text
                if markdown_mode:
                    comment_refs = _docx_xpath(run, ".//w:commentReference")
                    for comment_ref in comment_refs:
                        comment_id = _docx_get_namespaced_attr(comment_ref, "id")
                        if comment_id:
                            text_by_location[run_idx] = f"[Comment {comment_id}]"
                            ordered_runs.append(run_idx)
                            run_idx += 1

            # Calculate dynamic starting indices to avoid collisions
            # Pre-count elements to determine actual space needed for each content type
            current_idx = run_idx + 100  # Small buffer after regular text runs

            # Process drawing objects (incl. text boxes) - these need special handling
            # Skip this section if include_textboxes is False
            if include_textboxes:
                # We keep track of which drawings we've seen to avoid duplicates
                # but still permit intentional repetition of text boxes.

                # Group 1: Standard VML textboxes
                vml_idx = current_idx
                textboxes = _docx_xpath(para_element, ".//v:textbox")
                for textbox in textboxes:
                    textbox_results = self._process_textbox_runs(
                        container_element=textbox,
                        hyperlink_info_by_run=hyperlink_info_by_run,
                        para_element=para_element,
                        include_inline_formatting=include_inline_formatting,
                        markdown_mode=markdown_mode,
                        include_links=include_links,
                    )
                    for formatted_text in textbox_results:
                        text_by_location[vml_idx] = formatted_text
                        ordered_runs.append(vml_idx)
                        vml_idx += 1

                # Update current_idx to next available position
                current_idx = vml_idx + 100  # Small buffer between content types

                # Group 2: DrawingML textboxes (Office 2007+ format)
                dml_idx = current_idx
                txbx_content_elems = _docx_xpath(para_element, ".//w:txbxContent")
                for txbx_content in txbx_content_elems:
                    # Process each paragraph in the text box content
                    paragraphs = _docx_xpath(txbx_content, ".//w:p")
                    for p in paragraphs:
                        textbox_results = self._process_textbox_runs(
                            container_element=p,
                            hyperlink_info_by_run=hyperlink_info_by_run,
                            para_element=para_element,
                            include_inline_formatting=include_inline_formatting,
                            markdown_mode=markdown_mode,
                            include_links=include_links,
                        )
                        for formatted_text in textbox_results:
                            text_by_location[dml_idx] = formatted_text
                            ordered_runs.append(dml_idx)
                            dml_idx += 1

                # Update current_idx to next available position
                current_idx = dml_idx + 100  # Small buffer between content types

                # Group 3: DrawingML text directly in shapes
                shape_idx = current_idx
                # First find all runs that contain a:t elements to check for hyperlinks
                shape_runs = _docx_xpath(para_element, ".//w:r[.//a:t]")
                for run in shape_runs:
                    # Create a temporary container to use our helper method
                    textbox_results = self._process_textbox_runs(
                        container_element=run,
                        hyperlink_info_by_run=hyperlink_info_by_run,
                        para_element=para_element,
                        include_inline_formatting=include_inline_formatting,
                        markdown_mode=markdown_mode,
                        include_links=include_links,
                        text_xpath=".//a:t",  # Use a:t instead of w:t for shapes
                    )
                    for formatted_text in textbox_results:
                        text_by_location[shape_idx] = formatted_text
                        ordered_runs.append(shape_idx)
                        shape_idx += 1

                # Update current_idx to next available position
                current_idx = shape_idx + 100  # Small buffer between content types

                # Group 4: Drawing elements that might contain text not captured by other groups
                drawing_idx = current_idx
                drawings = _docx_xpath(para_element, ".//w:drawing")
                for drawing in drawings:
                    textbox_results = self._process_textbox_runs(
                        container_element=drawing,
                        hyperlink_info_by_run=hyperlink_info_by_run,
                        para_element=para_element,
                        include_inline_formatting=include_inline_formatting,
                        markdown_mode=markdown_mode,
                        include_links=include_links,
                    )
                    for formatted_text in textbox_results:
                        text_by_location[drawing_idx] = formatted_text
                        ordered_runs.append(drawing_idx)
                        drawing_idx += 1

                # Update current_idx to next available position
                current_idx = drawing_idx + 100  # Small buffer between content types

            # Group 5: Handle Markup Compatibility (mc) alternate content
            # This is outside include_textboxes condition because MC elements can contain mixed content:
            # regular text, hyperlinks, drawings, etc. The include_textboxes parameter is passed through
            # to allow granular control within the MC processor (process text always, textboxes conditionally)
            mc_idx = current_idx
            mc_elements = _docx_xpath(para_element, ".//mc:AlternateContent")
            for mc_elem in mc_elements:
                # First try the Choice content (preferred for newer versions of Word)
                choice_elems = _docx_xpath(mc_elem, ".//mc:Choice")
                fallback_elems = _docx_xpath(mc_elem, ".//mc:Fallback")

                # We only want to process either Choice OR Fallback, not both, as they represent
                # alternate representations of the same content
                if choice_elems:
                    for choice in choice_elems:
                        mc_results = self._process_markup_compatibility_element(
                            mc_element=choice,
                            hyperlink_info_by_run=hyperlink_info_by_run,
                            para_element=para_element,
                            markdown_mode=markdown_mode,
                            include_inline_formatting=include_inline_formatting,
                            include_links=include_links,
                            include_textboxes=include_textboxes,
                        )
                        for formatted_text in mc_results:
                            text_by_location[mc_idx] = formatted_text
                            ordered_runs.append(mc_idx)
                            mc_idx += 1
                # Only use Fallback if we didn't find any usable Choice elements
                elif fallback_elems:
                    for fallback in fallback_elems:
                        mc_results = self._process_markup_compatibility_element(
                            mc_element=fallback,
                            hyperlink_info_by_run=hyperlink_info_by_run,
                            para_element=para_element,
                            markdown_mode=markdown_mode,
                            include_inline_formatting=include_inline_formatting,
                            include_links=include_links,
                            include_textboxes=include_textboxes,
                        )
                        for formatted_text in mc_results:
                            text_by_location[mc_idx] = formatted_text
                            ordered_runs.append(mc_idx)
                            mc_idx += 1

            # Sort the runs to maintain document order
            ordered_runs.sort()

            # Get raw text parts
            text_parts = [text_by_location[idx] for idx in ordered_runs]

            # Post-processing step: fix text box duplication where identical text appears
            # consecutively. This handles cases where Word stores the same text multiple times
            # in the XML. Only apply deduplication to textbox content (idx > run_idx),
            # not regular paragraph text (idx <= run_idx).
            textbox_start_idx = run_idx + 100  # First textbox content starts here
            processed_text = []
            seen_textbox_content = set()  # Track textbox content to avoid duplicates
            i = 0
            while i < len(text_parts):
                current_segment = text_parts[i]
                current_idx = ordered_runs[i]

                # Only deduplicate textbox content (technical duplicates), not regular text
                # (intentional duplicates)
                if (
                    current_idx >= textbox_start_idx
                ):  # Textbox content starts after regular text runs
                    # For textbox content, use content-based deduplication to handle
                    # variations in formatting of the same underlying text

                    # Strip markdown formatting to get base content for comparison
                    normalized_content = re.sub(
                        r"\*+([^*]+)\*+", r"\1", current_segment
                    )  # Remove *italics* and **bold**
                    normalized_content = re.sub(
                        r"~~([^~]+)~~", r"\1", normalized_content
                    )  # Remove ~~strikethrough~~
                    normalized_content = re.sub(
                        r"\[([^\]]+)\]\([^)]+\)", r"\1", normalized_content
                    )  # Remove [text](url)
                    normalized_content = re.sub(
                        r"</?[^>]+>", "", normalized_content
                    )  # Remove HTML tags
                    normalized_content = normalized_content.strip()

                    if (
                        normalized_content
                        and normalized_content not in seen_textbox_content
                    ):
                        processed_text.append(current_segment)
                        seen_textbox_content.add(normalized_content)
                    # Skip if we've already seen this textbox content
                    i += 1
                else:
                    # For regular paragraph text, keep all content including intentional duplicates
                    processed_text.append(current_segment)
                    i += 1

            return _join_text_parts(processed_text)
        except Exception as e:
            if strict_mode:
                raise DocxContentError(f"Error extracting paragraph text: {e}") from e
            else:
                logger.warning(f"Error extracting paragraph text: {e}")
                return ""

    def _process_docx_elements(
        self,
        package: _DocxPackage,
        markdown_mode: bool = False,
        include_tables: bool = True,
        include_comments: bool = True,
        include_footnotes: bool = True,
        include_headers: bool = True,
        include_footers: bool = True,
        include_textboxes: bool = True,
        include_links: bool = True,
        include_inline_formatting: bool = True,
        use_markdown_text_in_paragraphs: bool = False,
        populate_md_text: bool = False,
        strict_mode: bool = False,
    ) -> list[str | Paragraph]:
        """
        Processes all elements in the DOCX document and returns appropriate objects.

        :param package: _DocxPackage object
        :param markdown_mode: If True, return markdown formatted lines,
            otherwise return objects (default: False)
        :param include_tables: If True, include tables in the output (default: True)
        :param include_comments: If True, include comments in the output (default: True)
        :param include_footnotes: If True, include footnotes in the output (default: True)
        :param include_headers: If True, include headers in the output (default: True)
        :param include_footers: If True, include footers in the output (default: True)
        :param include_textboxes: If True, include textbox content (default: True)
        :param include_links: If True, process and format hyperlinks (default: True)
        :param include_inline_formatting: If True, apply inline formatting (bold, italic, etc.)
            in markdown mode (default: True)
        :param use_markdown_text_in_paragraphs: If True, format comments and hyperlinks
            in markdown style even when creating Paragraph objects (default: False)
        :param populate_md_text: If True, populate the _md_text field in Paragraph objects
            with markdown representation (default: False)
        :param strict_mode: If True, raise exceptions for any processing
            error instead of skipping problematic elements (default: False)
        :return: List of markdown lines or Paragraph objects
        """
        result = []

        if package.main_document is None:
            raise DocxContentError("Main document content is missing")

        try:
            # Get the body element
            body_elements = _docx_xpath(package.main_document, ".//w:body")
            if not body_elements:
                raise DocxContentError("Document body element is missing")
            body = body_elements[0]

            # Process headers
            if include_headers:
                try:
                    header_paragraphs = self._process_headers(
                        package,
                        strict_mode=strict_mode,
                        include_textboxes=include_textboxes,
                        populate_md_text=populate_md_text,
                    )
                    if markdown_mode and header_paragraphs:
                        self._handle_section_in_markdown_mode(
                            header_paragraphs, "Header", result
                        )
                    else:
                        # For object mode, add headers at the beginning
                        result.extend(header_paragraphs)
                except Exception as e:
                    # In strict mode, re-raise as DocxContentError
                    if strict_mode:
                        raise DocxContentError(f"Error processing headers: {e}") from e
                    # Otherwise, log error and continue without headers
                    logger.warning(f"Error processing headers: {e}")

            # Track tables for indexing
            table_count = 0

            # Track numbered lists for proper sequencing
            list_counters = {}  # {(list_id, level): counter}
            last_list_id = None
            last_list_level = -1

            # Process each element in order
            for element in body:
                # Attribute is defined in _Element
                tag = element.tag.split("}")[  # type: ignore[attr-defined]
                    -1
                ]  # Remove namespace prefix

                if tag == "p":
                    # Process paragraph
                    try:
                        # Update list counters for both markdown and paragraph modes
                        # Extract list information
                        p_pr_elements = _docx_xpath(element, "w:pPr")
                        if p_pr_elements:
                            p_pr = p_pr_elements[0]
                            num_pr_elements = _docx_xpath(p_pr, "w:numPr")
                            if num_pr_elements:
                                num_pr = num_pr_elements[0]
                                # This is a list item
                                _, list_level, __, ___, is_numbered = (
                                    self._get_list_info(element, package)
                                )

                                # Get num_id for this list item
                                num_id_elements = _docx_xpath(num_pr, "w:numId")
                                if num_id_elements:
                                    num_id = _docx_get_namespaced_attr(
                                        num_id_elements[0], "val"
                                    )

                                    # If it's a numbered list, we need to track counter
                                    if is_numbered:
                                        list_key = (num_id, list_level)

                                        # Reset counter if this is a new list or a higher
                                        # level in the same list
                                        if (
                                            last_list_id != num_id
                                            or list_level < last_list_level
                                        ):
                                            # Reset counters for all levels below
                                            # the current level
                                            for key in list(list_counters.keys()):
                                                if (
                                                    key[0] == num_id
                                                    and key[1] > list_level
                                                ):
                                                    list_counters.pop(key)

                                        # Initialize counter if needed
                                        if list_key not in list_counters:
                                            list_counters[list_key] = 1
                                        else:
                                            list_counters[list_key] += 1

                                        # Remember this list for next iteration
                                        last_list_id = num_id
                                        last_list_level = list_level

                        # Process paragraph with list counter information
                        # Use markdown_mode OR use_markdown_text_in_paragraphs for text formatting
                        apply_text_formatting = (
                            markdown_mode or use_markdown_text_in_paragraphs
                        )
                        processed_para = self._process_paragraph(
                            para_element=element,
                            package=package,
                            markdown_mode=markdown_mode,
                            strict_mode=strict_mode,
                            include_textboxes=include_textboxes,
                            apply_text_formatting=apply_text_formatting,
                            populate_md_text=populate_md_text,
                            include_links=include_links,
                            include_inline_formatting=include_inline_formatting,
                            list_counters=list_counters,
                        )
                        if processed_para is not None:
                            result.append(processed_para)
                            # Add blank line after paragraphs in markdown mode
                            if markdown_mode:
                                result.append("")
                    except Exception as e:
                        if strict_mode:
                            # In strict mode, re-raise as DocxContentError
                            raise DocxContentError(
                                f"Error processing paragraph: {e}"
                            ) from e
                        # Log error and continue with next paragraph
                        logger.warning(f"Error processing paragraph: {e}")

                elif tag == "tbl" and include_tables:
                    # Process table
                    try:
                        table_items = self._process_table(
                            table_element=element,
                            package=package,
                            markdown_mode=markdown_mode,
                            table_idx=table_count,
                            strict_mode=strict_mode,
                            include_textboxes=include_textboxes,
                            populate_md_text=populate_md_text,
                            include_links=include_links,
                            include_inline_formatting=include_inline_formatting,
                        )
                        result.extend(table_items)
                        table_count += 1
                    except Exception as e:
                        if strict_mode:
                            # In strict mode, re-raise as DocxContentError
                            raise DocxContentError(
                                f"Error processing table: {e}"
                            ) from e
                        # Log error and continue with next element
                        logger.warning(f"Error processing table: {e}")

            # Process footnotes and add them as regular paragraphs
            if include_footnotes and package.footnotes is not None:
                try:
                    footnote_paragraphs = self._process_footnotes(
                        package,
                        strict_mode=strict_mode,
                        include_textboxes=include_textboxes,
                        populate_md_text=populate_md_text,
                    )

                    if markdown_mode and footnote_paragraphs:
                        self._handle_section_in_markdown_mode(
                            footnote_paragraphs,
                            "Footnote",
                            result,
                            _extract_footnote_id_from_context,
                        )
                    else:
                        # For object mode, just add footnotes as paragraphs
                        result.extend(footnote_paragraphs)
                except Exception as e:
                    if strict_mode:
                        # In strict mode, re-raise as DocxContentError
                        raise DocxContentError(
                            f"Error processing footnotes: {e}"
                        ) from e
                    # Log error and continue without footnotes
                    logger.warning(f"Error processing footnotes: {e}")

            # Process comments and add them as regular paragraphs
            if include_comments and package.comments is not None:
                try:
                    comment_paragraphs = self._process_comments(
                        package,
                        strict_mode=strict_mode,
                        include_textboxes=include_textboxes,
                        populate_md_text=populate_md_text,
                    )

                    if markdown_mode and comment_paragraphs:
                        self._handle_section_in_markdown_mode(
                            comment_paragraphs,
                            "Comment",
                            result,
                            _extract_comment_id_from_context,
                        )
                    else:
                        # For object mode, just add comments as paragraphs
                        result.extend(comment_paragraphs)
                except Exception as e:
                    if strict_mode:
                        # In strict mode, re-raise as DocxContentError
                        raise DocxContentError(f"Error processing comments: {e}") from e
                    # Log error and continue without comments
                    logger.warning(f"Error processing comments: {e}")

            # Process footers
            if include_footers and package.footers:
                try:
                    footer_paragraphs = self._process_footers(
                        package,
                        strict_mode=strict_mode,
                        include_textboxes=include_textboxes,
                        populate_md_text=populate_md_text,
                    )
                    if markdown_mode and footer_paragraphs:
                        self._handle_section_in_markdown_mode(
                            footer_paragraphs, "Footer", result
                        )
                    else:
                        # For object mode, add footers at the end
                        result.extend(footer_paragraphs)
                except Exception as e:
                    if strict_mode:
                        # In strict mode, re-raise as DocxContentError
                        raise DocxContentError(f"Error processing footers: {e}") from e
                    # Log error and continue without footers
                    logger.warning(f"Error processing footers: {e}")

            return result
        except DocxConverterError:
            # Re-raise specific converter errors
            raise
        except Exception as e:
            # Handle general errors in document processing
            raise DocxXmlError(f"Error processing document elements: {e}") from e

    def _process_text_run(
        self,
        run: etree._Element,
        hyperlink_info_by_run: dict,
        para_element: etree._Element,
        include_inline_formatting: bool,
        markdown_mode: bool,
        include_links: bool,
        text_xpath: str = ".//w:t",
    ) -> list[tuple[str, bool]]:
        """
        Processes a single text run and returns formatted text segments.

        :param run: XML run element
        :param hyperlink_info_by_run: Dictionary mapping run IDs to hyperlink info
        :param para_element: Parent paragraph element for context
        :param include_inline_formatting: Whether to apply inline formatting
        :param markdown_mode: Whether in markdown mode
        :param include_links: Whether to process hyperlinks
        :param text_xpath: XPath to find text elements (default: ".//w:t")
        :return: List of (text, is_content) tuples where is_content indicates if
            it's text content
        """
        results = []
        run_element_id = id(run)
        is_hyperlink_run = run_element_id in hyperlink_info_by_run

        # Skip page number fields
        if self._is_page_number_field(run, para_element):
            return results

        # Extract formatting for this run if needed
        run_formatting = {}
        if include_inline_formatting and markdown_mode:
            run_formatting = self._extract_run_formatting(run)

        # Process text elements
        text_elements = _docx_xpath(run, text_xpath)
        for text_elem in text_elements:
            text_content = (
                # Attribute is defined in _Element
                text_elem.text_content()  # type: ignore[attr-defined]
                if hasattr(text_elem, "text_content")
                else (text_elem.text or "")
            )

            if text_content:
                # Check for tracked changes
                is_in_move_from = _docx_xpath(text_elem, "ancestor::w:moveFrom")
                is_in_deletion = _docx_xpath(text_elem, "ancestor::w:del")

                if not is_in_move_from and not is_in_deletion:
                    formatted_text = self._apply_text_formatting(
                        text_content=text_content,
                        run_formatting=run_formatting,
                        is_hyperlink_run=is_hyperlink_run,
                        hyperlink_info_by_run=hyperlink_info_by_run,
                        run_element_id=run_element_id,
                        markdown_mode=markdown_mode,
                        include_inline_formatting=include_inline_formatting,
                        include_links=include_links,
                    )
                    results.append((formatted_text, True))

        return results

    def _get_style_name(self, style_id: str, package: _DocxPackage) -> str:
        """
        Gets the style name from its ID by looking it up in the styles.xml.

        :param style_id: Style ID to look up
        :param package: _DocxPackage object containing the styles
        :return: Style name with Title Case formatting
        """
        if not style_id:
            return "Normal"

        if package.styles is None:
            return (style_id or "Normal").title()

        try:
            style_elements = _docx_xpath(
                package.styles, f".//w:style[@w:styleId='{style_id}']"
            )
            if style_elements:
                style_element = style_elements[0]
                name_elements = _docx_xpath(style_element, "w:name")
                if name_elements:
                    val = _docx_get_namespaced_attr(name_elements[0], "val")
                    if val:
                        return val.title()
        except Exception as e:
            # If there's an error finding the style, log it but continue with default
            logger.warning(f"Error looking up style '{style_id}': {e}")

        return (style_id or "Normal").title()

    def _get_paragraph_style(self, para_element: etree._Element) -> str:
        """
        Extracts the style information from a paragraph element.

        :param para_element: Paragraph XML element
        :return: Style ID string
        """
        # Find the paragraph properties element
        p_pr_elements = _docx_xpath(para_element, "w:pPr")
        if p_pr_elements:
            p_pr = p_pr_elements[0]
            # Find the style element within paragraph properties
            style_elements = _docx_xpath(p_pr, "w:pStyle")
            if style_elements:
                val = _docx_get_namespaced_attr(style_elements[0], "val")
                if val:
                    return val

        return "Normal"

    def _apply_markdown_formatting(
        self,
        text: str,
        style_name: str,
        is_list: bool,
        list_level: int,
        is_numbered: bool,
        para_element: etree._Element,
        list_counters: dict | None = None,
    ) -> str:
        """
        Applies document-level markdown formatting to text.

        :param text: Base text to format
        :param style_name: Style name of the paragraph
        :param is_list: Whether this is a list item
        :param list_level: List level (0-based)
        :param is_numbered: Whether this is a numbered list
        :param para_element: Paragraph XML element
        :param list_counters: Dictionary to track list numbering counters
        :return: Formatted markdown text
        """
        if style_name.lower().startswith("heading"):
            # Extract heading level (e.g., "Heading 1" -> 1)
            heading_level = 1
            match = re.search(r"(\d+)", style_name)
            if match:
                heading_level = int(match.group(1))
            return "#" * heading_level + " " + text

        elif is_list:
            # Add indentation based on list level
            indent = "    " * list_level

            if is_numbered and list_counters is not None:
                # Get the list ID for counter tracking
                p_pr_elements = _docx_xpath(para_element, "w:pPr")
                num_id = None
                if p_pr_elements:
                    p_pr = p_pr_elements[0]
                    num_pr_elements = _docx_xpath(p_pr, "w:numPr")
                    if num_pr_elements:
                        num_pr = num_pr_elements[0]
                        num_id_elements = _docx_xpath(num_pr, "w:numId")
                        if num_id_elements:
                            num_id = _docx_get_namespaced_attr(
                                num_id_elements[0], "val"
                            )

                if num_id:
                    list_key = (num_id, list_level)
                    # Get the counter value (should be set by the calling method)
                    counter = list_counters.get(list_key, 1)
                    return f"{indent}{counter}. {text}"
                else:
                    # Fallback to bullet format when numbering cannot be determined
                    logger.warning(
                        f"No numId found for numbered list item, "
                        f"using bullet format: {text[:50]}..."
                    )
                    return f"{indent}- {text}"
            elif is_numbered:
                # For numbered lists without counter tracking, use bullet format
                return f"{indent}- {text}"
            else:
                # For bullet lists, use "- " format
                return f"{indent}- {text}"

        else:
            # Regular paragraph
            return text

    def _extract_run_formatting(self, run_element: etree._Element) -> dict:
        """
        Extracts formatting properties from a run element.

        :param run_element: Run XML element (w:r)
        :return: Dictionary with formatting properties
        """
        formatting = {
            "bold": False,
            "italic": False,
            "underline": False,
            "strikethrough": False,
        }

        # Check for run properties (w:rPr)
        r_pr_elements = _docx_xpath(run_element, "w:rPr")
        if r_pr_elements:
            r_pr = r_pr_elements[0]

            # Bold (w:b or w:bCs for complex scripts)
            bold_elements = _docx_xpath(r_pr, "w:b")
            if bold_elements:
                # Check if bold is explicitly disabled (val="0" or val="false")
                val = _docx_get_namespaced_attr(bold_elements[0], "val")
                if val in ("", "1", "true", "on"):  # Empty means enabled by default
                    formatting["bold"] = True

            # Italic (w:i or w:iCs for complex scripts)
            italic_elements = _docx_xpath(r_pr, "w:i")
            if italic_elements:
                val = _docx_get_namespaced_attr(italic_elements[0], "val")
                if val in ("", "1", "true", "on"):
                    formatting["italic"] = True

            # Underline (w:u)
            underline_elements = _docx_xpath(r_pr, "w:u")
            if underline_elements:
                val = _docx_get_namespaced_attr(underline_elements[0], "val")
                # Only consider actual underline styles (not "none")
                if val and val != "none":
                    formatting["underline"] = True

            # Strikethrough (w:strike)
            strike_elements = _docx_xpath(r_pr, "w:strike")
            if strike_elements:
                val = _docx_get_namespaced_attr(strike_elements[0], "val")
                if val in ("", "1", "true", "on"):
                    formatting["strikethrough"] = True

        return formatting

    def _apply_inline_formatting(self, text: str, formatting: dict) -> str:
        """
        Applies markdown formatting to text based on detected formatting properties.

        :param text: Text content to format
        :param formatting: Dictionary with formatting properties from _extract_run_formatting
        :return: Text with markdown formatting applied
        """
        if not any(formatting.values()) or not text.strip():
            return text

        result = text

        # Apply formatting in order to ensure proper nesting
        # Order: strikethrough > underline > italic > bold
        # This creates the most readable nested markdown

        if formatting["strikethrough"]:
            result = f"~~{result}~~"  # GitHub-flavored markdown

        if formatting["underline"]:
            result = (
                f"<u>{result}</u>"  # HTML fallback (underline not standard markdown)
            )

        if formatting["italic"]:
            result = f"*{result}*"

        if formatting["bold"]:
            result = f"**{result}**"

        return result

    def _apply_text_formatting(
        self,
        text_content: str,
        run_formatting: dict,
        is_hyperlink_run: bool,
        hyperlink_info_by_run: dict,
        run_element_id: int,
        markdown_mode: bool,
        include_inline_formatting: bool,
        include_links: bool,
    ) -> str:
        """
        Applies formatting to text content.

        :param text_content: Raw text content
        :param run_formatting: Formatting dictionary from _extract_run_formatting
        :param is_hyperlink_run: Whether this run is part of a hyperlink
        :param hyperlink_info_by_run: Dictionary with hyperlink information
        :param run_element_id: ID of the run element
        :param markdown_mode: Whether in markdown mode
        :param include_inline_formatting: Whether to apply inline formatting
        :param include_links: Whether to process hyperlinks
        :return: Formatted text
        """
        formatted_text = text_content

        # Apply inline formatting if enabled
        if include_inline_formatting and markdown_mode and run_formatting:
            formatted_text = self._apply_inline_formatting(
                formatted_text, run_formatting
            )

        # Apply hyperlink formatting
        if is_hyperlink_run and markdown_mode and include_links:
            url = hyperlink_info_by_run[run_element_id]["url"]
            if url:
                formatted_text = f"[{formatted_text}]({url})"

        return formatted_text

    def _get_list_info(
        self, para_element: etree._Element, package: _DocxPackage
    ) -> tuple[bool, int, str, str, bool]:
        """
        Extracts list information from a paragraph element.

        :param para_element: Paragraph XML element
        :param package: _DocxPackage object
        :return: Tuple of (is_list, list_level, list_info_string, list_type, is_numbered)
        """
        is_list = False
        list_level = 0
        list_info = ""
        list_type = ""
        is_numbered = False

        p_pr_elements = _docx_xpath(para_element, "w:pPr")
        if p_pr_elements:
            p_pr = p_pr_elements[0]
            num_pr_elements = _docx_xpath(p_pr, "w:numPr")
            if num_pr_elements:
                num_pr = num_pr_elements[0]

                # Get list ID first to check if it's a valid list
                num_id_elements = _docx_xpath(num_pr, "w:numId")
                num_id = (
                    _docx_get_namespaced_attr(num_id_elements[0], "val")
                    if num_id_elements
                    else None
                )

                # Only treat as a list if numId exists and is not "0"
                # numId="0" in DOCX typically means "no numbering" - used to remove
                # numbering from paragraphs that inherit list formatting from styles
                if num_id and num_id != "0":
                    is_list = True

                    # Get level
                    ilvl_elements = _docx_xpath(num_pr, "w:ilvl")
                    if ilvl_elements:
                        list_level = int(
                            _docx_get_namespaced_attr(ilvl_elements[0], "val")
                        )

                    # Determine list type and numbering format if numbering is available
                    if package.numbering is not None:
                        # First find the abstractNumId associated with this numId
                        num_def_elements = _docx_xpath(
                            package.numbering, f".//w:num[@w:numId='{num_id}']"
                        )
                        if num_def_elements:
                            num_def = num_def_elements[0]
                            abstract_num_id_elements = _docx_xpath(
                                num_def, "w:abstractNumId"
                            )
                            if abstract_num_id_elements:
                                abstract_num_id = _docx_get_namespaced_attr(
                                    abstract_num_id_elements[0], "val"
                                )

                                # Now find the level formatting in the abstractNum
                                abstract_num_elements = _docx_xpath(
                                    package.numbering,
                                    f".//w:abstractNum[@w:abstractNumId='{abstract_num_id}']",
                                )
                                if abstract_num_elements:
                                    abstract_num = abstract_num_elements[0]
                                    # Find the level formatting for this specific level
                                    level_elements = _docx_xpath(
                                        abstract_num,
                                        f".//w:lvl[@w:ilvl='{list_level}']",
                                    )
                                    if level_elements:
                                        level_elem = level_elements[0]
                                        # Get the numFmt element which defines if it's
                                        # bullet or numbered
                                        num_fmt_elements = _docx_xpath(
                                            level_elem, "w:numFmt"
                                        )
                                        if (
                                            num_fmt_elements
                                            and _docx_get_namespaced_attr(
                                                num_fmt_elements[0], "val"
                                            )
                                        ):
                                            fmt_val = _docx_get_namespaced_attr(
                                                num_fmt_elements[0], "val"
                                            )
                                            list_type = fmt_val

                                            # Check if it's a numbered list format
                                            is_numbered = (
                                                fmt_val in NUMBERED_LIST_FORMATS
                                            )

                    # Only add list info if it's actually a list
                    list_info = f", List ID: {num_id}, List Level: {list_level}"
                    if list_type:
                        list_info += f", List Format: {list_type.capitalize()}"

        return is_list, list_level, list_info, list_type, is_numbered

    def _update_list_counters(
        self,
        para_element: etree._Element,
        package: _DocxPackage,
        list_counters: dict,
        last_list_id: str,
        last_list_level: int,
    ) -> tuple[str, int]:
        """
        Updates list counters for numbered lists and returns the new last list state.

        :param para_element: Paragraph XML element
        :param package: _DocxPackage object
        :param list_counters: Dictionary to track list numbering counters
        :param last_list_id: Last processed list ID
        :param last_list_level: Last processed list level
        :return: Tuple of (new_last_list_id, new_last_list_level)
        """
        # Extract list information
        p_pr_elements = _docx_xpath(para_element, "w:pPr")
        if p_pr_elements:
            p_pr = p_pr_elements[0]
            num_pr_elements = _docx_xpath(p_pr, "w:numPr")
            if num_pr_elements:
                num_pr = num_pr_elements[0]
                # This is a list item
                _, list_level, __, ___, is_numbered = self._get_list_info(
                    para_element, package
                )

                # Get num_id for this list item
                num_id_elements = _docx_xpath(num_pr, "w:numId")
                if num_id_elements:
                    num_id = _docx_get_namespaced_attr(num_id_elements[0], "val")

                    # If it's a numbered list, we need to track counter
                    if is_numbered:
                        list_key = (num_id, list_level)

                        # Reset counter if this is a new list or a higher level in the same list
                        if last_list_id != num_id or list_level < last_list_level:
                            # Reset counters for all levels below the current level
                            for key in list(list_counters.keys()):
                                if key[0] == num_id and key[1] > list_level:
                                    list_counters.pop(key)

                        # Initialize counter if needed
                        if list_key not in list_counters:
                            list_counters[list_key] = 1
                        else:
                            list_counters[list_key] += 1

                        # Remember this list for next iteration
                        return num_id, list_level

        return last_list_id, last_list_level

    def _process_table(
        self,
        table_element: etree._Element,
        package: _DocxPackage,
        markdown_mode: bool = False,
        table_idx: int = 0,
        populate_md_text: bool = False,
        include_textboxes: bool = True,
        include_links: bool = True,
        include_inline_formatting: bool = True,
        strict_mode: bool = False,
    ) -> list[str | Paragraph]:
        """
        Processes a table element and returns either paragraphs or markdown lines.

        :param table_element: Table XML element
        :param package: _DocxPackage object
        :param markdown_mode: If True, return markdown formatted lines,
            otherwise return Paragraph objects (default: False)
        :param table_idx: Index of the table in the document (default: 0)
        :param populate_md_text: If True, populate the _md_text field in Paragraph objects
            with markdown representation (default: False)
        :param include_textboxes: If True, include textbox content (default: True)
        :param include_links: If True, process and format hyperlinks (default: True)
        :param include_inline_formatting: If True, apply inline formatting (bold, italic, etc.)
            in markdown mode (default: True)
        :param strict_mode: If True, raise exceptions for any processing error
            instead of skipping problematic elements (default: False)
        :return: List of markdown lines or Paragraph objects
        """
        result = []

        try:
            if markdown_mode:
                # Process table for markdown output - only direct rows, not nested table rows
                rows = _docx_xpath(table_element, "./w:tr")
                if not rows:
                    return result

                # Collect all cell data and determine column widths
                all_rows = []
                col_widths = []

                for row in rows:
                    row_cells = []
                    cells = _docx_xpath(row, "./w:tc")
                    for cell in cells:
                        # Check for nested tables in this cell
                        nested_tables = _docx_xpath(cell, "./w:tbl")

                        # Combine all text from paragraphs in the cell
                        cell_text = []

                        # Process direct paragraphs (not in nested tables)
                        paragraphs = _docx_xpath(cell, "./w:p")
                        for para in paragraphs:
                            # Process paragraph
                            processed_para = self._process_paragraph(
                                para_element=para,
                                package=package,
                                markdown_mode=True,
                                strict_mode=strict_mode,
                                include_textboxes=include_textboxes,
                                apply_text_formatting=None,
                                populate_md_text=populate_md_text,
                                include_links=include_links,
                                include_inline_formatting=include_inline_formatting,
                                list_counters=None,  # list_counters - tables handle their own formatting
                            )
                            if processed_para:
                                cell_text.append(processed_para)

                        # Process nested tables if any
                        for nested_table in nested_tables:
                            nested_content = self._process_table(
                                table_element=nested_table,
                                package=package,
                                markdown_mode=True,
                                table_idx=-1,
                                strict_mode=strict_mode,
                                include_textboxes=include_textboxes,
                                populate_md_text=populate_md_text,
                                include_links=include_links,
                                include_inline_formatting=include_inline_formatting,
                            )
                            if nested_content:
                                # Join nested table content as text
                                # Safe cast: nested_content is a list of str objects
                                nested_text = "\n".join(
                                    cast(str, line)
                                    for line in nested_content
                                    if cast(str, line).strip()
                                )
                                if nested_text:
                                    cell_text.append(nested_text)

                        # Join cell content, using appropriate separators
                        if len(cell_text) > 1 and any(
                            "|" in text and "---" in text for text in cell_text
                        ):
                            # Contains nested table - use newlines to separate content
                            cell_content = "\n".join(cell_text).strip() or " "
                        elif len(cell_text) > 1:
                            # Multiple paragraphs in cell - use newlines to separate
                            cell_content = "\n".join(cell_text).strip() or " "
                        else:
                            # Single paragraph or empty content
                            cell_content = " ".join(cell_text).strip() or " "
                        row_cells.append(cell_content)

                    all_rows.append(row_cells)

                    # Update max widths
                    if not col_widths:
                        col_widths = [len(cell) for cell in row_cells]
                    else:
                        for i, cell in enumerate(row_cells):
                            if i < len(col_widths):
                                col_widths[i] = max(col_widths[i], len(cell))

                # Format the table as markdown
                for row_idx, row_cells in enumerate(all_rows):
                    # Pad cells for alignment
                    padded_cells = []
                    for i, cell in enumerate(row_cells):
                        if i < len(col_widths):
                            padded_cells.append(cell.ljust(col_widths[i]))
                        else:
                            padded_cells.append(cell)

                    result.append("| " + " | ".join(padded_cells) + " |")

                    # Add header separator after first row
                    if row_idx == 0:
                        separator = []
                        for width in col_widths[: len(row_cells)]:
                            separator.append("-" * width)
                        result.append("| " + " | ".join(separator) + " |")

                # Add blank line after table
                result.append("")
            else:
                # Process table for Paragraph objects - only direct rows, not nested table rows
                table_metadata = f"Table ID: {table_idx + 1}"
                rows = _docx_xpath(table_element, "./w:tr")

                for row_idx, row in enumerate(rows):
                    cells = _docx_xpath(row, "./w:tc")
                    for cell_idx, cell in enumerate(cells):
                        # Check for nested tables in this cell
                        nested_tables = _docx_xpath(cell, "./w:tbl")

                        # Process direct paragraphs (not in nested tables)
                        paragraphs = _docx_xpath(cell, "./w:p")
                        for para in paragraphs:
                            # Process paragraph
                            processed_para = self._process_paragraph(
                                para_element=para,
                                package=package,
                                markdown_mode=False,
                                strict_mode=strict_mode,
                                include_textboxes=include_textboxes,
                                apply_text_formatting=None,
                                populate_md_text=populate_md_text,
                                include_links=include_links,
                                include_inline_formatting=include_inline_formatting,
                                list_counters=None,  # list_counters - tables handle their own formatting
                            )
                            if processed_para:
                                # Safe cast: processed_para is a Paragraph object
                                processed_para = cast(Paragraph, processed_para)
                                style_id = self._get_paragraph_style(para)
                                style_name = self._get_style_name(style_id, package)
                                cell_style_info = f"Style: {style_name}"
                                if not processed_para.additional_context:
                                    raise RuntimeError(
                                        "Paragraph must have additional context."
                                    )

                                # Copy the paragraph with added table metadata
                                cell_para = Paragraph(
                                    raw_text=processed_para.raw_text,
                                    additional_context=f"{cell_style_info}, {table_metadata}, "
                                    f"Row: {row_idx + 1}, Column: {cell_idx + 1}, "
                                    f"Table Cell"
                                    + (
                                        ", "
                                        + processed_para.additional_context.split(
                                            ", ", 1
                                        )[1]
                                        if ", " in processed_para.additional_context
                                        else ""
                                    ),
                                )
                                # Assign _md_text if populate_md_text was requested
                                if populate_md_text:
                                    cell_para._md_text = processed_para._md_text

                                result.append(cell_para)

                        # Process nested tables if any
                        for nested_table_idx, nested_table in enumerate(nested_tables):
                            nested_content = self._process_table(
                                table_element=nested_table,
                                package=package,
                                markdown_mode=False,
                                table_idx=nested_table_idx,  # Use the actual nested table index
                                strict_mode=strict_mode,
                                include_textboxes=include_textboxes,
                                populate_md_text=populate_md_text,
                                include_links=include_links,
                                include_inline_formatting=include_inline_formatting,
                            )
                            # Add nested table content with proper metadata
                            for nested_para in nested_content:
                                if isinstance(nested_para, Paragraph):
                                    # Extract style and other info from nested table metadata
                                    nested_context = nested_para.additional_context
                                    if not nested_context:
                                        raise RuntimeError(
                                            "Nested paragraph must have additional context."
                                        )
                                    parts = nested_context.split(", ")

                                    # Find and extract the style (should be first)
                                    style_part = ""
                                    remaining_parts = []

                                    for part in parts:
                                        if part.startswith("Style: "):
                                            style_part = part
                                        elif not part.startswith(
                                            "Table ID: "
                                        ):  # Skip redundant "Table ID: X"
                                            remaining_parts.append(part)

                                    # Build the final metadata: Style first, then parent table info,
                                    # then nested table info
                                    final_context = (
                                        f"{style_part}, {table_metadata}, Row: {row_idx + 1}, "
                                        f"Column: {cell_idx + 1}, "
                                        f"Nested Table ID: {nested_table_idx + 1}"
                                    )

                                    # Add remaining parts (like Row, Column, Table Cell from
                                    # nested table)
                                    if remaining_parts:
                                        final_context += ", " + ", ".join(
                                            remaining_parts
                                        )

                                    nested_meta = Paragraph(
                                        raw_text=nested_para.raw_text,
                                        additional_context=final_context,
                                    )
                                    # Assign _md_text if populate_md_text was requested
                                    if populate_md_text:
                                        nested_meta._md_text = nested_para._md_text

                                    result.append(nested_meta)

            return result
        except Exception as e:
            # Handle table parsing errors
            if isinstance(e, DocxConverterError):
                # Re-raise specific converter errors
                raise
            else:
                if strict_mode:
                    raise DocxContentError(f"Error processing table: {e}") from e
                else:
                    logger.warning(f"Error processing table (idx: {table_idx}): {e}")
                    # Return whatever we've processed so far
                    return result

    def _process_document_section(
        self,
        package: _DocxPackage,
        section_name: str,
        section_data: dict,
        metadata_key: str,
        include_textboxes: bool = True,
        populate_md_text: bool = False,
        strict_mode: bool = False,
    ) -> list[Paragraph]:
        """
        Generic processor for document sections (headers, footers, footnotes, comments).

        :param package: _DocxPackage object
        :param section_name: Name of the section for error reporting
        :param section_data: Data structure containing the section elements
        :param metadata_key: Key to use in metadata (e.g., "Header ID", "Footer ID")
        :param include_textboxes: Whether to include textbox content
        :param populate_md_text: Whether to populate _md_text field
        :param strict_mode: Whether to raise exceptions on errors
        :return: List of Paragraph objects
        """
        paragraphs = []

        if not section_data:
            return paragraphs

        try:
            for section_id, section_info in section_data.items():
                # Handle different data structures
                if isinstance(section_info, dict):
                    if "content" in section_info:
                        # Headers/footers structure
                        content = section_info["content"]
                        xpath_expr = ".//w:p"
                    else:
                        # Other structures
                        content = section_info
                        xpath_expr = ".//w:p"
                else:
                    # Direct XML content
                    content = section_info
                    xpath_expr = ".//w:p"

                # Extract additional metadata for comments
                extra_metadata = ""
                if hasattr(content, "attrib"):
                    # Safe cast: content is an _Element object
                    content = cast(etree._Element, content)
                    author = _docx_get_namespaced_attr(content, "author")
                    if author:
                        extra_metadata += f", Author: {author}"
                    date = _docx_get_namespaced_attr(content, "date")
                    if date:
                        extra_metadata += f", Date: {date}"

                # Find paragraphs in the content
                # Safe cast: content is an _Element object
                content_paragraphs = _docx_xpath(
                    cast(etree._Element, content), xpath_expr
                )

                for para in content_paragraphs:
                    processed_para = self._process_paragraph(
                        para_element=para,
                        package=package,
                        markdown_mode=False,
                        strict_mode=strict_mode,
                        include_textboxes=include_textboxes,
                        apply_text_formatting=None,
                        populate_md_text=populate_md_text,
                        include_links=False,  # Always disabled for headers/footers/comments/footnotes
                        list_counters=None,  # list_counters - tables handle their own formatting
                        include_inline_formatting=False,  # Always disabled for headers/footers/comments/footnotes
                    )

                    if processed_para:
                        # Add section-specific metadata
                        # Safe cast: processed_para is a Paragraph object
                        processed_para = cast(Paragraph, processed_para)
                        original_context = processed_para.additional_context
                        section_context = (
                            f"{original_context}, {metadata_key}: "
                            f"{section_id}{extra_metadata}"
                        )

                        # Create paragraph with updated metadata
                        paragraph = Paragraph(
                            raw_text=processed_para.raw_text,
                            additional_context=section_context,
                        )
                        if populate_md_text:
                            paragraph._md_text = processed_para._md_text

                        paragraphs.append(paragraph)

        except Exception as e:
            if strict_mode:
                raise DocxContentError(f"Error processing {section_name}: {e}") from e
            else:
                logger.warning(f"Error processing {section_name}: {e}")

        return paragraphs

    def _handle_section_in_markdown_mode(
        self,
        paragraphs: list[Paragraph],
        section_title: str,
        result: list[str],
        extract_id_func: Callable[[str], str] | None = None,
    ) -> None:
        """
        Handles displaying a document section in markdown mode.

        :param paragraphs: List of paragraph objects from the section
        :param section_title: Title to display for the section (e.g., "Header", "Footer")
        :param result: Result list to append to
        :param extract_id_func: Optional function to extract ID from paragraph context
        """
        if not paragraphs:
            return

        for para in paragraphs:
            # Extract ID for display if function provided
            display_title = section_title
            if extract_id_func:
                try:
                    if not para.additional_context:
                        raise RuntimeError("Paragraph must have additional context.")
                    section_id = extract_id_func(para.additional_context)
                    if section_id:
                        display_title = f"{section_title} {section_id}"
                except (IndexError, AttributeError, TypeError) as e:
                    # Use default title if extraction fails
                    logger.debug(f"Failed to extract section ID from context: {e}")

            # Use _md_text if available and different from raw_text
            display_text = para.raw_text
            if para._md_text and para._md_text != para.raw_text:
                display_text = para._md_text

            result.append(f"**{display_title}**: {display_text}")
            result.append("")

    def _process_markup_compatibility_element(
        self,
        mc_element: etree._Element,
        hyperlink_info_by_run: dict,
        para_element: etree._Element,
        markdown_mode: bool,
        include_inline_formatting: bool,
        include_links: bool,
        include_textboxes: bool,
    ) -> list[str]:
        """
        Processes a markup compatibility element (Choice or Fallback).

        :param mc_element: Choice or Fallback XML element
        :param hyperlink_info_by_run: Dictionary mapping run IDs to hyperlink info
        :param para_element: Parent paragraph element for context
        :param markdown_mode: Whether in markdown mode
        :param include_inline_formatting: Whether to apply inline formatting
        :param include_links: Whether to process hyperlinks
        :param include_textboxes: Whether to include textbox content
        :return: List of formatted text strings
        """
        # If include_textboxes is False, skip textboxes in markup compatibility content
        if not include_textboxes:
            # Skip textbox content within this element
            has_textbox = (
                _docx_xpath(mc_element, ".//v:textbox")
                or _docx_xpath(mc_element, ".//w:txbxContent")
                or _docx_xpath(mc_element, ".//a:t")
                or _docx_xpath(mc_element, ".//w:drawing")
            )
            if has_textbox:
                return []

        return self._process_textbox_runs(
            mc_element,
            hyperlink_info_by_run,
            para_element,
            include_inline_formatting,
            markdown_mode,
            include_links,
        )

    def _process_headers(
        self,
        package: _DocxPackage,
        include_textboxes: bool = True,
        populate_md_text: bool = False,
        strict_mode: bool = False,
    ) -> list[Paragraph]:
        """
        Processes headers from the header XML files and converts them to Paragraph objects.

        :param package: _DocxPackage object
        :param include_textboxes: If True, include textbox content (default: True)
        :param populate_md_text: If True, populate the _md_text field in Paragraph objects
            with markdown representation (default: False)
        :param strict_mode: If True, raise exceptions for any processing error
            instead of skipping problematic elements (default: False)
        :return: List of Paragraph objects representing headers
        """
        return self._process_document_section(
            package=package,
            section_name="headers",
            section_data=package.headers,
            metadata_key="Header ID",
            strict_mode=strict_mode,
            include_textboxes=include_textboxes,
            populate_md_text=populate_md_text,
        )

    def _process_footers(
        self,
        package: _DocxPackage,
        include_textboxes: bool = True,
        populate_md_text: bool = False,
        strict_mode: bool = False,
    ) -> list[Paragraph]:
        """
        Processes footers from the footer XML files and converts them to Paragraph objects.

        :param package: _DocxPackage object
        :param include_textboxes: If True, include textbox content (default: True)
        :param populate_md_text: If True, populate the _md_text field in Paragraph objects
            with markdown representation (default: False)
        :param strict_mode: If True, raise exceptions for any processing error
            instead of skipping problematic elements (default: False)
        :return: List of Paragraph objects representing footers
        """
        return self._process_document_section(
            package=package,
            section_name="footers",
            section_data=package.footers,
            metadata_key="Footer ID",
            strict_mode=strict_mode,
            include_textboxes=include_textboxes,
            populate_md_text=populate_md_text,
        )

    def _process_footnotes(
        self,
        package: _DocxPackage,
        include_textboxes: bool = True,
        populate_md_text: bool = True,
        strict_mode: bool = False,
    ) -> list[Paragraph]:
        """
        Processes footnotes from the footnotes.xml file and converts them to Paragraph objects.

        :param package: _DocxPackage object
        :param include_textboxes: If True, include textbox content (default: True)
        :param populate_md_text: If True, populate the _md_text field in Paragraph objects
            with markdown representation (default: True)
        :param strict_mode: If True, raise exceptions for any processing error
            instead of skipping problematic elements (default: False)
        :return: List of Paragraph objects representing footnotes
        """
        if package.footnotes is None:
            return []

        # Build a dictionary of footnote elements, filtering out separators
        footnote_elements = _docx_xpath(package.footnotes, ".//w:footnote")
        footnote_data = {}
        for footnote_elem in footnote_elements:
            footnote_id = _docx_get_namespaced_attr(footnote_elem, "id")
            if not footnote_id:
                continue
            if footnote_id not in ("-1", "0"):  # Skip separators
                footnote_data[footnote_id] = footnote_elem

        return self._process_document_section(
            package=package,
            section_name="footnotes",
            section_data=footnote_data,
            metadata_key="Footnote ID",
            strict_mode=strict_mode,
            include_textboxes=include_textboxes,
            populate_md_text=populate_md_text,
        )

    def _process_comments(
        self,
        package: _DocxPackage,
        include_textboxes: bool = True,
        populate_md_text: bool = True,
        strict_mode: bool = False,
    ) -> list[Paragraph]:
        """
        Processes comments from the comments.xml file and converts them to Paragraph objects.

        :param package: _DocxPackage object
        :param include_textboxes: If True, include textbox content (default: True)
        :param populate_md_text: If True, populate the _md_text field in Paragraph objects
            with markdown representation (default: True)
        :param strict_mode: If True, raise exceptions for any processing error
            instead of skipping problematic elements (default: False)
        :return: List of Paragraph objects representing comments
        """
        if package.comments is None:
            return []

        # Build a dictionary of comment elements
        comment_elements = _docx_xpath(package.comments, ".//w:comment")
        comment_data = {}
        for comment_elem in comment_elements:
            comment_id = _docx_get_namespaced_attr(comment_elem, "id")
            if not comment_id:
                continue
            comment_data[comment_id] = comment_elem

        return self._process_document_section(
            package=package,
            section_name="comments",
            section_data=comment_data,
            metadata_key="Comment ID",
            strict_mode=strict_mode,
            include_textboxes=include_textboxes,
            populate_md_text=populate_md_text,
        )

    def _extract_footnote_references(self, para_element: etree._Element) -> list[str]:
        """
        Extracts footnote references from a paragraph.

        :param para_element: Paragraph XML element
        :return: List of footnote IDs
        """
        footnote_ids = []

        # Find all footnote references in this paragraph
        runs = _docx_xpath(para_element, ".//w:r")
        for run in runs:
            footnote_refs = _docx_xpath(run, ".//w:footnoteReference")
            for footnote_ref in footnote_refs:
                footnote_id = _docx_get_namespaced_attr(footnote_ref, "id")
                if footnote_id:
                    footnote_ids.append(footnote_id)

        return footnote_ids

    def _extract_comment_references(self, para_element: etree._Element) -> list[str]:
        """
        Extracts comment references from a paragraph.

        :param para_element: Paragraph XML element
        :return: List of comment IDs
        """
        comment_ids = []

        # Find all comment references in this paragraph
        runs = _docx_xpath(para_element, ".//w:r")
        for run in runs:
            comment_refs = _docx_xpath(run, ".//w:commentReference")
            for comment_ref in comment_refs:
                comment_id = _docx_get_namespaced_attr(comment_ref, "id")
                if comment_id:
                    comment_ids.append(comment_id)

        return comment_ids

    def _extract_hyperlink_references(
        self, para_element: etree._Element, package: _DocxPackage
    ) -> list[dict]:
        """
        Extracts hyperlink references from a paragraph.

        :param para_element: Paragraph XML element
        :param package: _DocxPackage object containing hyperlink relationships
        :return: List of dictionaries with hyperlink info:
            [{"text": str, "url": str, "rel_id": str}]
        """
        hyperlinks = []
        seen_urls = set()  # Track URLs to avoid duplication

        # Find all hyperlink elements in this paragraph
        hyperlink_elements = _docx_xpath(para_element, ".//w:hyperlink")
        for hyperlink_elem in hyperlink_elements:
            # Get the relationship ID
            rel_id = _docx_get_namespaced_attr(hyperlink_elem, "id", "r")

            # Get the hyperlink URL from relationships
            url = package.hyperlinks.get(rel_id, "")

            # Extract the text content of the hyperlink
            text_elements = _docx_xpath(hyperlink_elem, ".//w:t")
            link_text = ""
            for text_elem in text_elements:
                text_content = (
                    # Attribute is defined in _Element
                    text_elem.text_content()  # type: ignore[attr-defined]
                    if hasattr(text_elem, "text_content")
                    else (text_elem.text or "")
                )
                link_text += text_content

            # Only add if we have text, URL, and haven't seen this URL before
            if link_text and url and url not in seen_urls:
                hyperlinks.append({"text": link_text, "url": url, "rel_id": rel_id})
                seen_urls.add(url)

        return hyperlinks

    def _is_text_box_paragraph(self, para_element: etree._Element) -> bool:
        """
        Determines if a paragraph is from a text box.

        :param para_element: Paragraph XML element
        :return: True if the paragraph is part of a text box
        """
        # Check for various types of text boxes in Word
        # 1. VML textbox (older Word format)
        if _docx_xpath(para_element, ".//v:textbox"):
            return True

        # 2. DrawingML text box (Office 2007+)
        if _docx_xpath(para_element, ".//w:txbxContent"):
            return True

        # 3. Check for shape with text
        if _docx_xpath(para_element, ".//a:t"):
            return True

        # 4. Check for drawing element
        return bool(_docx_xpath(para_element, ".//w:drawing"))

    def _process_textbox_runs(
        self,
        container_element: etree._Element,
        hyperlink_info_by_run: dict,
        para_element: etree._Element,
        include_inline_formatting: bool,
        markdown_mode: bool,
        include_links: bool,
        text_xpath: str = ".//w:t",
        run_xpath: str = ".//w:r",
    ) -> list[str]:
        """
        Processes runs within textbox containers.

        :param container_element: Container element (textbox, drawing, etc.)
        :param hyperlink_info_by_run: Dictionary mapping run IDs to hyperlink info
        :param para_element: Parent paragraph element for context
        :param include_inline_formatting: Whether to apply inline formatting
        :param markdown_mode: Whether in markdown mode
        :param include_links: Whether to process hyperlinks
        :param text_xpath: XPath to find text elements (default: ".//w:t")
        :param run_xpath: XPath to find run elements (default: ".//w:r")
        :return: List of formatted text strings
        """
        results = []
        runs = _docx_xpath(container_element, run_xpath)

        for run in runs:
            run_results = self._process_text_run(
                run=run,
                hyperlink_info_by_run=hyperlink_info_by_run,
                para_element=para_element,
                include_inline_formatting=include_inline_formatting,
                markdown_mode=markdown_mode,
                include_links=include_links,
                text_xpath=text_xpath,
            )
            text_elements = _docx_xpath(run, text_xpath)
            content_results = [r for r in run_results if r[1]]
            for text_elem, text_result in zip(
                text_elements, content_results, strict=True
            ):
                formatted_text, _ = text_result
                original_content = (
                    # Attribute is defined in _Element
                    text_elem.text_content()  # type: ignore[attr-defined]
                    if hasattr(text_elem, "text_content")
                    else (text_elem.text or "")
                )
                if original_content:
                    # Include all textbox content
                    results.append(formatted_text)

        return results

    def _is_page_number_field(
        self, run_element: etree._Element, para_element: etree._Element
    ) -> bool:
        """
        Determines if a run element contains a page number field.

        Page numbers in DOCX can appear as:
        1. Simple fields: w:fldSimple with PAGE, NUMPAGES, etc.
        2. Complex fields: w:fldChar with PAGE instruction
        3. In headers/footers as standalone numeric content

        :param run_element: Run XML element (w:r)
        :param para_element: Parent paragraph XML element for context
        :return: True if the run contains a page number field
        """

        # Check for simple page number fields in this run
        simple_fields = _docx_xpath(run_element, ".//w:fldSimple")
        for field in simple_fields:
            instr = _docx_get_namespaced_attr(field, "instr").upper()
            if any(page_field in instr for page_field in PAGE_FIELD_KEYWORDS):
                return True

        # Check for complex page number fields
        # Look for field begin in this run or previous runs in the same paragraph
        all_runs = _docx_xpath(para_element, ".//w:r")
        current_run_idx = None
        for idx, run in enumerate(all_runs):
            if run == run_element:
                current_run_idx = idx
                break

        if current_run_idx is not None:
            # Check if we're inside a page number field
            field_started = False
            for idx in range(current_run_idx + 1):  # Include current run
                run = all_runs[idx]

                # Check for field begin
                field_begins = _docx_xpath(run, ".//w:fldChar[@w:fldCharType='begin']")
                if field_begins:
                    field_started = True

                # Check for field instruction with page number keywords
                if field_started:
                    instr_texts = _docx_xpath(run, ".//w:instrText")
                    for instr in instr_texts:
                        if instr.text:
                            instr_upper = instr.text.upper()
                            if any(
                                page_field in instr_upper
                                for page_field in PAGE_FIELD_KEYWORDS
                            ):
                                # Check if current run is between field begin and end
                                for check_idx in range(idx, len(all_runs)):
                                    check_run = all_runs[check_idx]
                                    if check_run == run_element:
                                        return True
                                    field_ends = _docx_xpath(
                                        check_run, ".//w:fldChar[@w:fldCharType='end']"
                                    )
                                    if field_ends:
                                        break

                # Check for field end (reset field_started)
                field_ends = _docx_xpath(run, ".//w:fldChar[@w:fldCharType='end']")
                if field_ends:
                    field_started = False

        return False

    def _extract_images(
        self, package: _DocxPackage, strict_mode: bool = False
    ) -> list[Image]:
        """
        Extracts images from the DOCX document.

        :param package: _DocxPackage object
        :param strict_mode: If True, raise exceptions for any processing error
            instead of skipping problematic elements (default: False)
        :return: List of Image objects
        """
        images = []
        img_count = 0
        error_count = 0
        duplicate_count = 0
        seen_base64_data = set()  # Track base64 data to avoid duplicates

        try:
            logger.debug(
                f"Extracting images from DOCX (found {len(package.images)} images)"
            )
            for rel_id, image_info in package.images.items():
                # Get image data and mime type
                image_bytes = image_info["data"]
                mime_type = image_info["mime_type"]

                # Ensure mime type is supported
                if mime_type not in {
                    "image/jpg",
                    "image/jpeg",
                    "image/png",
                    "image/webp",
                }:
                    # Default to PNG if unsupported
                    logger.debug(
                        f"Unsupported image MIME type: {mime_type}, defaulting to image/png"
                    )
                    mime_type = "image/png"

                try:
                    # Convert to base64
                    b64_data = base64.b64encode(image_bytes).decode("utf-8")

                    # Check for duplicates based on base64 data content
                    # If we create duplicate Image objects and add them to the document,
                    # a ValueError will be raised.
                    if b64_data in seen_base64_data:
                        duplicate_count += 1
                        logger.debug("Skipping duplicate image")
                        continue

                    # Add to seen set and create image instance
                    seen_base64_data.add(b64_data)
                    img_instance = Image(base64_data=b64_data, mime_type=mime_type)
                    images.append(img_instance)
                    img_count += 1
                except Exception as e:
                    # If in strict mode, raise the error
                    if strict_mode:
                        raise DocxContentError(
                            f"Error converting image '{image_info.get('target', rel_id)}': "
                            f"{e}"
                        ) from e

                    # Otherwise log the error and continue with the next image
                    error_count += 1
                    logger.warning(
                        f"Error converting image '{image_info.get('target', rel_id)}': "
                        f"{e}"
                    )
                    continue

            if img_count > 0:
                logger.info(f"Successfully extracted {img_count} images from DOCX")
            if duplicate_count > 0:
                logger.info(f"Skipped {duplicate_count} duplicate images from DOCX")
            if error_count > 0:
                logger.warning(f"Failed to extract {error_count} images from DOCX")

            return images
        except Exception as e:
            # Handle critical errors extracting images
            raise DocxConverterError(f"Error extracting images from DOCX: {e}") from e
