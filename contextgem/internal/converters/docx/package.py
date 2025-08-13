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
Package for the DOCX converter module.

This module provides a class for representing a DOCX file as a package and
providing access to its XML parts.
"""

from __future__ import annotations

import os
import zipfile
from pathlib import Path
from typing import BinaryIO

from lxml import etree

from contextgem.internal.converters.docx.utils import _docx_get_attr, _docx_xpath
from contextgem.internal.exceptions import (
    DocxContentError,
    DocxConverterError,
    DocxFormatError,
    DocxXmlError,
)
from contextgem.internal.loggers import logger


class _DocxPackage:
    """
    Represents a DOCX file as a package and provides access to its XML parts.
    """

    def __init__(self, docx_path_or_file: str | Path | BinaryIO):
        """
        Initialize with either a path to a DOCX file or a file-like object.

        :param docx_path_or_file: Path to DOCX file (as string or Path object) or file-like object
        """
        self.archive: zipfile.ZipFile
        self.rels = {}
        self.main_document = None
        self.styles = None
        self.numbering = None
        self.footnotes = None
        self.comments = None
        self.headers = {}
        self.footers = {}
        self.images = {}
        self.hyperlinks = {}

        file_desc = (
            docx_path_or_file
            if isinstance(docx_path_or_file, str | Path)
            else "file object"
        )

        try:
            self.archive = zipfile.ZipFile(docx_path_or_file)
        except zipfile.BadZipFile as e:
            raise DocxFormatError(f"'{file_desc}' is not a valid ZIP file") from e
        except FileNotFoundError as e:
            raise DocxFormatError(f"File '{file_desc}' not found") from e
        except PermissionError as e:
            raise DocxFormatError(
                f"Permission denied when accessing '{file_desc}'"
            ) from e
        except Exception as e:
            raise DocxFormatError(f"Failed to open DOCX file '{file_desc}': {e}") from e

        try:
            # Check if this is actually a DOCX file by looking for key parts
            if "word/document.xml" not in self.archive.namelist():
                raise DocxFormatError(
                    f"'{file_desc}' is not a valid DOCX file (missing word/document.xml)"
                )

            # Load main parts
            self._load_relationships()
            self._load_main_document()
            self._load_styles()
            self._load_numbering()
            self._load_footnotes()
            self._load_comments()
            self._load_headers_footers()
            self._load_images()
        except DocxConverterError:
            # Re-raise specific converter errors
            raise
        except etree.XMLSyntaxError as e:
            raise DocxXmlError(f"XML parsing error in '{file_desc}': {e}") from e
        except KeyError as e:
            raise DocxContentError(
                f"Missing required content in '{file_desc}': {e}"
            ) from e
        except Exception as e:
            raise DocxConverterError(
                f"Error processing DOCX file '{file_desc}': {e}"
            ) from e

    def _load_xml_part(self, part_path: str) -> etree._Element | None:
        """
        Loads an XML part from the DOCX package.

        :param part_path: Path to the XML part within the DOCX package
        :return: lxml Element or None if the part doesn't exist
        """
        if part_path not in self.archive.namelist():
            return None

        try:
            data = self.archive.read(part_path)
            return etree.fromstring(data)
        except etree.XMLSyntaxError as e:
            raise DocxXmlError(f"Failed to parse XML in '{part_path}': {e}") from e
        except Exception as e:
            raise DocxConverterError(f"Error reading '{part_path}': {e}") from e

    def _load_relationships(self) -> None:
        """
        Loads the document relationship definitions that connect document parts.
        """
        # Document relationships
        self.rels["document"] = {}
        if "word/_rels/document.xml.rels" in self.archive.namelist():
            doc_rels_root = self._load_xml_part("word/_rels/document.xml.rels")
            if doc_rels_root is not None:
                relationships = _docx_xpath(doc_rels_root, ".//rels:Relationship")
                for rel in relationships:
                    # Relationship attributes are not namespaced
                    rel_id = _docx_get_attr(rel, "Id")
                    rel_type = _docx_get_attr(rel, "Type")
                    rel_target = _docx_get_attr(rel, "Target")

                    self.rels["document"][rel_id] = {
                        "type": rel_type,
                        "target": rel_target,
                    }

                    # Store hyperlinks separately for easy access
                    if "hyperlink" in rel_type.lower():
                        self.hyperlinks[rel_id] = rel_target

    def _load_main_document(self) -> None:
        """
        Loads the main document.xml content.
        """
        self.main_document = self._load_xml_part("word/document.xml")
        if self.main_document is None:
            raise DocxContentError(
                "Main document (word/document.xml) is missing or invalid"
            )

    def _load_styles(self) -> None:
        """
        Loads the styles.xml content.
        """
        self.styles = self._load_xml_part("word/styles.xml")

    def _load_numbering(self) -> None:
        """
        Loads the numbering.xml content for lists.
        """
        self.numbering = self._load_xml_part("word/numbering.xml")

    def _load_footnotes(self) -> None:
        """
        Loads the footnotes.xml content.
        """
        self.footnotes = self._load_xml_part("word/footnotes.xml")

    def _load_comments(self) -> None:
        """
        Loads the comments.xml content.
        """
        self.comments = self._load_xml_part("word/comments.xml")

    def _load_headers_footers(self) -> None:
        """
        Loads headers and footers referenced in the document.
        """
        if not self.rels.get("document"):
            return

        # Find all header and footer relationships
        for rel_id, rel_info in self.rels["document"].items():
            rel_type = rel_info["type"].lower()
            target = rel_info["target"]

            # Handle relative paths
            target = f"word/{target}" if not target.startswith("/") else target[1:]

            # Load headers
            if "header" in rel_type:
                try:
                    header_content = self._load_xml_part(target)
                    if header_content is not None:
                        self.headers[rel_id] = {
                            "target": target,
                            "content": header_content,
                        }
                except DocxXmlError:
                    # Re-raise XML errors
                    raise
                except Exception as e:
                    raise DocxConverterError(
                        f"Error loading header '{target}': {e}"
                    ) from e

            # Load footers
            elif "footer" in rel_type:
                try:
                    footer_content = self._load_xml_part(target)
                    if footer_content is not None:
                        self.footers[rel_id] = {
                            "target": target,
                            "content": footer_content,
                        }
                except DocxXmlError:
                    # Re-raise XML errors
                    raise
                except Exception as e:
                    raise DocxConverterError(
                        f"Error loading footer '{target}': {e}"
                    ) from e

    def _load_images(self) -> None:
        """
        Loads all images embedded in the document.
        """
        if not self.rels.get("document"):
            return

        for rel_id, rel_info in self.rels["document"].items():
            if "image" in rel_info["type"].lower():
                target = rel_info["target"]
                # Handle relative paths
                target = f"word/{target}" if not target.startswith("/") else target[1:]

                try:
                    if target in self.archive.namelist():
                        image_data = self.archive.read(target)
                        self.images[rel_id] = {
                            "data": image_data,
                            "target": target,
                            # Extract mime type from target extension
                            "mime_type": self._get_mime_type(target),
                        }
                except Exception as e:
                    raise DocxConverterError(
                        f"Error loading image '{target}': {e}"
                    ) from e

    def _get_mime_type(self, target: str) -> str:
        """
        Determines the MIME type from the file extension.

        :param target: Image file path
        :return: MIME type string
        """
        ext = os.path.splitext(target.lower())[1]
        mime_types = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".webp": "image/webp",
        }
        return mime_types.get(ext, "image/png")  # Default to PNG if unknown

    def close(self) -> None:
        """Closes the zip archive."""
        if self.archive:
            try:
                self.archive.close()
            except Exception as e:
                # Just log the error but don't raise, as this is cleanup code
                logger.warning(f"Error closing DOCX archive: {e}")
