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
Utility functions for the DOCX converter.

This module contains helper functions used by the DOCX converter components.
"""

from lxml import etree

from contextgem.internal.converters.docx.namespaces import WORD_XML_NAMESPACES


NUMBERED_LIST_FORMATS = frozenset(
    {
        "decimal",
        "decimalZero",
        "upperRoman",
        "lowerRoman",
        "upperLetter",
        "lowerLetter",
        "ordinal",
        "cardinalText",
        "ordinalText",
        "hex",
        "chicago",
        "ideographDigital",
        "japaneseCounting",
        "aiueo",
        "iroha",
        "arabicFullWidth",
        "hindiNumbers",
        "thaiNumbers",
    }
)

PAGE_FIELD_KEYWORDS = frozenset(["PAGE", "NUMPAGES", "PAGEREF", "SECTIONPAGES"])


def _docx_get_namespaced_attr(
    element: etree._Element, attr_name: str, namespace: str = "w"
) -> str:
    """
    Helper function to get namespaced attributes using lxml efficiently.

    :param element: The XML element
    :param attr_name: The attribute name (e.g., 'val', 'id')
    :param namespace: The namespace key from WORD_XML_NAMESPACES (default: 'w')
    :return: The attribute value or empty string if not found
    """
    return element.get(f"{{{WORD_XML_NAMESPACES[namespace]}}}{attr_name}", "")


def _docx_get_attr(element: etree._Element, attr_name: str) -> str:
    """
    Helper function to get non-namespaced attributes using lxml efficiently.

    :param element: The XML element
    :param attr_name: The attribute name (e.g., 'Id', 'Type', 'Target')
    :return: The attribute value or empty string if not found
    """
    return element.get(attr_name, "")


def _docx_xpath(element: etree._Element, expression: str) -> list[etree._Element]:
    """
    Helper function to perform XPath queries with Word XML namespaces.

    Simplifies the common pattern of xpath calls with WORD_XML_NAMESPACES by providing
    a consistent interface that automatically includes the required namespace mappings.

    :param element: The XML element to query
    :param expression: XPath expression string (e.g., "w:pPr", ".//w:r")
    :return: List of matching elements
    """
    return element.xpath(expression, namespaces=WORD_XML_NAMESPACES)


def _extract_footnote_id_from_context(context: str) -> str:
    """
    Extracts footnote ID from paragraph context string.

    :param context: Context string containing footnote information
    :return: Footnote ID
    """
    return context.split("Footnote ID: ")[1].split(",")[0]


def _extract_comment_id_from_context(context: str) -> str:
    """
    Extracts comment ID and author from paragraph context string.

    :param context: Context string containing comment information
    :return: Formatted comment ID with author if available
    """
    comment_id = context.split("Comment ID: ")[1].split(",")[0]
    author = ""
    if "Author: " in context:
        author = context.split("Author: ")[1].split(",")[0]
        author = f" (by {author})"
    return f"{comment_id}{author}"


def _join_text_parts(parts: list[str]) -> str:
    """
    Joins text parts, filtering out empty ones.

    :param parts: List of text parts to join
    :return: Joined text
    """
    return "".join(part for part in parts if part)
